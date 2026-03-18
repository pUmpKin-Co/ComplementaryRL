from dataclasses import dataclass
from typing import Callable, Optional, Unpack

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_outputs import BaseModelOutput
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
from transformers.models.qwen3.modeling_qwen3 import (ALL_ATTENTION_FUNCTIONS,
                                                      Cache,
                                                      FlashAttentionKwargs,
                                                      Qwen3Attention,
                                                      Qwen3Model, Qwen3RMSNorm,
                                                      apply_rotary_pos_emb)

# TODO: Should we also adjust the RMSNorm?

@dataclass
class LatentOutput(BaseModelOutput):
    latent_memory: torch.Tensor


class Qwen3LatentAttention(Qwen3Attention):
    """
    Custom attention that skips RoPE for latent tokens.
    Expects 'full_position_ids' in kwargs where values < 0 indicate latent tokens.
    Note: actual position_ids passed to RoPE use valid values (e.g., 0) for latents.
    """
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        
        # Check if we have position_ids to identify latent tokens
        full_position_ids = kwargs.get('full_position_ids', None)
        
        if full_position_ids is not None:
            # Latent tokens are marked with position_id < 0
            # Only apply RoPE to non-latent tokens (position_id >= 0)
            non_latent_mask = full_position_ids[0] >= 0  # [seq_len]
            latent_mask = full_position_ids[0] < 0  # [seq_len]
            
            if non_latent_mask.any():
                # Apply RoPE only to non-latent positions
                if latent_mask.any():
                    # Mixed case: some latents, some content
                    # Clone to avoid in-place modification issues
                    query_states_rotated = query_states.clone()
                    key_states_rotated = key_states.clone()
                    
                    # Apply RoPE only to non-latent positions
                    non_latent_indices = torch.where(non_latent_mask)[0]
                    q_content = query_states[:, :, non_latent_indices, :]
                    k_content = key_states[:, :, non_latent_indices, :]
                    cos_content = cos[:, non_latent_indices, :]
                    sin_content = sin[:, non_latent_indices, :]
                    
                    q_content, k_content = apply_rotary_pos_emb(
                        q_content, k_content, cos_content, sin_content
                    )
                    
                    query_states_rotated[:, :, non_latent_indices, :] = q_content
                    key_states_rotated[:, :, non_latent_indices, :] = k_content
                    
                    query_states = query_states_rotated
                    key_states = key_states_rotated
                else:
                    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        else:
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

        attention_interface: Callable = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=self.sliding_window,
            **{k: v for k, v in kwargs.items() if k != 'full_position_ids'},  # Remove custom kwarg
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class QwenLatentModelConfig(Qwen3Config):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.latent_size = kwargs.get("latent_size", 128)


class QwenLatentModel(Qwen3Model):
    config: QwenLatentModelConfig
    config_class = QwenLatentModelConfig

    def __init__(self, config: QwenLatentModelConfig, **kwargs):
        super().__init__(config, **kwargs)
        self.latent_tokens = nn.Parameter(
            torch.randn(self.config.latent_size, self.config.hidden_size),
            requires_grad=True,
        )
        
        for layer_idx, layer in enumerate(self.layers):
            original_attn_state = layer.self_attn.state_dict()        
            custom_attn = Qwen3LatentAttention(config, layer_idx)
            custom_attn.load_state_dict(original_attn_state)
            layer.self_attn = custom_attn
    
    @property
    def latent_size(self):
        return self.config.latent_size
    
    def prepare_for_forward(self, input_ids, attention_mask, **kwargs):
        if kwargs.get("inputs_embeds", None) is not None and input_ids is not None:
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if input_ids is not None:
            inputs_embeds = self.embed_tokens(input_ids)
        else:
            inputs_embeds = kwargs["inputs_embeds"]
        
        bsz, seq_len = inputs_embeds.shape[:2]
        content_position_ids = torch.arange(seq_len, device=inputs_embeds.device, dtype=torch.long).unsqueeze(0)

        latents = self.latent_tokens.unsqueeze(0).repeat(bsz, 1, 1)
        post_latents_mask = torch.ones(latents.shape[:-1], dtype=attention_mask.dtype, device=inputs_embeds.device)
        post_latents_position_ids = torch.zeros(
            (1, self.config.latent_size), device=inputs_embeds.device, dtype=torch.long
        )
        
        attention_mask = torch.cat([attention_mask, post_latents_mask], dim=1)
        inputs_embeds = torch.cat([inputs_embeds, latents], dim=1)
        position_ids = torch.cat([content_position_ids, post_latents_position_ids], dim=1)
        
        full_position_ids = torch.cat([
            content_position_ids,
            torch.full((1, self.config.latent_size), -1, device=inputs_embeds.device, dtype=torch.long)
        ], dim=1)

        if kwargs.get("last_turn_latent", None) is not None:
            last_turn_latent = kwargs["last_turn_latent"]
            pre_latents_mask = torch.ones(last_turn_latent.shape[:-1], dtype=attention_mask.dtype, device=inputs_embeds.device)
            pre_latents_position_ids = torch.zeros(
                (1, last_turn_latent.shape[1]), device=inputs_embeds.device, dtype=torch.long
            )
            
            attention_mask = torch.cat([pre_latents_mask, attention_mask], dim=1)
            inputs_embeds = torch.cat([last_turn_latent, inputs_embeds], dim=1)
            position_ids = torch.cat([pre_latents_position_ids, position_ids], dim=1)
            
            full_position_ids = torch.cat([
                torch.full((1, last_turn_latent.shape[1]), -1, device=inputs_embeds.device, dtype=torch.long),
                full_position_ids
            ], dim=1)

        return inputs_embeds, attention_mask, position_ids, full_position_ids
        

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ):
        inputs_embeds, attention_mask, position_ids, full_position_ids = self.prepare_for_forward(
            input_ids, attention_mask, **kwargs
        )
        
        kwargs['full_position_ids'] = full_position_ids
        
        outputs = super().forward(
            input_ids=None,  # We're using inputs_embeds instead
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            **kwargs,
        )

        latent_memory = outputs.last_hidden_state[:, -self.config.latent_size:, :]
        return LatentOutput(
            last_hidden_state=outputs.last_hidden_state,
            latent_memory=latent_memory,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


AutoConfig.register("Qwen3LatentMemory", QwenLatentModelConfig)
AutoModel.register(QwenLatentModelConfig, QwenLatentModel)