from typing import List, Union

import numpy as np
import torch
from tensordict import TensorDict

from roll.configs.worker_config import WorkerConfig
from roll.distributed.executor.worker import Worker
from roll.distributed.scheduler.decorator import Dispatch, register
from roll.distributed.scheduler.protocol import DataProto
from roll.distributed.strategy.factory import create_strategy
from roll.models.model_providers import default_embedding_model_provider
from roll.pipeline.agentic.memory.memory_config import PoolType
from roll.utils.checkpoint_manager import download_model
from roll.utils.logging import get_logger

logger = get_logger()


class EmbeddingWorker(Worker):
    """
    Strategy for managing embedding models in a distributed setting.
    """

    def __init__(self, worker_config: WorkerConfig):
        """
        Initialize embedding strategy.
        """
        super().__init__(worker_config=worker_config)
        self.logger.info(
            f"EmbeddingWorker initialized on rank {self.rank}/{self.world_size}"
        )

    @register(dispatch_mode=Dispatch.ONE_TO_ALL, clear_cache=False)
    def initialize(self):
        """
        Initialize the embedding model and tokenizer.
        """
        strategy_name = self.worker_config.strategy_args.strategy_name
        assert strategy_name in [
            "hf_infer",
            "vllm",
        ], "EmbeddingWorker only supports hf_infer or vllm strategy now"
        self.strategy = create_strategy(worker=self)
        self.worker_config.model_args.model_name_or_path = download_model(self.worker_config.model_args.model_name_or_path)
        self.strategy.initialize(
            model_provider=default_embedding_model_provider
        )
        self.tokenizer = self.strategy.tokenizer
        self.logger.info(
            f"EmbeddingWorker initialized on rank {self.rank}/{self.world_size}"
        )

    def encode(self, texts: Union[str, List[str]]) -> torch.Tensor:
        """
        Encode texts into embeddings using strategy pattern.

        Args:
            texts: Single text or list of texts to encode

        Returns:
            torch.Tensor: Embeddings of shape (len(texts), embedding_dim)
        """
        if self.strategy is None:
            raise RuntimeError(
                "Strategy not initialized. Call initialize() first."
            )

        if isinstance(texts, str):
            texts = [texts]

        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )

        batch = TensorDict(
            source={
                "input_ids": encoded["input_ids"],
                "attention_mask": encoded["attention_mask"],
            },
            batch_size=(len(texts),),
        )

        data_proto = DataProto(
            batch=batch,
            non_tensor_batch={
                "texts": np.array(texts, dtype=object),
            },
            meta_info={
                "micro_batch_size": len(texts),
                "num_texts": len(texts),
            },
        )

        data_proto = data_proto.to("cuda")

        if self.worker_config.pool_type == PoolType.cls:
            pool_func = self._cls_pooling
        elif self.worker_config.pool_type == PoolType.last_token:
            pool_func = self._last_tokon_pooling
        elif self.worker_config.pool_type == PoolType.mean:
            pool_func = self._mean_pooling
        else:
            raise ValueError(
                f"Unsupported pool type: {self.worker_config.pool_type}"
            )

        embeddings = self.strategy.embed(data_proto, pool_func)
        assert embeddings.dim() == 2 and embeddings.shape[0] == len(
            texts
        ), f"Embeddings shape: {embeddings.shape}"

        return embeddings.cpu()

    def _mean_pooling(self, last_hidden_state, attention_mask):
        """
        Mean pooling with attention mask.
        Takes the mean of the last hidden state, weighted by attention mask.
        """
        token_embeddings = last_hidden_state
        input_mask_expanded = (
            attention_mask.unsqueeze(-1)
            .expand(token_embeddings.size())
            .float()
        )
        return torch.sum(
            token_embeddings * input_mask_expanded, 1
        ) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def _last_tokon_pooling(self, last_hidden_state, attention_mask):
        """
        Last token pooling with attention mask.
        """
        left_padding = (
            attention_mask[:, -1].sum() == attention_mask.shape[0]
        )
        if left_padding:
            return last_hidden_state[:, -1]
        else:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden_state.shape[0]
            return last_hidden_state[
                torch.arange(batch_size, device=last_hidden_state.device),
                sequence_lengths,
            ]

    def _cls_pooling(self, last_hidden_state, attention_mask):
        """
        CLS token pooling.
        """
        return last_hidden_state[:, 0]
