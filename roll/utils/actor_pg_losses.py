from typing import Dict, Optional, Tuple

import torch

from roll.utils.functionals import agg_loss, masked_mean


def _mean_with_optional_mask(values: torch.Tensor, mask: Optional[torch.Tensor]) -> float:
    if mask is None:
        return values.detach().mean().item()
    return masked_mean(values.detach(), mask).detach().item()


def compute_ppo_pg_loss_mat_and_metrics(
    *,
    ratio: torch.Tensor,
    advantages: torch.Tensor,
    pg_clip: float,
    pg_clip_low: float,
    pg_clip_high: float,
    use_pg_clip_range: bool,
    dual_clip_loss_enabled: bool,
    loss_mask: torch.Tensor,
    loss_agg_mode: str,
    clipfrac_mask: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Compute PPO-style clipped policy gradient loss matrix and metrics.

    Returns:
        - pg_loss_mat: token-level loss matrix (same shape as ratio/advantages).
        - metrics: dict containing PPO-specific metrics (unprefixed).
    """
    pg_clip_low_eff = pg_clip_low if use_pg_clip_range else pg_clip
    pg_clip_high_eff = pg_clip_high if use_pg_clip_range else pg_clip

    surr1 = ratio * advantages
    surr2 = ratio.clamp(1 - pg_clip_low_eff, 1 + pg_clip_high_eff) * advantages
    pg_loss_mat = -torch.min(surr1, surr2)
    if dual_clip_loss_enabled:
        dual_clip_loss = -torch.max(-pg_loss_mat, (1 + pg_clip * 2) * advantages)
        pg_loss_mat = torch.where(advantages < 0, dual_clip_loss, pg_loss_mat)

    # Metrics should not build autograd graphs.
    with torch.no_grad():
        clipped_low = (ratio < 1 - pg_clip_low_eff).float()
        clipped_high = (ratio > 1 + pg_clip_high_eff).float()
        clipped = (clipped_low + clipped_high).float()

        metrics = {
            "ppo_ratio_high_clipfrac": _mean_with_optional_mask(clipped_high, clipfrac_mask),
            "ppo_ratio_low_clipfrac": _mean_with_optional_mask(clipped_low, clipfrac_mask),
            "ppo_ratio_clipfrac": _mean_with_optional_mask(clipped, clipfrac_mask),
            "clipfrac": agg_loss(
                loss_mat=torch.lt(surr2, surr1).float(),
                loss_mask=loss_mask,
                loss_agg_mode=loss_agg_mode,
            ).item(),
        }
    return pg_loss_mat, metrics


def compute_tis_pg_loss_mat_and_metrics(
    *,
    ratio: torch.Tensor,
    log_probs: torch.Tensor,
    advantages: torch.Tensor,
    tis_lower_bound: float,
    tis_upper_bound: float,
    loss_mask: torch.Tensor,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    # TIS uses a detached/clipped importance ratio by design; gradients should still flow via log_probs.
    clipped_ratio = torch.clamp(ratio, min=tis_lower_bound, max=tis_upper_bound).detach()
    pg_loss_mat = -(clipped_ratio * advantages * log_probs)

    with torch.no_grad():
        lower_clipped = (ratio < tis_lower_bound).float()
        upper_clipped = (ratio > tis_upper_bound).float()
        total_clipped = lower_clipped + upper_clipped

        metrics = {
            "tis_lower_bound": float(tis_lower_bound),
            "tis_upper_bound": float(tis_upper_bound),
            "tis_lower_clipfrac": masked_mean(lower_clipped, loss_mask).item(),
            "tis_upper_clipfrac": masked_mean(upper_clipped, loss_mask).item(),
            "tis_total_clipfrac": masked_mean(total_clipped, loss_mask).item(),
            "tis_clipped_ratio_mean": masked_mean(clipped_ratio, loss_mask).item(),
        }
    return pg_loss_mat, metrics


def compute_topr_pg_loss_mat_and_metrics(
    *,
    ratio: torch.Tensor,
    log_probs: torch.Tensor,
    advantages: torch.Tensor,
    scores: torch.Tensor,
    topr_positive_weight: float,
    topr_negative_weight: float,
    loss_mask: torch.Tensor,
    loss_agg_mode: str,
    sample_mask: Optional[torch.Tensor],
    clipfrac_mask: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, Dict[str, float]]:
    if sample_mask is None:
        sample_mask = torch.ones(scores.shape[0], dtype=torch.bool, device=scores.device)

    positive_mask = (scores > 0).float()
    negative_mask = (scores <= 0).float()

    positive_token_mask = positive_mask.unsqueeze(-1).expand_as(log_probs)
    negative_token_mask = negative_mask.unsqueeze(-1).expand_as(log_probs)

    positive_loss = -advantages * log_probs * positive_token_mask
    clipped_ratio = torch.clamp(ratio, min=0.0, max=1.0).detach()
    negative_loss = -clipped_ratio * advantages * log_probs * negative_token_mask

    weighted_positive_loss = float(topr_positive_weight) * positive_loss
    weighted_negative_loss = float(topr_negative_weight) * negative_loss
    pg_loss_mat = weighted_positive_loss + weighted_negative_loss

    with torch.no_grad():
        sample_count = float(sample_mask.sum().item())
        positive_count = float(positive_mask[sample_mask].sum().item())
        negative_count = float(negative_mask[sample_mask].sum().item())

        negative_token_mask_bool = negative_token_mask > 0
        negative_lower_clipped = ((ratio < 0.0) & negative_token_mask_bool).float()
        negative_upper_clipped = ((ratio > 1.0) & negative_token_mask_bool).float()
        negative_total_clipped = negative_lower_clipped + negative_upper_clipped

        metrics = {
            "topr_positive_loss": agg_loss(
                loss_mat=positive_loss, loss_mask=loss_mask, loss_agg_mode=loss_agg_mode
            ).item(),
            "topr_negative_loss": agg_loss(
                loss_mat=negative_loss, loss_mask=loss_mask, loss_agg_mode=loss_agg_mode
            ).item(),
            "topr_weighted_positive_loss": agg_loss(
                loss_mat=weighted_positive_loss,
                loss_mask=loss_mask,
                loss_agg_mode=loss_agg_mode,
            ).item(),
            "topr_weighted_negative_loss": agg_loss(
                loss_mat=weighted_negative_loss,
                loss_mask=loss_mask,
                loss_agg_mode=loss_agg_mode,
            ).item(),
            "topr_positive_samples": positive_count,
            "topr_negative_samples": negative_count,
            "topr_positive_ratio": (positive_count / (sample_count + 1e-8)) if sample_count > 0 else 0.0,
            "topr_negative_ratio": (negative_count / (sample_count + 1e-8)) if sample_count > 0 else 0.0,
            "topr_negative_lower_clipfrac": _mean_with_optional_mask(negative_lower_clipped, clipfrac_mask),
            "topr_negative_upper_clipfrac": _mean_with_optional_mask(negative_upper_clipped, clipfrac_mask),
            "topr_negative_total_clipfrac": _mean_with_optional_mask(negative_total_clipped, clipfrac_mask),
            "topr_scores_mean": scores[sample_mask].mean().item() if sample_mask.any() else 0.0,
            "topr_scores_std": scores[sample_mask].std(unbiased=False).item() if sample_mask.any() else 0.0,
            "topr_positive_weight": float(topr_positive_weight),
            "topr_negative_weight": float(topr_negative_weight),
        }
    return pg_loss_mat, metrics


def compute_cispo_pg_loss_mat_and_metrics(
    *,
    ratio: torch.Tensor,
    log_probs: torch.Tensor,
    advantages: torch.Tensor,
    epsilon_low: float,
    epsilon_high: float,
    use_unified_mask: bool,
    loss_mask: torch.Tensor,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    clip_lower = 1.0 - float(epsilon_low)
    clip_upper = 1.0 + float(epsilon_high)

    clipped_ratio = torch.clamp(ratio, min=clip_lower, max=clip_upper).detach()

    if use_unified_mask:
        with torch.no_grad():
            positive_advantages = advantages > 0
            negative_advantages = advantages < 0
            mask_positive = positive_advantages & (ratio > clip_upper)
            mask_negative = negative_advantages & (ratio < clip_lower)
            token_mask_unified = ~(mask_positive | mask_negative)
        pg_loss_mat = -(clipped_ratio * advantages * log_probs * token_mask_unified.float())
    else:
        mask_positive = None
        mask_negative = None
        token_mask_unified = None
        pg_loss_mat = -(clipped_ratio * advantages * log_probs)

    with torch.no_grad():
        lower_clipped = (ratio < clip_lower).float()
        upper_clipped = (ratio > clip_upper).float()
        total_clipped = lower_clipped + upper_clipped

        metrics = {
            "cispo_epsilon_low": float(epsilon_low),
            "cispo_epsilon_high": float(epsilon_high),
            "cispo_clip_lower": float(clip_lower),
            "cispo_clip_upper": float(clip_upper),
            "cispo_use_unified_mask": float(use_unified_mask),
            "cispo_lower_clipfrac": masked_mean(lower_clipped, loss_mask).item(),
            "cispo_upper_clipfrac": masked_mean(upper_clipped, loss_mask).item(),
            "cispo_total_clipfrac": masked_mean(total_clipped, loss_mask).item(),
            "cispo_clipped_ratio_mean": masked_mean(clipped_ratio, loss_mask).item(),
        }
        if use_unified_mask:
            assert mask_positive is not None and mask_negative is not None and token_mask_unified is not None
            metrics.update(
                {
                    "cispo_masked_positive_tokens": masked_mean(mask_positive.float(), loss_mask).item(),
                    "cispo_masked_negative_tokens": masked_mean(mask_negative.float(), loss_mask).item(),
                    "cispo_kept_tokens": masked_mean(token_mask_unified.float(), loss_mask).item(),
                }
            )
    return pg_loss_mat, metrics
