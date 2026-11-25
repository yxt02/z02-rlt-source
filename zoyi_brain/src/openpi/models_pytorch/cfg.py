"""CFG / RECAP routing and metrics for advantage-conditioned training (PyTorch side).

Torch mirror of `openpi.models.cfg` — same key names and semantics as RLinf
(fa1579c). Must not import JAX; `CFG_RAW_KEYS` is kept in sync with the JAX module
by `openpi.models_pytorch.cfg_test`.
"""

import torch

# The 14 raw per-step metrics: 8 counts + 6 masked loss sums. SUM-aggregated across
# steps/devices; ratios derived afterwards via `openpi.models.cfg.derive_cfg_ratios`.
CFG_RAW_KEYS = (
    "conditional_count",
    "unconditional_count",
    "positive_label_count",
    "negative_label_count",
    "positive_conditional_count",
    "positive_unconditional_count",
    "negative_conditional_count",
    "negative_unconditional_count",
    "conditional_loss_sum",
    "unconditional_loss_sum",
    "positive_conditional_loss_sum",
    "positive_unconditional_loss_sum",
    "negative_conditional_loss_sum",
    "negative_unconditional_loss_sum",
)


def compute_routing_masks(
    advantage: torch.Tensor,
    *,
    positive_only_conditional: bool,
    unconditional_prob: float,
    random_values: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Compute the 7 RLinf sample-routing masks for CFG training.

    Args:
        advantage: Boolean tensor where True marks positive samples.
        positive_only_conditional: Route only positive samples to the conditional branch.
        unconditional_prob: Dropout probability p for unconditional routing. When
            `positive_only_conditional` is True, applies only to positive samples.
        random_values: Optional pre-sampled uniforms in [0, 1) for deterministic tests.

    Returns:
        Dict of boolean masks: positive/negative/conditional plus the four
        {positive,negative} x {conditional,unconditional} cells.
    """
    advantage = advantage.to(dtype=torch.bool)
    batch_size = advantage.shape[0]
    device = advantage.device

    random_values = torch.rand(batch_size, device=device) if random_values is None else random_values.to(device=device)

    positive_mask = advantage
    negative_mask = ~positive_mask

    if positive_only_conditional:
        positive_conditional_mask = positive_mask & (random_values > unconditional_prob)
        negative_conditional_mask = torch.zeros_like(positive_mask)
    else:
        guidance_mask = random_values > unconditional_prob
        positive_conditional_mask = positive_mask & guidance_mask
        negative_conditional_mask = negative_mask & guidance_mask

    conditional_mask = positive_conditional_mask | negative_conditional_mask
    positive_unconditional_mask = positive_mask & ~positive_conditional_mask
    negative_unconditional_mask = negative_mask & ~negative_conditional_mask

    return {
        "positive_mask": positive_mask,
        "negative_mask": negative_mask,
        "conditional_mask": conditional_mask,
        "positive_conditional_mask": positive_conditional_mask,
        "positive_unconditional_mask": positive_unconditional_mask,
        "negative_conditional_mask": negative_conditional_mask,
        "negative_unconditional_mask": negative_unconditional_mask,
    }


def select_guidance_tokens(observation, routing: dict[str, torch.Tensor], *, positive_only_conditional: bool):
    """Pick the per-sample prompt tokens/masks according to the routing masks.

    Mirrors RLinf's prompt selection: positive-only mode uses the positive guidance
    prompt only for `positive_conditional_mask`; full mode first picks the guidance
    prompt by label, then swaps to it where `conditional_mask` is set.

    Returns:
        (tokenized_prompt, tokenized_prompt_mask) to feed into the prefix embedding.
    """
    orig_tokens = observation.tokenized_prompt
    orig_masks = observation.tokenized_prompt_mask
    pos_tokens = observation.tokenized_positive_guidance_prompt
    pos_masks = observation.tokenized_positive_guidance_prompt_mask
    neg_tokens = observation.tokenized_negative_guidance_prompt
    neg_masks = observation.tokenized_negative_guidance_prompt_mask
    if pos_tokens is None or pos_masks is None or neg_tokens is None or neg_masks is None:
        raise ValueError(
            "Observation is missing guidance prompt fields "
            "(tokenized_{positive,negative}_guidance_prompt[_mask]). "
            "Ensure TokenizePromptWithGuidance is in the model transforms and "
            "cfg_guidance is set on the model config (inputs_spec)."
        )

    if positive_only_conditional:
        sel = routing["positive_conditional_mask"]
        tokens = torch.where(sel[:, None], pos_tokens, orig_tokens)
        masks = torch.where(sel[:, None], pos_masks, orig_masks)
    else:
        positive = routing["positive_mask"]
        conditional = routing["conditional_mask"]
        guidance_tokens = torch.where(positive[:, None], pos_tokens, neg_tokens)
        guidance_masks = torch.where(positive[:, None], pos_masks, neg_masks)
        tokens = torch.where(conditional[:, None], guidance_tokens, orig_tokens)
        masks = torch.where(conditional[:, None], guidance_masks, orig_masks)
    return tokens, masks


def compute_cfg_metrics(routing: dict[str, torch.Tensor], per_sample_loss: torch.Tensor) -> dict[str, torch.Tensor]:
    """Compute the 14 raw CFG metrics (8 counts + 6 masked loss sums) as detached
    float32 scalar tensors, ready for `dist.all_reduce(SUM)`.

    `per_sample_loss` is the detached per-sample mean loss, shape [B].
    """
    loss = per_sample_loss.detach().to(dtype=torch.float32)

    def count(mask: torch.Tensor) -> torch.Tensor:
        return mask.to(torch.float32).sum()

    def loss_sum(mask: torch.Tensor) -> torch.Tensor:
        return (loss * mask.to(torch.float32)).sum()

    positive = routing["positive_mask"]
    negative = routing["negative_mask"]
    conditional = routing["conditional_mask"]
    pos_cond = routing["positive_conditional_mask"]
    pos_uncond = routing["positive_unconditional_mask"]
    neg_cond = routing["negative_conditional_mask"]
    neg_uncond = routing["negative_unconditional_mask"]

    return {
        "conditional_count": count(conditional),
        "unconditional_count": count(~conditional),
        "positive_label_count": count(positive),
        "negative_label_count": count(negative),
        "positive_conditional_count": count(pos_cond),
        "positive_unconditional_count": count(pos_uncond),
        "negative_conditional_count": count(neg_cond),
        "negative_unconditional_count": count(neg_uncond),
        "conditional_loss_sum": loss_sum(conditional),
        "unconditional_loss_sum": loss_sum(~conditional),
        "positive_conditional_loss_sum": loss_sum(pos_cond),
        "positive_unconditional_loss_sum": loss_sum(pos_uncond),
        "negative_conditional_loss_sum": loss_sum(neg_cond),
        "negative_unconditional_loss_sum": loss_sum(neg_uncond),
    }
