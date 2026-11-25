"""CFG / RECAP routing and metrics for advantage-conditioned training (JAX side).

Semantics mirror RLinf's `compute_cfg_routing_masks`, prompt selection and
`FSDPCfgWorker` metric aggregation (RLinf fa1579c). The PyTorch mirror lives in
`openpi.models_pytorch.cfg` and must keep `CFG_RAW_KEYS` in sync (enforced by test).
"""

import jax
import jax.numpy as jnp

import openpi.shared.array_typing as at

# The 14 raw per-step metrics: 8 counts + 6 masked loss sums. These are SUM-aggregated
# across steps/devices; ratios are derived afterwards via `derive_cfg_ratios`.
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


def draw_routing_random(rng: at.KeyArrayLike, batch_size: int) -> at.Float[at.Array, "*b"]:
    """Draw the per-sample routing uniforms for one forward pass.

    Uses `fold_in(rng, 3)`: the `split(rng, 3)` children used for preprocess/noise/time
    are `fold_in(rng, 0..2)`, so datum 3 gives a distinct stream without perturbing
    those three (they stay bit-identical to non-CFG runs). Test:
    `cfg_test.test_draw_routing_random_determinism_and_range`.
    """
    return jax.random.uniform(jax.random.fold_in(rng, 3), (batch_size,))


def compute_routing_masks(
    advantage: at.Bool[at.Array, "*b"],
    random_values: at.Float[at.Array, "*b"],
    *,
    positive_only_conditional: bool,
    unconditional_prob: float,
) -> dict[str, at.Bool[at.Array, "*b"]]:
    """Compute the 7 RLinf sample-routing masks for CFG training.

    Args:
        advantage: Boolean per-sample label; True marks positive samples.
        random_values: Pre-sampled uniforms in [0, 1), normally from `draw_routing_random`.
        positive_only_conditional: Route only positive samples to the conditional branch.
        unconditional_prob: Dropout probability p for unconditional routing. When
            `positive_only_conditional` is True, applies only to positive samples.

    Returns:
        Dict of boolean masks: positive/negative/conditional plus the four
        {positive,negative} x {conditional,unconditional} cells.
    """
    advantage = advantage.astype(jnp.bool_)
    random_values = jnp.asarray(random_values)

    positive_mask = advantage
    negative_mask = ~positive_mask

    if positive_only_conditional:
        positive_conditional_mask = positive_mask & (random_values > unconditional_prob)
        negative_conditional_mask = jnp.zeros_like(positive_mask)
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


def select_guidance_tokens(
    observation, routing: dict[str, at.Bool[at.Array, "*b"]], *, positive_only_conditional: bool
):
    """Pick the per-sample prompt tokens/masks according to the routing masks.

    Mirrors RLinf's prompt selection: positive-only mode uses the positive guidance
    prompt only for `positive_conditional_mask`; full mode first picks the guidance
    prompt by label, then swaps to it where `conditional_mask` is set.

    Returns:
        (tokenized_prompt, tokenized_prompt_mask) to install on the observation.
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
        tokens = jnp.where(sel[:, None], pos_tokens, orig_tokens)
        masks = jnp.where(sel[:, None], pos_masks, orig_masks)
    else:
        positive = routing["positive_mask"]
        conditional = routing["conditional_mask"]
        guidance_tokens = jnp.where(positive[:, None], pos_tokens, neg_tokens)
        guidance_masks = jnp.where(positive[:, None], pos_masks, neg_masks)
        tokens = jnp.where(conditional[:, None], guidance_tokens, orig_tokens)
        masks = jnp.where(conditional[:, None], guidance_masks, orig_masks)
    return tokens, masks


def compute_cfg_metrics(
    routing: dict[str, at.Bool[at.Array, "*b"]],
    per_sample_loss: at.Float[at.Array, "*b"],
) -> dict[str, at.Float[at.Array, ""]]:
    """Compute the 14 raw CFG metrics (8 counts + 6 masked loss sums), float32 scalars.

    `per_sample_loss` is the detached per-sample mean loss, shape [B].
    """
    loss = per_sample_loss.astype(jnp.float32)

    def count(mask):
        return mask.astype(jnp.float32).sum()

    def loss_sum(mask):
        return (loss * mask.astype(jnp.float32)).sum()

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


def derive_cfg_ratios(raw: dict[str, float]) -> dict[str, float]:
    """Derive display ratios from raw CFG counts/sums, following RLinf's aggregation.

    Accepts either window means or window sums of the raw keys (ratios are invariant).
    Keys with zero denominators are skipped; result contains only finite entries.
    """

    def get(key: str) -> float:
        return float(raw.get(key, 0.0))

    derived: dict[str, float] = {}
    total = get("conditional_count") + get("unconditional_count")
    if total > 0:
        derived["conditional_ratio"] = get("conditional_count") / total
        derived["unconditional_ratio"] = get("unconditional_count") / total
        derived["positive_label_ratio"] = get("positive_label_count") / total
        derived["negative_label_ratio"] = get("negative_label_count") / total
        derived["positive_conditional_ratio"] = get("positive_conditional_count") / total
        derived["positive_unconditional_ratio"] = get("positive_unconditional_count") / total
        derived["negative_conditional_ratio"] = get("negative_conditional_count") / total
        derived["negative_unconditional_ratio"] = get("negative_unconditional_count") / total

    positive_total = get("positive_label_count")
    if positive_total > 0:
        derived["positive_effective_conditional_ratio"] = get("positive_conditional_count") / positive_total
        derived["positive_effective_unconditional_ratio"] = get("positive_unconditional_count") / positive_total

    negative_total = get("negative_label_count")
    if negative_total > 0:
        derived["negative_effective_conditional_ratio"] = get("negative_conditional_count") / negative_total
        derived["negative_effective_unconditional_ratio"] = get("negative_unconditional_count") / negative_total

    loss_map = {
        "conditional_loss": ("conditional_loss_sum", "conditional_count"),
        "unconditional_loss": ("unconditional_loss_sum", "unconditional_count"),
        "positive_conditional_loss": ("positive_conditional_loss_sum", "positive_conditional_count"),
        "positive_unconditional_loss": ("positive_unconditional_loss_sum", "positive_unconditional_count"),
        "negative_conditional_loss": ("negative_conditional_loss_sum", "negative_conditional_count"),
        "negative_unconditional_loss": ("negative_unconditional_loss_sum", "negative_unconditional_count"),
    }
    for metric_name, (loss_key, count_key) in loss_map.items():
        count = get(count_key)
        if count > 0:
            derived[metric_name] = get(loss_key) / count

    return derived
