from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest
import torch

from openpi.models import cfg as _jax_cfg
from openpi.models_pytorch import cfg as _torch_cfg

_LABELS = [True, True, False, False]
_RANDOM_VALUES = [0.5, 0.05, 0.5, 0.05]
_P = 0.1


def test_raw_keys_in_sync_with_jax():
    # Drift guard: the torch mirror must expose the exact same 14 raw metric keys.
    assert _torch_cfg.CFG_RAW_KEYS == _jax_cfg.CFG_RAW_KEYS


def _torch_routing(*, positive_only: bool, labels=_LABELS, random_values=_RANDOM_VALUES, p=_P):
    return _torch_cfg.compute_routing_masks(
        torch.tensor(labels),
        positive_only_conditional=positive_only,
        unconditional_prob=p,
        random_values=torch.tensor(random_values, dtype=torch.float32),
    )


def _jax_routing(*, positive_only: bool, labels=_LABELS, random_values=_RANDOM_VALUES, p=_P):
    return _jax_cfg.compute_routing_masks(
        jnp.asarray(labels),
        jnp.asarray(random_values),
        positive_only_conditional=positive_only,
        unconditional_prob=p,
    )


@pytest.mark.parametrize("positive_only", [True, False])
def test_routing_masks_match_jax(positive_only):
    torch_masks = _torch_routing(positive_only=positive_only)
    jax_masks = _jax_routing(positive_only=positive_only)

    assert torch_masks.keys() == jax_masks.keys()
    for key in jax_masks:
        np.testing.assert_array_equal(torch_masks[key].numpy(), np.asarray(jax_masks[key]), err_msg=key)


def test_routing_masks_positive_only_table():
    routing = _torch_routing(positive_only=True)
    np.testing.assert_array_equal(routing["positive_conditional_mask"].numpy(), [True, False, False, False])
    np.testing.assert_array_equal(routing["negative_conditional_mask"].numpy(), [False, False, False, False])
    np.testing.assert_array_equal(routing["conditional_mask"].numpy(), [True, False, False, False])
    np.testing.assert_array_equal(routing["positive_unconditional_mask"].numpy(), [False, True, False, False])
    np.testing.assert_array_equal(routing["negative_unconditional_mask"].numpy(), [False, False, True, True])


def _synthetic_observation(batch_size: int = 4, seq_len: int = 5):
    return SimpleNamespace(
        tokenized_prompt=torch.full((batch_size, seq_len), 11, dtype=torch.int32),
        tokenized_prompt_mask=torch.zeros((batch_size, seq_len), dtype=torch.bool),
        tokenized_positive_guidance_prompt=torch.full((batch_size, seq_len), 22, dtype=torch.int32),
        tokenized_positive_guidance_prompt_mask=torch.ones((batch_size, seq_len), dtype=torch.bool),
        tokenized_negative_guidance_prompt=torch.full((batch_size, seq_len), 33, dtype=torch.int32),
        tokenized_negative_guidance_prompt_mask=torch.zeros((batch_size, seq_len), dtype=torch.bool),
    )


def test_select_guidance_tokens_positive_only():
    obs = _synthetic_observation()
    routing = _torch_routing(positive_only=True)

    tokens, masks = _torch_cfg.select_guidance_tokens(obs, routing, positive_only_conditional=True)

    # Row 0 is positive-conditional -> positive guidance (22); all others -> original (11).
    np.testing.assert_array_equal(tokens[0].numpy(), np.full((5,), 22, np.int32))
    np.testing.assert_array_equal(tokens[1:].numpy(), np.full((3, 5), 11, np.int32))
    np.testing.assert_array_equal(masks[0].numpy(), np.ones((5,), bool))
    np.testing.assert_array_equal(masks[1:].numpy(), np.zeros((3, 5), bool))


def test_select_guidance_tokens_full_mode():
    obs = _synthetic_observation()
    routing = _torch_routing(positive_only=False)

    tokens, masks = _torch_cfg.select_guidance_tokens(obs, routing, positive_only_conditional=False)

    # Same table as the JAX test: conditional [T, F, T, F] — row 3 (neg, r=0.05 <= p)
    # stays on the original prompt.
    expected = np.full((4, 5), 11, np.int32)
    expected[0] = 22  # positive, conditional
    expected[2] = 33  # negative, conditional
    np.testing.assert_array_equal(tokens.numpy(), expected)
    expected_masks = np.zeros((4, 5), bool)
    expected_masks[0] = True  # only the positive mask is ones in this fixture
    np.testing.assert_array_equal(masks.numpy(), expected_masks)


def test_select_guidance_tokens_missing_fields():
    obs = SimpleNamespace(
        tokenized_prompt=torch.zeros((2, 5), dtype=torch.int32),
        tokenized_prompt_mask=torch.ones((2, 5), dtype=torch.bool),
        tokenized_positive_guidance_prompt=None,
        tokenized_positive_guidance_prompt_mask=None,
        tokenized_negative_guidance_prompt=None,
        tokenized_negative_guidance_prompt_mask=None,
    )
    routing = _torch_routing(positive_only=True, labels=[True, False], random_values=[0.5, 0.5])

    with pytest.raises(ValueError, match="guidance"):
        _torch_cfg.select_guidance_tokens(obs, routing, positive_only_conditional=True)


def test_compute_cfg_metrics_matches_jax():
    labels = [True] * 30 + [False] * 70
    random_values = [0.5] * 27 + [0.05] * 3 + [0.5] * 70
    per_sample_loss = [1.0] * 27 + [2.0] * 3 + [3.0] * 70

    torch_metrics = _torch_cfg.compute_cfg_metrics(
        _torch_routing(positive_only=True, labels=labels, random_values=random_values),
        torch.tensor(per_sample_loss, dtype=torch.float32),
    )
    jax_metrics = _jax_cfg.compute_cfg_metrics(
        _jax_routing(positive_only=True, labels=labels, random_values=random_values),
        jnp.asarray(per_sample_loss, jnp.float32),
    )

    assert torch_metrics.keys() == jax_metrics.keys() == set(_jax_cfg.CFG_RAW_KEYS)
    for key in jax_metrics:
        assert torch_metrics[key].dtype == torch.float32, key
        np.testing.assert_allclose(torch_metrics[key].numpy(), float(jax_metrics[key]), rtol=1e-6, err_msg=key)

    # Headline values (same expectations as the JAX test).
    assert float(torch_metrics["positive_conditional_count"]) == 27.0
    assert float(torch_metrics["negative_unconditional_count"]) == 70.0

    # Derived ratios come from the shared JAX helper — the torch side only emits raw keys.
    derived = _jax_cfg.derive_cfg_ratios({k: float(v) for k, v in torch_metrics.items()})
    assert derived["positive_effective_conditional_ratio"] == pytest.approx(0.9)
    assert derived["negative_effective_conditional_ratio"] == pytest.approx(0.0)
    assert derived["conditional_ratio"] == pytest.approx(0.27)
