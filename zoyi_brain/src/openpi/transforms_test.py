import numpy as np
import pytest

import openpi.models.tokenizer as _tokenizer
import openpi.shared.normalize as _normalize
import openpi.transforms as _transforms


def test_repack_transform():
    transform = _transforms.RepackTransform(
        structure={
            "a": {"b": "b/c"},
            "d": "e/f",
        }
    )
    item = {"b": {"c": 1}, "e": {"f": 2}}
    assert transform(item) == {"a": {"b": 1}, "d": 2}


def test_delta_actions():
    item = {"state": np.array([1, 2, 3]), "actions": np.array([[3, 4, 5], [5, 6, 7]])}

    transform = _transforms.DeltaActions(mask=[False, True])
    transformed = transform(item)

    assert np.all(transformed["state"] == np.array([1, 2, 3]))
    assert np.all(transformed["actions"] == np.array([[3, 2, 5], [5, 4, 7]]))


def test_delta_actions_noop():
    item = {"state": np.array([1, 2, 3]), "actions": np.array([[3, 4, 5], [5, 6, 7]])}

    # No-op when the mask is disabled.
    transform = _transforms.DeltaActions(mask=None)
    assert transform(item) is item

    # No-op when there are no actions in the input.
    del item["actions"]
    transform = _transforms.DeltaActions(mask=[True, False])
    assert transform(item) is item


def test_absolute_actions():
    item = {"state": np.array([1, 2, 3]), "actions": np.array([[3, 4, 5], [5, 6, 7]])}

    transform = _transforms.AbsoluteActions(mask=[False, True])
    transformed = transform(item)

    assert np.all(transformed["state"] == np.array([1, 2, 3]))
    assert np.all(transformed["actions"] == np.array([[3, 6, 5], [5, 8, 7]]))


def test_absolute_actions_noop():
    item = {"state": np.array([1, 2, 3]), "actions": np.array([[3, 4, 5], [5, 6, 7]])}

    # No-op when the mask is disabled.
    transform = _transforms.AbsoluteActions(mask=None)
    assert transform(item) is item

    # No-op when there are no actions in the input.
    del item["actions"]
    transform = _transforms.AbsoluteActions(mask=[True, False])
    assert transform(item) is item


def test_make_bool_mask():
    assert _transforms.make_bool_mask(2, -2, 2) == (True, True, False, False, True, True)
    assert _transforms.make_bool_mask(2, 0, 2) == (True, True, True, True)


def test_tokenize_prompt():
    tokenizer = _tokenizer.PaligemmaTokenizer(max_len=12)
    transform = _transforms.TokenizePrompt(tokenizer)

    data = transform({"prompt": "Hello, world!"})

    tok_prompt, tok_mask = tokenizer.tokenize("Hello, world!")
    assert np.allclose(tok_prompt, data["tokenized_prompt"])
    assert np.allclose(tok_mask, data["tokenized_prompt_mask"])


def test_tokenize_no_prompt():
    transform = _transforms.TokenizePrompt(_tokenizer.PaligemmaTokenizer())

    with pytest.raises(ValueError, match="Prompt is required"):
        transform({})


def test_tokenize_prompt_with_guidance():
    tokenizer = _tokenizer.PaligemmaTokenizer(max_len=64)
    transform = _transforms.TokenizePromptWithGuidance(tokenizer)

    data = transform({"prompt": "Hello, world!"})

    # The raw prompt is consumed; six token keys are produced.
    assert "prompt" not in data
    for key in (
        "tokenized_prompt",
        "tokenized_prompt_mask",
        "tokenized_positive_guidance_prompt",
        "tokenized_positive_guidance_prompt_mask",
        "tokenized_negative_guidance_prompt",
        "tokenized_negative_guidance_prompt_mask",
    ):
        assert key in data, key
    assert data["tokenized_prompt"].shape == (64,)

    # Each variant is self-consistent with tokenizing the full string (the tokenizer
    # normalizes "\n" to a space, matching the reference implementation).
    tok_prompt, tok_mask = tokenizer.tokenize("Hello, world!")
    pos_prompt, pos_mask = tokenizer.tokenize("Hello, world!\nAdvantage: positive")
    neg_prompt, neg_mask = tokenizer.tokenize("Hello, world!\nAdvantage: negative")
    np.testing.assert_array_equal(data["tokenized_prompt"], tok_prompt)
    np.testing.assert_array_equal(data["tokenized_prompt_mask"], tok_mask)
    np.testing.assert_array_equal(data["tokenized_positive_guidance_prompt"], pos_prompt)
    np.testing.assert_array_equal(data["tokenized_positive_guidance_prompt_mask"], pos_mask)
    np.testing.assert_array_equal(data["tokenized_negative_guidance_prompt"], neg_prompt)
    np.testing.assert_array_equal(data["tokenized_negative_guidance_prompt_mask"], neg_mask)

    # All three token variants are distinct.
    assert not np.array_equal(data["tokenized_prompt"], data["tokenized_positive_guidance_prompt"])
    assert not np.array_equal(data["tokenized_prompt"], data["tokenized_negative_guidance_prompt"])
    assert not np.array_equal(data["tokenized_positive_guidance_prompt"], data["tokenized_negative_guidance_prompt"])


def test_tokenize_prompt_with_guidance_no_prompt():
    transform = _transforms.TokenizePromptWithGuidance(_tokenizer.PaligemmaTokenizer())

    with pytest.raises(ValueError, match="Prompt is required"):
        transform({})


def test_transform_dict():
    # Rename and remove keys.
    input = {"a": {"b": 1, "c": 2}}
    output = _transforms.transform_dict({"a/b": "a/c", "a/c": None}, input)
    assert output == {"a": {"c": 1}}

    # Raises and error since the renamed key conflicts with an existing key.
    with pytest.raises(ValueError, match="Key 'a/c' already exists in output"):
        _transforms.transform_dict({"a/b": "a/c"}, input)

    # Full match is required and so nothing will be removed.
    input = {"a": {"b": 1, "c": 2}}
    output = _transforms.transform_dict({"a": None}, input)
    assert output == input

    # The regex matches the entire key and so the entire input will be removed.
    input = {"a": {"b": 1, "c": 2}}
    output = _transforms.transform_dict({"a.+": None}, input)
    assert output == {}

    # Replace keys using backreferences. All leaves named 'c' are replaced with 'd'.
    input = {"a": {"b": 1, "c": 1}, "b": {"c": 2}}
    output = _transforms.transform_dict({"(.+)/c": r"\1/d"}, input)
    assert output == {"a": {"b": 1, "d": 1}, "b": {"d": 2}}


def test_extract_prompt_from_task():
    transform = _transforms.PromptFromLeRobotTask({1: "Hello, world!"})

    data = transform({"task_index": 1})
    assert data["prompt"] == "Hello, world!"

    with pytest.raises(ValueError, match="task_index=2 not found in task mapping"):
        transform({"task_index": 2})


@pytest.mark.parametrize("use_quantiles", [False, True])
def test_constant_normalization_restore_and_padding(use_quantiles):
    stats = _transforms.NormStats(
        mean=np.array([3.0, 1.0]),
        std=np.array([0.0, 2.0]),
        q01=np.array([3.0, -1.0]),
        q99=np.array([3.0, 3.0]),
        constant_mask=np.array([True, False]),
    )
    norm = _transforms.Normalize({"state": stats, "actions": stats}, use_quantiles=use_quantiles)
    original = np.array([[3.001, 2.0], [2.999, -0.5]])
    data = norm({"state": original[0], "actions": original})
    np.testing.assert_array_equal(data["actions"][:, 0], 0)
    assert data["state"][0] == 0
    np.testing.assert_array_equal(data["action_loss_mask"], [False, True])
    data = _transforms.PadStatesAndActions(4)(data)
    np.testing.assert_array_equal(data["action_loss_mask"], [False, True, False, False])
    data["actions"][:, 0] = [1e9, -1e9]  # Fixed outputs do not depend on model predictions.
    restored = _transforms.Unnormalize({"state": stats, "actions": stats}, use_quantiles=use_quantiles)(data)
    np.testing.assert_array_equal(restored["actions"][:, 0], 3.0)
    np.testing.assert_allclose(restored["actions"][:, 1], original[:, 1])
    np.testing.assert_array_equal(restored["actions"][:, 2:], 0)
    np.testing.assert_allclose(restored["state"][:2], original[0])
    assert "_unnormalized_state" not in restored
    # Inference has no action labels but still supplies the same sampling mask.
    inference_data = _transforms.PadStatesAndActions(4)(norm({"state": np.array([3.001, 2.0])}))
    np.testing.assert_array_equal(inference_data["action_loss_mask"], data["action_loss_mask"])


@pytest.mark.parametrize("use_quantiles", [False, True])
def test_legacy_normalization_keeps_original_formula(use_quantiles):
    stats = _transforms.NormStats(
        mean=np.array([3.0, 1.0]),
        std=np.array([0.0, 2.0]),
        q01=np.array([3.0, -1.0]),
        q99=np.array([3.0, 3.0]),
    )
    x = np.array([[3.001, 2.0]])
    data = _transforms.Normalize({"actions": stats}, use_quantiles=use_quantiles)({"actions": x})
    expected = (
        (x - stats.q01) / (stats.q99 - stats.q01 + 1e-6) * 2 - 1
        if use_quantiles
        else (x - stats.mean) / (stats.std + 1e-6)
    )
    np.testing.assert_array_equal(data["actions"], expected)
    assert "action_loss_mask" not in data
    np.testing.assert_allclose(
        _transforms.Unnormalize({"actions": stats}, use_quantiles=use_quantiles)(data)["actions"], x
    )


@pytest.mark.parametrize("constant_atol", [None, 0.0, 1e-6])
def test_rare_motion_quantile_fallback_survives_stats_file(tmp_path, constant_atol):
    samples = np.zeros((1000, 3), dtype=np.float64)
    samples[:, 0] = 3
    samples[:, 1] = np.linspace(-2, 2, 1000)
    samples[-1, 2] = 1  # Both quantiles remain zero, but this coordinate must stay trainable.
    running = _normalize.RunningStats()
    running.update(samples[:500])
    running.update(samples[500:])
    stats = running.get_statistics(constant_atol=constant_atol)
    assert stats.q99[2] - stats.q01[2] <= 1e-6
    assert stats.quantile_fallback_mask[2]
    _normalize.save(tmp_path, {"actions": stats})
    saved = _normalize.load(tmp_path)
    data = _transforms.Normalize(saved, use_quantiles=True)({"actions": samples})
    normalized = data["actions"]
    assert np.max(np.abs(normalized[:, 2])) <= 1
    assert normalized[-1, 2] > 0.99
    if constant_atol is not None:
        assert data["action_loss_mask"][2]
        np.testing.assert_array_equal(normalized[:, 0], 0)
    q01, q99 = stats.q01[1], stats.q99[1]
    np.testing.assert_array_equal(normalized[:, 1], (samples[:, 1] - q01) / (q99 - q01 + 1e-6) * 2 - 1)
    padded = _transforms.PadStatesAndActions(5)({**data, "state": samples[0].copy()})
    restored = _transforms.Unnormalize(saved, use_quantiles=True)(padded)["actions"]
    np.testing.assert_allclose(restored[:, :3], samples, atol=1e-12)
    np.testing.assert_array_equal(restored[:, 3:], 0)


def test_narrow_quantile_interval_uses_full_range_without_freezing():
    stats = _transforms.NormStats(
        mean=np.array([0.0]),
        std=np.array([0.1]),
        q01=np.array([0.3]),
        q99=np.array([0.30000001]),
        min=np.array([-0.5]),
        max=np.array([1.7]),
        constant_mask=np.array([False]),
        quantile_fallback_mask=np.array([True]),
    )
    samples = np.array([[-0.5], [0.3], [1.7]])
    data = _transforms.Normalize({"actions": stats}, use_quantiles=True)({"actions": samples})
    assert np.max(np.abs(data["actions"])) <= 1
    assert data["action_loss_mask"][0]
    restored = _transforms.Unnormalize({"actions": stats}, use_quantiles=True)(data)["actions"]
    np.testing.assert_allclose(restored, samples)


def test_legacy_stats_with_range_keep_original_quantile_formula():
    # Stats produced before this fix can contain min/max; do not change an existing checkpoint's scaling.
    stats = _transforms.NormStats(
        mean=np.array([0.0]),
        std=np.array([0.1]),
        q01=np.array([0.0]),
        q99=np.array([0.0]),
        min=np.array([0.0]),
        max=np.array([1.0]),
        constant_mask=np.array([False]),
    )
    samples = np.array([[1.0]])
    data = _transforms.Normalize({"actions": stats}, use_quantiles=True)({"actions": samples})
    np.testing.assert_array_equal(data["actions"], samples / 1e-6 * 2 - 1)
    np.testing.assert_allclose(
        _transforms.Unnormalize({"actions": stats}, use_quantiles=True)(data)["actions"], samples
    )


def test_fixed_delta_actions_restore_in_delta_coordinates():
    stats = _transforms.NormStats(mean=np.array([0.0, 0.7]), std=np.zeros(2), constant_mask=np.ones(2, dtype=bool))
    data = _transforms.Unnormalize({"actions": stats})(
        {"state": np.array([1.2, 0.4]), "actions": np.array([[99.0, 99.0]])}
    )
    result = _transforms.AbsoluteActions([True, False])(data)
    np.testing.assert_allclose(result["actions"], [[1.2, 0.7]])


@pytest.mark.parametrize("framework", ["jax", "torch"])
def test_policy_fixed_delta_uses_current_state(framework):
    from flax import nnx
    import jax.numpy as jnp
    import torch

    from openpi.policies.policy import Policy

    class TorchFixedModel(torch.nn.Module):
        def sample_actions(self, device, observation, **kwargs):
            return torch.tensor([[[999.0, -999.0]]])

    class JaxFixedModel(nnx.Module):
        def sample_actions(self, rng, observation, **kwargs):
            return jnp.array([[[999.0, -999.0]]])

    stats = {
        "state": _transforms.NormStats(
            mean=np.array([1.2, 0.4]), std=np.zeros(2), constant_mask=np.ones(2, dtype=bool)
        ),
        "actions": _transforms.NormStats(
            mean=np.array([0.0, 0.7]), std=np.zeros(2), constant_mask=np.ones(2, dtype=bool)
        ),
    }
    policy = Policy(
        TorchFixedModel() if framework == "torch" else JaxFixedModel(),
        is_pytorch=framework == "torch",
        transforms=[_transforms.Normalize(stats)],
        output_transforms=[_transforms.Unnormalize(stats), _transforms.AbsoluteActions([True, False])],
    )
    result = policy.infer(
        {
            "state": np.array([1.25, 0.45]),
            "image": {"test": np.zeros((2, 2, 3), dtype=np.uint8)},
            "image_mask": {"test": True},
        }
    )
    np.testing.assert_allclose(result["actions"], [[1.25, 0.7]])
    np.testing.assert_allclose(result["state"], [1.25, 0.45])
