import numpy as np
import pytest

import openpi.shared.normalize as normalize


def test_normalize_update():
    arr = np.arange(12).reshape(4, 3)  # 4 vectors of length 3

    stats = normalize.RunningStats()
    for i in range(len(arr)):
        stats.update(arr[i : i + 1])  # Update with one vector at a time
    results = stats.get_statistics()

    assert np.allclose(results.mean, np.mean(arr, axis=0))
    assert np.allclose(results.std, np.std(arr, axis=0))


def test_serialize_deserialize():
    stats = normalize.RunningStats()
    stats.update(np.arange(12).reshape(4, 3))  # 4 vectors of length 3

    norm_stats = {"test": stats.get_statistics()}
    norm_stats2 = normalize.deserialize_json(normalize.serialize_json(norm_stats))
    assert np.allclose(norm_stats["test"].mean, norm_stats2["test"].mean)
    assert np.allclose(norm_stats["test"].std, norm_stats2["test"].std)


def test_multiple_batch_dimensions():
    # Test with multiple batch dimensions: (2, 3, 4) where 4 is vector dimension
    batch_shape = (2, 3, 4)
    arr = np.random.rand(*batch_shape)

    stats = normalize.RunningStats()
    stats.update(arr)  # Should handle (2, 3, 4) -> reshape to (6, 4)
    results = stats.get_statistics()

    # Flatten batch dimensions and compute expected stats
    flattened = arr.reshape(-1, arr.shape[-1])  # (6, 4)
    expected_mean = np.mean(flattened, axis=0)
    expected_std = np.std(flattened, axis=0)

    assert np.allclose(results.mean, expected_mean)
    assert np.allclose(results.std, expected_std)


def test_constant_mask_uses_full_range_and_survives_serialization():
    data = np.zeros((1000, 4), dtype=np.float64)
    data[:, 0] = 3.0
    data[:, 1] = 2.0 + np.linspace(0.0, 5e-7, 1000)
    data[:, 2] = np.linspace(0.0, 1.0, 1000)
    data[-1, 3] = 1.0  # Rare movement: quantiles alone would incorrectly freeze this dimension.
    running = normalize.RunningStats()
    running.update(data[:500])
    running.update(data[500:])
    stats = running.get_statistics(constant_atol=1e-6)
    np.testing.assert_array_equal(stats.constant_mask, [True, True, False, False])
    np.testing.assert_array_equal(stats.quantile_fallback_mask, [False, False, False, True])
    assert stats.q99[3] < 0.01
    np.testing.assert_array_equal(stats.min, data.min(axis=0))
    np.testing.assert_array_equal(stats.max, data.max(axis=0))
    restored = normalize.deserialize_json(normalize.serialize_json({"actions": stats}))["actions"]
    np.testing.assert_array_equal(restored.constant_mask, stats.constant_mask)
    np.testing.assert_array_equal(restored.quantile_fallback_mask, stats.quantile_fallback_mask)
    np.testing.assert_array_equal(restored.mean, stats.mean)
    assert restored.constant_atol == 1e-6
    assert running.get_statistics(constant_atol=0).constant_mask[0]
    assert not running.get_statistics(constant_atol=0).constant_mask[1]


def test_legacy_stats_do_not_enable_masking():
    stats = normalize.deserialize_json('{"norm_stats":{"actions":{"mean":[1,2],"std":[0,1]}}}')
    assert stats["actions"].constant_mask is None
    assert stats["actions"].quantile_fallback_mask is None
    running = normalize.RunningStats()
    running.update(np.ones((2, 1)))
    assert running.get_statistics().constant_mask is None


@pytest.mark.parametrize(
    "updates",
    [
        {"quantile_fallback_mask": np.array([1])},
        {"quantile_fallback_mask": np.array([True, False])},
        {"min": None},
        {"max": np.zeros(2)},
    ],
)
def test_invalid_quantile_fallback_metadata(updates):
    fields = {
        "mean": np.zeros(1),
        "std": np.ones(1),
        "min": np.zeros(1),
        "max": np.ones(1),
        "quantile_fallback_mask": np.array([True]),
    }
    with pytest.raises(ValueError, match="quantile_fallback_mask"):
        normalize.NormStats(**(fields | updates))


@pytest.mark.parametrize("atol", [-1, np.nan, np.inf])
def test_invalid_constant_tolerance(atol):
    running = normalize.RunningStats()
    running.update(np.ones((2, 1)))
    with pytest.raises(ValueError, match="constant_atol"):
        running.get_statistics(constant_atol=atol)


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_non_finite_stats_rejected(value):
    with pytest.raises(ValueError, match="non-finite"):
        normalize.RunningStats().update(np.array([[value], [0.0]]))


@pytest.mark.parametrize("batch_size", [2, 20])
def test_stats_loader_includes_partial_batch(monkeypatch, batch_size):
    from scripts import compute_norm_stats

    data = [{"state": np.array([float(i)]), "actions": np.array([[float(i)]]), "optional": None} for i in range(5)]
    monkeypatch.setattr(compute_norm_stats._data_loader, "create_torch_dataset", lambda *args: data)  # noqa: SLF001
    config = compute_norm_stats._config.DataConfig(repo_id="fake")  # noqa: SLF001
    loader, num_batches = compute_norm_stats.create_torch_dataloader(config, 1, batch_size, None, 0)
    batches = list(loader)
    assert len(batches) == num_batches
    np.testing.assert_array_equal(np.concatenate([batch["state"] for batch in batches]), np.arange(5)[:, None])
    assert batches[0]["optional"] is None
