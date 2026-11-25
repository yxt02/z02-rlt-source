"""Numerical boundary cases for trajectory-local scores and checkpoint compatibility."""

import submodules  # noqa: F401

import unittest
import numpy as np
import pandas as pd
from submodules.advantage import (
    compute_advantage,
    label_scores,
    intervention_windows,
    export_advantage_table,
    quantile_threshold,
    label_improvement,
    ADVANTAGE_TABLE_COLUMNS,
)
from submodules.contracts import episode_returns


def make_scores(n=4, horizons=(50,), threshold=0.0):
    rows = []
    for h in horizons:
        continuous = np.linspace(-0.5, 0.5, n)
        rows.append(
            pd.DataFrame(
                dict(
                    dataset_id="ds",
                    episode_index=0,
                    frame_index=np.arange(n),
                    timestamp=np.arange(n) / 10.0,
                    fps=10.0,
                    split="test",
                    success=True,
                    reward_raw=-1.0,
                    intervention=np.nan,
                    value=-0.5,
                    effective_horizon=h,
                    reward_sum_raw=0.0,
                    value_next=-0.4,
                    terminal_reached=False,
                    advantage_continuous=continuous,
                    horizon=h,
                    threshold=threshold,
                    label=np.where(continuous > threshold, "advantage", "disadvantage"),
                    is_advantage=continuous > threshold,
                    advantage_forced=np.zeros(n, dtype=bool),
                )
            )
        )
    return pd.concat(rows, ignore_index=True)


class AdvantageTests(unittest.TestCase):
    def test_exact_returns_zero_residual_including_terminal(self):
        for success in (False, True):
            for gamma in (0.0, 0.9, 1.0):
                returns, rewards = episode_returns(80, success, gamma, -2000.0)
                for horizon in (1, 2, 50, 100):
                    result = compute_advantage(returns / 4000.0, rewards, horizon, 4000.0, gamma)
                    np.testing.assert_allclose(result["advantage_continuous"], 0, atol=1e-07)
                    self.assertEqual(result["effective_horizon"][-1], 1)
                    self.assertEqual(result["value_next"][-1], 0)
                    self.assertTrue(result["terminal_reached"][-1])

    def test_interior_and_failure_terminal(self):
        v = np.linspace(-0.8, -0.1, 100)
        r = np.full(100, -1.0)
        r[-1] = -2000
        a = compute_advantage(v, r)
        self.assertAlmostEqual(a["advantage_continuous"][0], v[50] - v[0] - 50 / 4000)
        self.assertAlmostEqual(a["advantage_continuous"][-1], -0.5 - v[-1])
        self.assertEqual(a["effective_horizon"][90], 10)
        self.assertEqual(a["reward_sum_raw"][90], -2009)

    def test_labels_strict_threshold(self):
        self.assertEqual(label_scores([-0.1, 0, 0.1]).tolist(), ["disadvantage", "disadvantage", "advantage"])
        self.assertEqual(label_scores([0.1, 0.2], 0.1).tolist(), ["disadvantage", "advantage"])

    def test_reject_invalid_inputs(self):
        for h in (0, -1, 1.5, True):
            with self.assertRaises(ValueError):
                compute_advantage([0], [-1], h)
        with self.assertRaises(ValueError):
            compute_advantage([np.nan], [-1])
        with self.assertRaises(ValueError):
            compute_advantage([0], [-1, -1])
        with self.assertRaises(ValueError):
            label_scores([1], np.nan)

    def test_intervention_alignment_and_missing_flags(self):
        times = np.arange(0, 5, 0.1)
        score = times.copy()
        flags = np.zeros(50)
        flags[:3] = 1
        flags[20:25] = 1
        flags[48:] = 1
        grid, events = intervention_windows(times, score, flags)
        self.assertEqual(events.shape, (1, 61))
        np.testing.assert_allclose(events[0], 2 + grid)
        self.assertEqual(intervention_windows(times, score, None)[1].shape, (0, 61))
        with self.assertRaises(ValueError):
            intervention_windows(times, score, np.full(50, 2))

    def test_quantile_threshold_top_fraction_on_advantages(self):
        scores = np.linspace(-1.0, 0.0, 101)
        self.assertAlmostEqual(quantile_threshold(scores, 0.3), -0.3)
        self.assertAlmostEqual(quantile_threshold(scores, 0.5), -0.5)
        self.assertAlmostEqual(quantile_threshold(scores, 0.01), -0.01)
        thr = quantile_threshold(scores, 0.3)
        self.assertAlmostEqual(float((scores >= thr).mean()), 0.3, places=1)
        for bad in ([], np.full(3, np.nan)):
            with self.assertRaises(ValueError):
                quantile_threshold(bad, 0.3)
        with self.assertRaises(ValueError):
            quantile_threshold(scores, 0.0)
        with self.assertRaises(ValueError):
            quantile_threshold(scores, 1.0)
        with self.assertRaises(ValueError):
            quantile_threshold(scores, -0.1)
        with self.assertRaises(ValueError):
            quantile_threshold(scores, 1.5)

    def test_label_improvement_forces_intervention_positive(self):
        score = np.array([-0.5, -0.1, 0.2])
        positive, forced = label_improvement(score, -0.2)
        np.testing.assert_array_equal(positive, [False, True, True])
        self.assertFalse(forced.any())
        positive, forced = label_improvement(score, -0.2, np.array([0.0, 1.0, np.nan]))
        np.testing.assert_array_equal(positive, [False, True, True])
        np.testing.assert_array_equal(forced, [False, True, False])
        positive, forced = label_improvement(np.array([-0.2]), -0.2, np.array([np.nan]), inclusive=True)
        np.testing.assert_array_equal(positive, [True])
        positive, forced = label_improvement(np.array([-0.2]), -0.2, np.array([np.nan]), inclusive=False)
        np.testing.assert_array_equal(positive, [False])
        self.assertFalse(forced.any())
        with self.assertRaises(ValueError):
            label_improvement(score, -0.2, np.array([0.0, 1.0]))
        with self.assertRaises(ValueError):
            label_improvement(score, -0.2, np.array([0.0, 2.0, 0.0]))
        with self.assertRaises(ValueError):
            label_improvement(np.array([np.nan]), 0.0)

    def test_export_advantage_table_projection(self):
        scores = make_scores()
        before = scores.copy()
        table = export_advantage_table(scores, inclusive=False)
        self.assertEqual(list(table.columns), ADVANTAGE_TABLE_COLUMNS)
        self.assertEqual(len(table), len(scores))
        self.assertEqual(table["advantage"].dtype, bool)
        np.testing.assert_array_equal(table["advantage"].to_numpy(), scores["is_advantage"].to_numpy())
        pd.testing.assert_frame_equal(scores, before)
        tied = make_scores(threshold=0.5)
        tied.loc[tied.index[-1], "is_advantage"] = True
        table = export_advantage_table(tied, inclusive=True)
        self.assertTrue(bool(table["advantage"].iloc[-1]))
        with self.assertRaises(ValueError):
            export_advantage_table(tied, inclusive=False)

    def test_export_advantage_table_multi_horizon_unique_keys(self):
        scores = make_scores(horizons=(1, 10, 50))
        table = export_advantage_table(scores, inclusive=False)
        keys = ["dataset_id", "episode_index", "frame_index", "horizon"]
        self.assertFalse(table.duplicated(keys).any())
        self.assertEqual(len(table), 12)
        self.assertEqual(sorted(table["horizon"].unique().tolist()), [1, 10, 50])

    def test_export_advantage_table_rejects_invalid(self):
        base = make_scores()
        for mutate in (
            lambda s: s.drop(columns=["horizon"]),
            lambda s: s.iloc[:0],
            lambda s: pd.concat([s, s.iloc[[0]]], ignore_index=True),
            lambda s: s.assign(advantage_continuous=lambda d: d["advantage_continuous"].where(d.index != 1, np.nan)),
            lambda s: s.assign(is_advantage=lambda d: ~d["is_advantage"].astype(bool)),
        ):
            with self.assertRaises(ValueError):
                export_advantage_table(mutate(base), inclusive=False)
        with self.assertRaises(ValueError):
            export_advantage_table(base.to_numpy())


if __name__ == "__main__":
    unittest.main()
