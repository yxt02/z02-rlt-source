"""Working-copy, complete-export, CLI and stage-reuse contracts."""

import submodules  # noqa: F401

import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd
from omegaconf import OmegaConf
from submodules.data_workflow import run
from submodules.dataset_export import export_training_dataset
from submodules.workspace import prepare_workspace, source_identity
from fixtures import make_dataset

ROOT = Path(__file__).resolve().parents[1]


def prepared(base, dim=25):
    source = make_dataset(base / "source" / "tiny", dim)
    config = OmegaConf.to_container(OmegaConf.load(ROOT / "config/z02_data.yaml"))
    config.update(
        datasets=[dict(name="tiny", expected_joint_dim=None, joint_indices=None, outcome_override=None)],
        train_ratio=0.5,
        val_ratio=0.25,
        test_ratio=0.25,
    )
    path = base / "data.yaml"
    OmegaConf.save(config, path)
    (base / "original_identity.json").write_text(json.dumps(source_identity(source)))
    effective = prepare_workspace(path, [source], base / "work")
    run(effective, write=True, split=True, verify_videos=True)
    return source, effective, base / "work/raw/tiny"


def score_rows(root):
    rows = []
    files = {}
    for file in sorted((root / "data").rglob("*.parquet")):
        table = pd.read_parquet(file)
        episode = int(table.episode_index.iloc[0])
        files[f"tiny:{episode}"] = dict(root=str(root), file=str(file))
        for frame in table.frame_index:
            rows.append(
                dict(
                    dataset_id="tiny",
                    episode_index=episode,
                    frame_index=int(frame),
                    horizon=50,
                    advantage_continuous=0.1 if frame % 2 else -0.1,
                    is_advantage=bool(frame % 2),
                )
            )
    return pd.DataFrame(rows), files


class PipelineTests(unittest.TestCase):
    def test_prepare_export_preserves_source_and_handles_dimensions(self):
        for dim in (25, 40):
            with self.subTest(dim=dim), tempfile.TemporaryDirectory() as temporary:
                base = Path(temporary)
                source, effective, working = prepared(base, dim)
                baseline = json.loads((base / "original_identity.json").read_text())
                self.assertEqual(baseline, source_identity(source))
                self.assertEqual(
                    json.loads((working / "meta/info.json").read_text())["features"]["action.joint_positions"]["shape"],
                    [dim],
                )
                self.assertEqual(json.loads((working / "meta/z02_adaptation.json").read_text())["raw_joint_dim"], dim)
                splits = [json.loads((base / f"work/splits/{s}.json").read_text()) for s in ("train", "val", "test")]
                self.assertEqual([len(s["episodes"]) for s in splits], [4, 2, 2])
                scores, files = score_rows(working)
                targets = export_training_dataset(base / "labeled", scores, files)
                target = Path(targets[0])
                self.assertTrue((target / "videos").is_symlink())
                self.assertEqual(source_identity(source), baseline)
                self.assertEqual(len(list((target / "data").rglob("*.parquet"))), 8)
                self.assertEqual(json.loads((target / "training_manifest.json").read_text())["status"], "complete")
                first = pd.read_parquet(next((target / "data").rglob("*.parquet")))
                self.assertEqual(first.advantage_positive.dtype, bool)
                self.assertEqual(first["action.joint_positions"].iloc[0].shape, (dim,))
                np.testing.assert_allclose(first.advantage, [-0.1, 0.1, -0.1, 0.1])

    def test_incomplete_labels_do_not_publish_dataset(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            _, _, working = prepared(base)
            scores, files = score_rows(working)
            for partial in (scores.iloc[:-1], scores[scores.episode_index != 0]):
                with self.assertRaisesRegex(ValueError, "Incomplete"):
                    export_training_dataset(base / "labeled", partial, files)
                self.assertFalse((base / "labeled").exists())

    def test_changed_source_and_invalid_video_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            source, effective, working = prepared(base)
            scores, files = score_rows(working)
            next((source / "videos").rglob("*.mp4")).write_bytes(b"broken video")
            with self.assertRaisesRegex(ValueError, "video"):
                export_training_dataset(base / "labeled", scores, files)
            with self.assertRaisesRegex(ValueError, "Source data changed"):
                prepare_workspace(base / "data.yaml", [source], base / "work")

    def test_invalid_labels_and_episode_metadata_do_not_publish(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            _, _, working = prepared(base)
            scores, files = score_rows(working)
            for key, value in (("is_advantage", None), ("is_advantage", "true"), ("advantage_continuous", np.nan)):
                bad = scores.astype({key: object}).copy()
                bad.loc[bad.index[0], key] = value
                with self.assertRaises(ValueError):
                    export_training_dataset(base / "labeled", bad, files)
                self.assertFalse((base / "labeled").exists())
            episodes_path = working / "meta/episodes.jsonl"
            entries = [json.loads(line) for line in episodes_path.read_text().splitlines()]
            entries[0]["length"] += 1
            episodes_path.write_text("".join(json.dumps(e) + "\n" for e in entries))
            with self.assertRaisesRegex(ValueError, "Episode metadata"):
                export_training_dataset(base / "labeled", scores, files)
            self.assertFalse((base / "labeled").exists())

    def test_v21_episode_and_aggregate_label_statistics(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            _, _, working = prepared(base)
            path = working / "meta/info.json"
            info = json.loads(path.read_text())
            info["codebase_version"] = "v2.1"
            path.write_text(json.dumps(info))
            episodes = [dict(episode_index=i, stats={}) for i in range(8)]
            (working / "meta/episodes_stats.jsonl").write_text("".join(json.dumps(e) + "\n" for e in episodes))
            scores, files = score_rows(working)
            target = Path(export_training_dataset(base / "labeled", scores, files)[0])
            stats = [json.loads(line) for line in (target / "meta/episodes_stats.jsonl").read_text().splitlines()]
            for record in stats:
                self.assertAlmostEqual(record["stats"]["advantage_positive"]["mean"][0], 0.5)
                self.assertEqual(record["stats"]["advantage"]["count"], [4])
            aggregate = json.loads((target / "meta/stats.json").read_text())
            self.assertEqual(aggregate["advantage_positive"]["count"], [32])

    def test_current_cli_help_and_no_legacy_training_flags(self):
        for path in (
            "scripts/train.py",
            "scripts/prepare_data.py",
            "scripts/calculate_advantage.py",
            "scripts/visualize_episode.py",
            "submodules/evaluate.py",
            "submodules/render.py",
            "runner.py",
        ):
            result = subprocess.run([sys.executable, str(ROOT / path), "--help"], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
        result = subprocess.run(
            [sys.executable, str(ROOT / "scripts/train.py"), "--prepare-cache"], capture_output=True, text=True
        )
        self.assertNotEqual(result.returncode, 0)

    def test_reuse_validates_outputs_and_failed_stage_remains_failed(self):
        spec = importlib.util.spec_from_file_location("runner", ROOT / "runner.py")
        runner = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(runner)
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            source = base / "input"
            source.write_text("input")
            target = base / "result"
            command = [
                sys.executable,
                "-c",
                'from pathlib import Path; import sys; Path(sys.argv[1]).write_text("ok")',
                str(target),
            ]
            kwargs = dict(work=base, inputs=[source], artifacts=[target])
            runner.run_stage("test", command, **kwargs)
            runner.run_stage("test", command, reuse=True, **kwargs)
            target.write_text("changed")
            with self.assertRaisesRegex(ValueError, "changed"):
                runner.run_stage("test", command, reuse=True, **kwargs)
            with self.assertRaises(subprocess.CalledProcessError):
                runner.run_stage(
                    "failure",
                    [sys.executable, "-c", "raise SystemExit(2)"],
                    work=base,
                    inputs=[source],
                    artifacts=[base / "missing"],
                )
            self.assertEqual(
                json.loads((base / "run_manifest.json").read_text())["stages"]["failure"]["status"], "failed"
            )


if __name__ == "__main__":
    unittest.main()
