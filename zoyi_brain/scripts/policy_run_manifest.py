"""Record policy launcher configuration, dataset identity and stage output paths."""

import argparse
import dataclasses
from datetime import datetime
from datetime import timezone
import hashlib
import json
import pathlib
import sys

try:
    from .compute_norm_stats import resolve_config
    from .compute_norm_stats import stats_source
except ImportError:
    from compute_norm_stats import resolve_config
    from compute_norm_stats import stats_source

ROOT = pathlib.Path(__file__).resolve().parents[1]


def file_identity(path):
    return {"path": str(path.resolve()), "size": path.stat().st_size, "mtime_ns": path.stat().st_mtime_ns}


def artifacts(output):
    return [file_identity(p) for p in sorted(output.rglob("*")) if p.is_file()]


def stage_inputs(config, train_args_json, stage):
    source = stats_source(config)
    source["video_files"] = [
        file_identity(file)
        for path in source["repo_ids"]
        for file in sorted((pathlib.Path(path) / "videos").rglob("*.mp4"))
    ]
    inputs = {"config": dataclasses.asdict(config), "train_args": json.loads(train_args_json), "source": source}
    code = [p for p in (ROOT / "src/openpi").rglob("*.py") if not p.name.endswith("_test.py")]
    code += [
        ROOT / "scripts" / name
        for name in (
            "compute_norm_stats.py",
            "policy_run_manifest.py",
            "train.py",
            "train_pytorch.py",
            "train_z02_common.sh",
        )
    ]
    inputs["code"] = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(code)}
    if stage == "policy":
        weights = [config.pytorch_weight_path, getattr(config.weight_loader, "params_path", None)]
        inputs["local_weights"] = [
            file_identity(file)
            for path in weights
            if path
            for file in sorted(pathlib.Path(path).rglob("*"))
            if file.is_file()
        ]
        assets = pathlib.Path(config.data.assets.assets_dir or config.assets_dirs)
        assets /= config.data.assets.asset_id or config.data.repo_id
        inputs["statistics"] = {
            str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (assets / "norm_stats.json", assets / "norm_stats_source.json")
            if p.is_file()
        }
    # Normalize tuples, Paths and enum-like values identically when reading the ledger.
    return json.loads(json.dumps(inputs, default=str))


def record(config_name, train_args_json, manifest_path, stage, status):
    config = resolve_config(config_name, train_args_json=train_args_json)
    manifest_path = pathlib.Path(manifest_path)
    ledger = json.loads(manifest_path.read_text()) if manifest_path.exists() else {"runs": {}}
    key = f"{config.name}:{config.exp_name}:{stage}"
    output = (
        pathlib.Path(config.data.assets.assets_dir or config.assets_dirs)
        / (config.data.assets.asset_id or config.data.repo_id)
        if stage == "norm"
        else config.checkpoint_dir
    )
    entry = ledger["runs"].get(key, {})
    if status == "reuse":
        if entry.get("status") != "complete":
            return False
        if entry["inputs"] != stage_inputs(config, train_args_json, stage) or entry["artifacts"] != artifacts(output):
            raise ValueError(f"{stage}: inputs or artifacts changed; choose a new output/experiment")
        if not output.is_dir() or not entry["artifacts"]:
            raise FileNotFoundError(f"{stage}: completed artifacts are missing")
        return True
    if status == "running":
        entry = {
            "started_utc": datetime.now(timezone.utc).isoformat(),
            "inputs": stage_inputs(config, train_args_json, stage),
        }
    entry.update(status=status, output_path=str(output), updated_utc=datetime.now(timezone.utc).isoformat())
    if status == "complete":
        entry["artifacts"] = artifacts(output)
        if not entry["artifacts"]:
            raise FileNotFoundError(f"{stage}: no completed artifacts at {output}")
    ledger["runs"][key] = entry
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = manifest_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(ledger, indent=2, default=str))
    temporary.replace(manifest_path)
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-name", required=True)
    parser.add_argument("--train-args-json", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--stage", choices=("norm", "policy"), required=True)
    parser.add_argument("--status", choices=("running", "complete", "failed", "reuse"), required=True)
    args = parser.parse_args()
    reusable = record(args.config_name, args.train_args_json, args.manifest, args.stage, args.status)
    if args.status == "reuse" and not reusable:
        sys.exit(3)  # No completed run: the launcher should execute this stage.


if __name__ == "__main__":
    main()
