#!/usr/bin/env python3
"""Dispatch isolated value stages and validate reusable artifacts (stdlib only)."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
STAGES = ("prepare", "value", "evaluate", "export")


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for block in iter(lambda: file.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def identities(paths):
    result = {}
    for root in paths:
        root = Path(root)
        files = sorted(root.rglob("*")) if root.is_dir() else [root]
        expanded = []
        for file in files:
            if file.is_symlink() and file.is_dir():
                result[str(file)] = ["directory_link", str(file.resolve())]
                expanded.extend(file.resolve().rglob("*"))
        for file in [*files, *expanded]:
            if any(part in ("__pycache__", ".pytest_cache") for part in file.parts):
                continue
            if file.is_file():
                # Video bytes stay external; track their target and stat identity.
                if file.suffix.lower() == ".mp4":
                    st = file.stat()
                    result[str(file)] = [str(file.resolve()), st.st_size, st.st_mtime_ns]
                else:
                    result[str(file)] = file_hash(file)
    return result


def run_stage(name, command, *, work, inputs, artifacts, dry_run=False, reuse=False, after=()):
    print(shlex.join(command), flush=True)
    if dry_run:
        return
    work = Path(work)
    work.mkdir(parents=True, exist_ok=True)
    ledger = work / "run_manifest.json"
    manifest = json.loads(ledger.read_text()) if ledger.exists() else dict(stages={})
    source = dict(command=command, after=after, inputs=identities(inputs))
    signature = hashlib.sha256(json.dumps(source, sort_keys=True).encode()).hexdigest()
    previous = manifest["stages"].get(name)
    if reuse and previous and previous["status"] == "complete":
        if previous["signature"] != signature or previous["artifacts"] != identities(artifacts):
            raise ValueError(f"{name}: inputs or artifacts changed; choose a new output/work directory")
        if not all(Path(p).exists() for p in artifacts):
            raise FileNotFoundError(f"{name}: expected artifacts are missing")
        print(f"Reusing completed {name}", flush=True)
        return
    if name != "prepare" and any(Path(p).exists() for p in artifacts):
        raise FileExistsError(f"{name}: output exists; use --reuse or choose new output paths")
    record = dict(
        status="running", signature=signature, source=source, started_utc=datetime.now(timezone.utc).isoformat()
    )
    manifest["stages"][name] = record
    ledger.write_text(json.dumps(manifest, indent=2))
    try:
        subprocess.run(command, cwd=ROOT, check=True)
        for following in after:
            subprocess.run(following, cwd=ROOT, check=True)
        if not all(Path(p).exists() for p in artifacts):
            raise FileNotFoundError(f"{name}: command did not produce expected artifacts")
        record.update(status="complete", artifacts=identities(artifacts))
    except BaseException as exc:
        record.update(status="failed", error=str(exc))
        raise
    finally:
        record["finished_utc"] = datetime.now(timezone.utc).isoformat()
        ledger.write_text(json.dumps(manifest, indent=2))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=(*STAGES, "all"), default="all")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--reuse", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--value-python", default=sys.executable)
    parser.add_argument("--source-dirs", nargs="+", required=True)
    parser.add_argument("--work-root", required=True)
    parser.add_argument("--data-config", default=str(ROOT / "config/z02_data.yaml"))
    parser.add_argument("--value-config", default=str(ROOT / "config/train_value.yaml"))
    parser.add_argument("--siglip-path", required=True)
    parser.add_argument("--gemma3-path", required=True)
    parser.add_argument("--tokenizer-path")
    parser.add_argument("--value-checkpoint")
    parser.add_argument("--training-dir")
    parser.add_argument("--nproc", type=int, default=1)
    parser.add_argument("--horizon", type=int, default=50)
    parser.add_argument("--label-rule", choices=("fixed", "percentile"), default="fixed")
    parser.add_argument("--threshold", type=float, default=0.0)
    parser.add_argument("--percentile", type=float, default=30.0)
    parser.add_argument("--reference-split", choices=("train", "val", "test"), default="train")
    parser.add_argument("--force-intervention-positive", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args(argv)
    if args.nproc < 1 or args.horizon < 1:
        parser.error("nproc and horizon must be positive")
    if not math.isfinite(args.threshold):
        parser.error("threshold must be finite")
    if args.label_rule == "percentile" and not 0.0 < args.percentile < 100.0:
        parser.error("percentile must lie in (0, 100) for top-fraction labeling")
    work = Path(args.work_root).resolve()
    if args.smoke_test:
        work /= "smoke"
    sources = [str(Path(p).resolve()) for p in args.source_dirs]
    adaptation = work / "adaptation.yaml"
    splits = work / "splits"
    save = work / "value"
    checkpoint = (
        Path(args.value_checkpoint).resolve()
        if args.value_checkpoint
        else save / ("smoke/best_model.pt" if args.smoke_test else "best_model.pt")
    )
    labeled = Path(args.training_dir).resolve() if args.training_dir else work / "labeled"
    models = [Path(p).resolve() for p in (args.siglip_path, args.gemma3_path, args.tokenizer_path or args.gemma3_path)]
    python = args.value_python
    commands = {
        "prepare": [
            python,
            str(ROOT / "scripts/prepare_data.py"),
            "--all",
            "--verify-videos",
            "--config",
            args.data_config,
            "--source-dirs",
            *sources,
            "--work-root",
            str(work),
        ],
        "value": [
            python,
            "-m",
            "torch.distributed.run",
            "--standalone",
            f"--nproc_per_node={args.nproc}",
            str(ROOT / "scripts/train.py"),
            "--config",
            args.value_config,
            "--adaptation_config",
            str(adaptation),
            "--save_dir",
            str(save),
            "--siglip_path",
            str(models[0]),
            "--gemma3_path",
            str(models[1]),
            "--tokenizer_path",
            str(models[2]),
        ],
        "evaluate": [
            python,
            str(ROOT / "submodules/evaluate.py"),
            "--checkpoint",
            str(checkpoint),
            "--output",
            str(work / "evaluation"),
        ],
        "export": [
            python,
            str(ROOT / "scripts/calculate_advantage.py"),
            "--checkpoint",
            str(checkpoint),
            "--split",
            "all",
            "--horizon",
            str(args.horizon),
            "--label-rule",
            args.label_rule,
            "--threshold",
            str(args.threshold),
            "--percentile",
            str(args.percentile),
            "--reference-split",
            args.reference_split,
            "--output",
            str(work / "advantage"),
            "--write-training-dataset",
            "--training-dir",
            str(labeled),
            "--force-intervention-positive" if args.force_intervention_positive else "--no-force-intervention-positive",
        ],
    }
    if args.nproc == 1:
        commands["value"] = [python, *commands["value"][5:]]
    if args.smoke_test:
        commands["value"].append("--smoke_test")
    inputs = dict(
        prepare=[args.data_config, *[Path(p) / folder for p in sources for folder in ("data", "meta", "videos")]],
        value=[args.value_config, adaptation, splits, work / "raw", *models],
        evaluate=[checkpoint, adaptation, splits, work / "raw", *models],
        export=[checkpoint, adaptation, splits, work / "raw", *models],
    )
    artifacts = dict(
        prepare=[adaptation, splits, work / "raw", work / "workspace_manifest.json"],
        value=[checkpoint, checkpoint.parent / "metrics.json"],
        evaluate=[work / "evaluation"],
        export=[work / "advantage", labeled],
    )
    selected = STAGES if args.stage == "all" else (args.stage,)
    for name in selected:
        # Read-only dry runs work before data/model installation.
        render = [[python, str(ROOT / "submodules/render.py"), str(work / "evaluation")]] if name == "evaluate" else []
        code_inputs = [ROOT / "submodules", ROOT / "scripts", ROOT / "runner.py", ROOT / "pyproject.toml"]
        run_stage(
            name,
            commands[name],
            work=work,
            inputs=[*inputs[name], *code_inputs],
            artifacts=artifacts[name],
            dry_run=args.dry_run,
            reuse=args.reuse,
            after=render,
        )
        if args.dry_run:
            for following in render:
                print(shlex.join(following))


if __name__ == "__main__":
    main()
