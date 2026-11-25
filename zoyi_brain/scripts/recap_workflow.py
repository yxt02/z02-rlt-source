#!/usr/bin/env python3
"""Run isolated value stages and the user's independently configured policy launchers."""

import argparse
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
STAGES = ("prepare", "value", "evaluate", "export", "norm", "policy")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=(*STAGES, "all"), default="all")
    parser.add_argument("--backend", choices=("jax", "pytorch"), default="pytorch")
    parser.add_argument("--model", choices=("pi0", "pi05", "pi05_lora"), default="pi05")
    parser.add_argument("--policy-config")
    parser.add_argument("--recap", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--reuse", action="store_true")
    args = parser.parse_args(argv)
    if args.model.endswith("lora") and args.backend != "jax":
        parser.error("The existing LoRA launcher is JAX only")
    policy = "train_recap_policy_z02.sh"
    env = dict(os.environ, POLICY_BACKEND=args.backend, POLICY_MODEL=args.model, POLICY_RECAP=str(args.recap).lower())
    for stage in STAGES if args.stage == "all" else (args.stage,):
        if stage in STAGES[:4]:
            command = ["bash", str(ROOT / "train_recap_value_z02.sh"), "--stage", stage]
            if args.reuse:
                command.append("--reuse")
        else:
            command = ["bash", str(ROOT / policy)]
            if args.policy_config:
                command.append(args.policy_config)
            command += ["--stage", "norm" if stage == "norm" else "train", "--recap" if args.recap else "--no-recap"]
            if args.reuse:
                command.append("--reuse")
        if args.dry_run:
            command.append("--dry-run")
        subprocess.run(command, cwd=ROOT, env=env, check=True)


if __name__ == "__main__":
    main()
