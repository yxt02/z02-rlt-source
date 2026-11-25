#!/usr/bin/env python3
"""Train the current RECAP value model using prepared split manifests."""

import argparse
import logging
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from submodules.runtime import load_config
from submodules.recap_workflow import train_recap


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/train_value.yaml")
    parser.add_argument("--smoke_test", action="store_true")
    fields = dict(
        batch_size=int,
        num_epochs=int,
        num_workers=int,
        eval_num_workers=int,
        gradient_accumulation_steps=int,
        max_total_steps=int,
        max_steps=int,
        val_steps=int,
        max_samples=int,
        vision_lr=float,
        gemma_lr=float,
        expert_lr=float,
        warmup_steps=int,
        save_dir=str,
        adaptation_config=str,
        siglip_path=str,
        gemma3_path=str,
        tokenizer_path=str,
    )
    for name, kind in fields.items():
        parser.add_argument("--" + name, type=kind)
    for name in ("freeze_vision_encoder", "freeze_vlm", "freeze_value_expert", "gradient_checkpointing"):
        parser.add_argument("--" + name, action=argparse.BooleanOptionalAction, default=None)
    args = parser.parse_args(argv)
    config = load_config(args.config)
    config.update({k: v for k, v in vars(args).items() if k not in ("config", "smoke_test") and v is not None})
    train_recap(config, smoke_test=args.smoke_test)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()
