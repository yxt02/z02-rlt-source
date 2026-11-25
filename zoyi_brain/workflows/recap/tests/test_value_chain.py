"""CPU smoke through real value training, reload, evaluation, scoring and publication.

Uses a small SigLIP/Gemma Transformer in place of pretrained backbones.
"""

from contextlib import ExitStack
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import submodules  # noqa: F401
import torch
from omegaconf import OmegaConf
from transformers import Gemma3ForCausalLM, Gemma3TextConfig, SiglipVisionConfig, SiglipVisionModel
from submodules.recap_model import EXPERT_VARIANTS, RecapValueModel
from submodules.recap_workflow import train_recap, load_recap_checkpoint
from submodules.evaluation_workflow import evaluate, render
from test_pipeline import prepared

ROOT = Path(__file__).resolve().parents[1]


def tiny_factory(config):
    vision = SiglipVisionModel(
        SiglipVisionConfig(
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            image_size=224,
            patch_size=224,
        )
    )
    gemma = Gemma3ForCausalLM(
        Gemma3TextConfig(
            vocab_size=32,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=1,
            num_key_value_heads=1,
            head_dim=256,
            layer_types=["full_attention"] * 2,
        )
    )
    tokenizer = SimpleNamespace(pad_token_id=0, encode=lambda *args, **kwargs: [1, 2, 3])
    with (
        patch("submodules.recap_model.SiglipVisionModel.from_pretrained", return_value=vision),
        patch("submodules.recap_model.Gemma3ForCausalLM.from_pretrained", return_value=gemma),
        patch("submodules.recap_model.AutoTokenizer.from_pretrained", return_value=tokenizer),
    ):
        return RecapValueModel(config)


def run_chain(base):
    source, adaptation, working = prepared(base)
    cfg = OmegaConf.to_container(OmegaConf.load(ROOT / "config/train_value.yaml"), resolve=True)
    models = base / "models"
    for name in ("vision", "gemma"):
        folder = models / name
        folder.mkdir(parents=True)
        (folder / "model.safetensors").write_bytes(b"fixture identity")
        (folder / "tokenizer.json").write_text("{}")
        (folder / "preprocessor_config.json").write_text(json.dumps(dict(image_mean=[0.5] * 3, image_std=[0.5] * 3)))
    cfg.update(
        adaptation_config=str(adaptation),
        siglip_path=str(models / "vision"),
        gemma3_path=str(models / "gemma"),
        tokenizer_path=str(models / "gemma"),
        critic_expert_variant="fixture",
        precision="fp32",
        batch_size=2,
        gradient_accumulation_steps=1,
        num_workers=0,
        eval_num_workers=0,
        pin_memory=False,
        num_epochs=1,
        max_total_steps=1,
        max_steps=1,
        val_steps=1,
        warmup_steps=0,
        save_dir=str(base / "value"),
        train_image_augmentation=False,
    )
    with (
        patch.dict(
            EXPERT_VARIANTS,
            fixture=dict(
                hidden_size=16,
                intermediate_size=32,
                num_hidden_layers=2,
                num_attention_heads=1,
                num_key_value_heads=1,
                head_dim=256,
            ),
        ),
        patch("submodules.recap_workflow.RecapValueModel", side_effect=tiny_factory),
    ):
        train_recap(cfg)
        checkpoint = base / "value/best_model.pt"
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        assert payload["global_step"] == 1 and payload["config"]["value_scale"] > 0
        load_recap_checkpoint(payload, torch.device("cpu"))
        evaluate(checkpoint, base / "evaluation")
        render(base / "evaluation")
        import importlib.util

        spec = importlib.util.spec_from_file_location("calculate", ROOT / "scripts/calculate_advantage.py")
        calculate = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(calculate)
        calculate.main(
            [
                "--checkpoint",
                str(checkpoint),
                "--split",
                "all",
                "--output",
                str(base / "advantage"),
                "--horizon",
                "50",
                "--write-training-dataset",
                "--training-dir",
                str(base / "labeled"),
            ]
        )
    return base / "labeled/tiny"


class ValueChainTests(unittest.TestCase):
    def test_current_value_chain_on_cpu(self):
        output = os.environ.get("RECAP_SMOKE_OUTPUT")
        with ExitStack() as stack:
            base = Path(output) if output else Path(stack.enter_context(tempfile.TemporaryDirectory()))
            base.mkdir(parents=True, exist_ok=True)
            target = run_chain(base)
            manifest = json.loads((base / "advantage/manifest.json").read_text())
            self.assertEqual(manifest["status"], "complete")
            self.assertEqual(manifest["horizons"], [50])
            self.assertTrue((base / "evaluation/index.html").exists())
            self.assertTrue((target / "videos").is_symlink())
            self.assertEqual(json.loads((target / "training_manifest.json").read_text())["frames"], 32)


if __name__ == "__main__":
    unittest.main()
