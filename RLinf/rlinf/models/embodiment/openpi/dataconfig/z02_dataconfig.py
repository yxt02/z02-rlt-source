# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""OpenPI data configuration for Z02 demonstrations and RLT rollout."""

import dataclasses
from pathlib import Path

from openpi import transforms
from openpi.models.model import BaseModelConfig
from openpi.training.config import (
    DataConfig,
    DataConfigFactory,
    ModelTransformFactory,
)
from typing_extensions import override

from rlinf.models.embodiment.openpi.policies import z02_policy


@dataclasses.dataclass(frozen=True)
class LeRobotZ02RltDataConfig(DataConfigFactory):
    """Use the same masked quantile normalization for SFT and rollout."""

    default_prompt: str | None = None

    @override
    def create(self, assets_dirs: Path, model_config: BaseModelConfig) -> DataConfig:
        asset_id = self.assets.asset_id or self.repo_id
        norm_path = (
            Path(self.assets.assets_dir or assets_dirs) / asset_id / "norm_stats.json"
        )
        state_stats, action_stats = z02_policy.load_z02_norm_stats(norm_path)
        return dataclasses.replace(
            self.base_config or DataConfig(),
            repo_id=self.repo_id,
            asset_id=asset_id,
            # Z02's transforms apply normalization and preserve its mask fields.
            norm_stats={},
            repack_transforms=transforms.Group(inputs=[z02_policy.Z02DemoRepack()]),
            data_transforms=transforms.Group(
                inputs=[z02_policy.Z02Inputs(state_stats, action_stats)],
                outputs=[z02_policy.Z02Outputs(action_stats)],
            ),
            model_transforms=ModelTransformFactory(default_prompt=self.default_prompt)(
                model_config
            ),
            action_sequence_keys=(z02_policy.Z02_ACTION_KEY,),
        )
