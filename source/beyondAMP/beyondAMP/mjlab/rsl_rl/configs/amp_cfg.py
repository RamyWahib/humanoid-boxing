"""AMP-specific RSL-RL configs for the mjlab backend.

Mirrors :mod:`beyondAMP.isaaclab.rsl_rl.configs.amp_cfg`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from beyondAMP.motion.motion_dataset import MotionDatasetCfg

from .rl_cfg import RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg


@dataclass
class AMPPPOAlgorithmCfg(RslRlPpoAlgorithmCfg):
  class_name: str = "AMPPPO"
  amp_replay_buffer_size: int = 100_000


@dataclass
class AMPPPOWeightedAlgorithmCfg(AMPPPOAlgorithmCfg):
  class_name: str = "AMPPPOWeighted"
  rescore_interval: int = 50


@dataclass
class AMPRunnerCfg(RslRlOnPolicyRunnerCfg):
  amp_data: Optional[MotionDatasetCfg] = None
  amp_reward_coef: float = 0.5
  amp_discr_hidden_dims: List[int] = field(
    default_factory=lambda: [256, 256]
  )
  # 1.0 → only task reward, 0.0 → only AMP reward.
  amp_task_reward_lerp: float = 0.9
  amp_min_normalized_std: float = 0.0
