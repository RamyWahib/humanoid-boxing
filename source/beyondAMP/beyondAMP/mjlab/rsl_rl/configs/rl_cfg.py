"""RSL-RL configs for the mjlab AMP backend.

Mirrors :mod:`beyondAMP.isaaclab.rsl_rl.configs.rl_cfg` but uses stdlib
``dataclass`` (mjlab does not use IsaacLab's ``@configclass``).

The :class:`AMPOnPolicyRunner` consumes these via
``asdict(...)`` / ``cfg.to_dict()``, expecting the keys ``policy`` and
``algorithm`` — not ``actor``/``critic`` like mjlab's stock
:class:`mjlab.rl.RslRlOnPolicyRunnerCfg`. We keep the IsaacLab-flavored
schema here so the AMP runner stays unchanged across backends.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from mjlab.rl.config import RslRlBaseRunnerCfg


@dataclass
class RslRlPpoActorCriticCfg:
  """PPO actor-critic network config (IsaacLab-style)."""

  class_name: str = "ActorCritic"
  init_noise_std: float = 1.0
  noise_std_type: Literal["scalar", "log"] = "scalar"
  actor_hidden_dims: list[int] = field(default_factory=lambda: [512, 256, 128])
  critic_hidden_dims: list[int] = field(default_factory=lambda: [512, 256, 128])
  activation: str = "elu"


@dataclass
class RslRlPpoAlgorithmCfg:
  class_name: str = "PPO"
  num_learning_epochs: int = 5
  num_mini_batches: int = 4
  learning_rate: float = 1.0e-3
  schedule: str = "adaptive"
  gamma: float = 0.99
  lam: float = 0.95
  entropy_coef: float = 0.01
  desired_kl: float = 0.01
  max_grad_norm: float = 1.0
  value_loss_coef: float = 1.0
  use_clipped_value_loss: bool = True
  clip_param: float = 0.2
  normalize_advantage_per_mini_batch: bool = False


@dataclass
class RslRlOnPolicyRunnerCfg(RslRlBaseRunnerCfg):
  empirical_normalization: bool = False
  policy: RslRlPpoActorCriticCfg = field(default_factory=RslRlPpoActorCriticCfg)
  algorithm: RslRlPpoAlgorithmCfg = field(default_factory=RslRlPpoAlgorithmCfg)
  device: str = "cuda:0"

  def to_dict(self) -> dict[str, Any]:
    from dataclasses import asdict

    return asdict(self)
