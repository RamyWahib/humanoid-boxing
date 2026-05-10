"""Unitree G1 velocity AMP environment configurations.

Reuses :mod:`mjlab.tasks.velocity.config.g1` factories and adds an ``amp``
observation group. The AMP group concatenates joint-space + key-body
kinematics so the discriminator can compare against motion clips.
"""

from __future__ import annotations

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.tasks.velocity.config.g1.env_cfgs import (
  unitree_g1_flat_env_cfg,
  unitree_g1_rough_env_cfg,
)

from beyondAMP.mjlab.obs_groups import amp_obs_basic_group

# G1 anchor / key bodies used by the AMP discriminator.
G1_ANCHOR_NAME: str = "pelvis"
G1_KEY_BODY_NAMES: list[str] = [
  "left_ankle_roll_link",
  "right_ankle_roll_link",
  "left_wrist_yaw_link",
  "right_wrist_yaw_link",
  "torso_link",
]


def _attach_amp_group(cfg: ManagerBasedRlEnvCfg) -> ManagerBasedRlEnvCfg:
  """Attach a basic (joint_pos + joint_vel) AMP group to the env cfg."""
  cfg.observations["amp"] = amp_obs_basic_group()
  return cfg


def unitree_g1_flat_amp_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  return _attach_amp_group(unitree_g1_flat_env_cfg(play=play))


def unitree_g1_rough_amp_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  return _attach_amp_group(unitree_g1_rough_env_cfg(play=play))
