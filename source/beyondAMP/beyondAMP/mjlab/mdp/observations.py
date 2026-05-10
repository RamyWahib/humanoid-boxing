"""Body/anchor tracking observations for the mjlab AMP backend.

mjlab exposes body kinematics under ``body_link_*`` fields rather than
IsaacLab's ``body_*``. This module re-publishes them under the same
function names that the AMP observation groups expect, so the obs group
code stays backend-agnostic.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api import math as math_utils

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def body_pos_w(
  env: "ManagerBasedRlEnv",
  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  pos = asset.data.body_link_pos_w[:, asset_cfg.body_ids]
  pos = pos - env.scene.env_origins.unsqueeze(1)
  return pos.reshape(env.num_envs, -1)


def body_quat_w(
  env: "ManagerBasedRlEnv",
  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  quat = asset.data.body_link_quat_w[:, asset_cfg.body_ids]
  return quat.reshape(env.num_envs, -1)


def body_lin_vel_w(
  env: "ManagerBasedRlEnv",
  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  vel = asset.data.body_link_lin_vel_w[:, asset_cfg.body_ids]
  return vel.reshape(env.num_envs, -1)


def body_ang_vel_w(
  env: "ManagerBasedRlEnv",
  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  vel = asset.data.body_link_ang_vel_w[:, asset_cfg.body_ids]
  return vel.reshape(env.num_envs, -1)


anchor_pos_w = body_pos_w
anchor_quat_w = body_quat_w
anchor_lin_vel_w = body_lin_vel_w
anchor_ang_vel_w = body_ang_vel_w


def base_lin_vel_yaw(
  env: "ManagerBasedRlEnv",
  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
  """Root linear velocity expressed in the yaw-only base frame."""
  asset: Entity = env.scene[asset_cfg.name]
  yaw_quat = math_utils.yaw_quat(asset.data.root_link_quat_w)
  return math_utils.quat_apply_inverse(yaw_quat, asset.data.root_link_lin_vel_w)


def base_ang_vel_yaw(
  env: "ManagerBasedRlEnv",
  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
  """Root angular velocity expressed in the yaw-only base frame."""
  asset: Entity = env.scene[asset_cfg.name]
  yaw_quat = math_utils.yaw_quat(asset.data.root_link_quat_w)
  return math_utils.quat_apply_inverse(yaw_quat, asset.data.root_link_ang_vel_w)
