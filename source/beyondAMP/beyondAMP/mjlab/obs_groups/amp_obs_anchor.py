"""Anchor-tracking AMP observation groups for the mjlab backend."""

from __future__ import annotations

from mjlab.managers.observation_manager import (
  ObservationGroupCfg,
  ObservationTermCfg,
)
from mjlab.managers.scene_entity_config import SceneEntityCfg

from beyondAMP.mjlab import mdp


AMPObsAnchorHardTrackTerms: list[str] = [
  "joint_pos",
  "joint_vel",
  "anchor_pos_w",
  "anchor_quat_w",
  "anchor_lin_vel_w",
  "anchor_ang_vel_w",
]


def amp_obs_anchor_hard_track_terms(
  anchor_name: str,
) -> dict[str, ObservationTermCfg]:
  asset = SceneEntityCfg("robot", body_names=[anchor_name])
  return {
    "joint_pos": ObservationTermCfg(func=mdp.joint_pos_rel),
    "joint_vel": ObservationTermCfg(func=mdp.joint_vel_rel),
    "anchor_pos_w": ObservationTermCfg(
      func=mdp.anchor_pos_w, params={"asset_cfg": asset}
    ),
    "anchor_quat_w": ObservationTermCfg(
      func=mdp.anchor_quat_w, params={"asset_cfg": asset}
    ),
    "anchor_lin_vel_w": ObservationTermCfg(
      func=mdp.anchor_lin_vel_w, params={"asset_cfg": asset}
    ),
    "anchor_ang_vel_w": ObservationTermCfg(
      func=mdp.anchor_ang_vel_w, params={"asset_cfg": asset}
    ),
  }


def amp_obs_anchor_hard_track_group(
  anchor_name: str, **group_kwargs
) -> ObservationGroupCfg:
  group_kwargs.setdefault("concatenate_terms", True)
  group_kwargs.setdefault("enable_corruption", False)
  return ObservationGroupCfg(
    terms=amp_obs_anchor_hard_track_terms(anchor_name), **group_kwargs
  )
