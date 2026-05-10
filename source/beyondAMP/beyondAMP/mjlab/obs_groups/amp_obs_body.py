"""Body-tracking AMP observation groups for the mjlab backend."""

from __future__ import annotations

from mjlab.managers.observation_manager import (
  ObservationGroupCfg,
  ObservationTermCfg,
)
from mjlab.managers.scene_entity_config import SceneEntityCfg

from beyondAMP.mjlab import mdp


AMPObsSoftTrackTerms: list[str] = [
  "joint_pos",
  "joint_vel",
  "body_quat_w",
  "body_lin_vel_w",
  "body_ang_vel_w",
]

AMPObsHardTrackTerms: list[str] = [
  "joint_pos",
  "joint_vel",
  "body_pos_w",
  "body_quat_w",
  "body_lin_vel_w",
  "body_ang_vel_w",
]


def _body_cfg(body_names: list[str] | None) -> SceneEntityCfg:
  return (
    SceneEntityCfg("robot", body_names=body_names)
    if body_names is not None
    else SceneEntityCfg("robot")
  )


def amp_obs_body_soft_track_terms(
  body_names: list[str] | None = None,
) -> dict[str, ObservationTermCfg]:
  asset = _body_cfg(body_names)
  return {
    "joint_pos": ObservationTermCfg(func=mdp.joint_pos_rel),
    "joint_vel": ObservationTermCfg(func=mdp.joint_vel_rel),
    "body_quat_w": ObservationTermCfg(
      func=mdp.body_quat_w, params={"asset_cfg": asset}
    ),
    "body_lin_vel_w": ObservationTermCfg(
      func=mdp.body_lin_vel_w, params={"asset_cfg": asset}
    ),
    "body_ang_vel_w": ObservationTermCfg(
      func=mdp.body_ang_vel_w, params={"asset_cfg": asset}
    ),
  }


def amp_obs_body_soft_track_group(
  body_names: list[str] | None = None, **group_kwargs
) -> ObservationGroupCfg:
  group_kwargs.setdefault("concatenate_terms", True)
  group_kwargs.setdefault("enable_corruption", False)
  return ObservationGroupCfg(
    terms=amp_obs_body_soft_track_terms(body_names), **group_kwargs
  )


def amp_obs_body_hard_track_terms(
  body_names: list[str] | None = None,
) -> dict[str, ObservationTermCfg]:
  asset = _body_cfg(body_names)
  return {
    "joint_pos": ObservationTermCfg(func=mdp.joint_pos_rel),
    "joint_vel": ObservationTermCfg(func=mdp.joint_vel_rel),
    "body_pos_w": ObservationTermCfg(
      func=mdp.body_pos_w, params={"asset_cfg": asset}
    ),
    "body_quat_w": ObservationTermCfg(
      func=mdp.body_quat_w, params={"asset_cfg": asset}
    ),
    "body_lin_vel_w": ObservationTermCfg(
      func=mdp.body_lin_vel_w, params={"asset_cfg": asset}
    ),
    "body_ang_vel_w": ObservationTermCfg(
      func=mdp.body_ang_vel_w, params={"asset_cfg": asset}
    ),
  }


def amp_obs_body_hard_track_group(
  body_names: list[str] | None = None, **group_kwargs
) -> ObservationGroupCfg:
  group_kwargs.setdefault("concatenate_terms", True)
  group_kwargs.setdefault("enable_corruption", False)
  return ObservationGroupCfg(
    terms=amp_obs_body_hard_track_terms(body_names), **group_kwargs
  )
