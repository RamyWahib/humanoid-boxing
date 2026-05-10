"""AMP observation groups using only joint-space + base-velocity terms.

mjlab's :class:`ObservationGroupCfg` takes a dict of terms (rather than
class attributes like IsaacLab), so we provide factory helpers that
return either the terms dict or a ready-to-use group.
"""

from __future__ import annotations

from typing import Iterable

from mjlab.managers.observation_manager import (
  ObservationGroupCfg,
  ObservationTermCfg,
)
from mjlab.managers.scene_entity_config import SceneEntityCfg

from beyondAMP.mjlab import mdp


AMPObsBaiscTerms: list[str] = ["joint_pos", "joint_vel"]
AMPObsClassicTerms: list[str] = [
  "joint_pos",
  "joint_vel",
  "base_lin_vel",
  "base_ang_vel",
]


def _set_body_names(
  terms: dict[str, ObservationTermCfg],
  term_names: Iterable[str],
  body_names: list[str],
) -> dict[str, ObservationTermCfg]:
  """In-place: attach a SceneEntityCfg with body_names to each term."""
  for name in term_names:
    term = terms[name]
    if "asset_cfg" in term.params:
      term.params["asset_cfg"].body_names = body_names
    else:
      term.params["asset_cfg"] = SceneEntityCfg("robot", body_names=body_names)
  return terms


def amp_obs_basic_terms() -> dict[str, ObservationTermCfg]:
  return {
    "joint_pos": ObservationTermCfg(func=mdp.joint_pos_rel),
    "joint_vel": ObservationTermCfg(func=mdp.joint_vel_rel),
  }


def amp_obs_basic_group(**group_kwargs) -> ObservationGroupCfg:
  group_kwargs.setdefault("concatenate_terms", True)
  group_kwargs.setdefault("enable_corruption", False)
  return ObservationGroupCfg(terms=amp_obs_basic_terms(), **group_kwargs)


def amp_obs_classic_terms() -> dict[str, ObservationTermCfg]:
  return {
    "joint_pos": ObservationTermCfg(func=mdp.joint_pos_rel),
    "joint_vel": ObservationTermCfg(func=mdp.joint_vel_rel),
    "base_lin_vel": ObservationTermCfg(func=mdp.base_lin_vel_yaw),
    "base_ang_vel": ObservationTermCfg(func=mdp.base_ang_vel_yaw),
  }


def amp_obs_classic_group(**group_kwargs) -> ObservationGroupCfg:
  group_kwargs.setdefault("concatenate_terms", True)
  group_kwargs.setdefault("enable_corruption", False)
  return ObservationGroupCfg(terms=amp_obs_classic_terms(), **group_kwargs)
