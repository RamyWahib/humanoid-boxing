from __future__ import annotations

from isaaclab.utils import configclass

from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm

import beyondAMP.mdp as mdp


@configclass
class AMPObsGrpCfg(ObsGroup):
    def adjust_key_body_indexes(self, terms:list, key_bodys:list):
        for term_name in terms:
            term:ObsTerm = getattr(self, term_name)
            if "asset_cfg" in term.params:
                term.params["asset_cfg"].body_names = key_bodys
            else:
                term.params["asset_cfg"] = SceneEntityCfg(name="robot", body_names=key_bodys)
        return self

@configclass
class AMPObsBaiscCfg(AMPObsGrpCfg):
    joint_pos = ObsTerm(func=mdp.joint_pos_rel)
    joint_vel = ObsTerm(func=mdp.joint_vel_rel)
    
AMPObsBaiscTerms = ["joint_pos", "joint_vel"]
    
@configclass
class AMPObsSoftTrackCfg(AMPObsGrpCfg):
    joint_pos = ObsTerm(func=mdp.joint_pos_rel)
    joint_vel = ObsTerm(func=mdp.joint_vel_rel)
    body_quat_w = ObsTerm(func=mdp.body_quat_w)
    body_lin_vel_w = ObsTerm(func=mdp.body_lin_vel_w)
    body_ang_vel_w = ObsTerm(func=mdp.body_ang_vel_w)

AMPObsSoftTrackTerms = ["joint_pos", "joint_vel", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w"]

@configclass
class AMPObsHardTrackCfg(AMPObsGrpCfg):
    joint_pos = ObsTerm(func=mdp.joint_pos_rel)
    joint_vel = ObsTerm(func=mdp.joint_vel_rel)
    body_pos_w = ObsTerm(func=mdp.body_pose_w)
    body_quat_w = ObsTerm(func=mdp.body_quat_w)
    body_lin_vel_w = ObsTerm(func=mdp.body_lin_vel_w)
    body_ang_vel_w = ObsTerm(func=mdp.body_ang_vel_w)
    
AMPObsHardTrackTerms = ["joint_pos", "joint_vel", "body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w"]