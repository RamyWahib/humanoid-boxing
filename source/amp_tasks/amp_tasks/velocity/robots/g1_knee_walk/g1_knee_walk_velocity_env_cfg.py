from __future__ import annotations

import math
from dataclasses import MISSING

import isaaclab.sim as sim_utils

##
# Pre-defined configs
##
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from beyondAMP.obs_groups import AMPObsBaiscCfg

import beyondAMP.mdp as mdp

from amp_tasks.velocity.isaac_velocity_env_cfg import IsaacVelocityEnvCfg, ObservationsCfg, RewardsCfg
from robotlib.beyondMimic.robots.g1 import G1_CYLINDER_CFG

@configclass
class KneeWalkRewardsCfg(RewardsCfg):
    # knee_air_time = RewTerm(
    #     func=mdp.feet_air_time,
    #     weight=35.0,
    #     params={
    #         "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*knee.*" , ".*ankle.*"]),
    #         "command_name": "base_velocity",
    #         "threshold": 0.15,
    #     },
    # )
    pass

@configclass
class AMPObservationsCfg(ObservationsCfg):
    amp = AMPObsBaiscCfg()

@configclass
class G1KneeWalkVelocityEnvCfg(IsaacVelocityEnvCfg):
    observations = AMPObservationsCfg()
    def __post_init__(self):
        self.scene.robot = G1_CYLINDER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        return super().__post_init__()
