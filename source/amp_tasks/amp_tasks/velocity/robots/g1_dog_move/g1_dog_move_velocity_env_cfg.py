from __future__ import annotations

import math
from dataclasses import MISSING

import isaaclab.sim as sim_utils

##
# Pre-defined configs
##
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from beyondAMP.obs_groups import AMPObsBaiscCfg

import beyondAMP.mdp as mdp

from amp_tasks.velocity.isaac_velocity_env_cfg import IsaacVelocityEnvCfg, ObservationsCfg
from robotlib.beyondMimic.robots.g1 import G1_CYLINDER_CFG

@configclass
class AMPObservationsCfg(ObservationsCfg):
    amp = AMPObsBaiscCfg()

@configclass
class G1DogMoveVelocityEnvCfg(IsaacVelocityEnvCfg):
    observations = AMPObservationsCfg()
    def __post_init__(self):
        self.scene.robot = G1_CYLINDER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        return super().__post_init__()
