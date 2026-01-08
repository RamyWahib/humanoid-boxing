from typing import List
from isaaclab.utils import configclass
from .rl_cfg import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg

from beyondAMP.obs_groups import AMPObsBaiscCfg
from rsl_rl_amp.runners.bmp_on_policy_runner import BMPOnPolicyRunner
from beyondAMP.motion.motion_dataset import MotionDatasetCfg
from dataclasses import MISSING

@configclass
class BMPPPOAlgorithmCfg(RslRlPpoAlgorithmCfg):
    class_name="BMPPPO"
    amp_replay_buffer_size: int = 100000

@configclass
class BMPRunnerCfg(RslRlOnPolicyRunnerCfg):
    runner_type             : type[BMPOnPolicyRunner] = BMPOnPolicyRunner
    amp_data                : MotionDatasetCfg = MISSING
    amp_reward_coef         : float = MISSING
    amp_task_reward_lerp    : float = 0.9
    amp_min_normalized_std  : float = 0.0 # recmended for no minimal explore std. since the action limit may large 
    