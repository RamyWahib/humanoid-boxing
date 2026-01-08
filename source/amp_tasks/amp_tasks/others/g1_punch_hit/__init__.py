import gymnasium as gym

from . import g1_punch_amp_only_env_cfg
from .agents import amp_ppo_cfg, base_ppo_cfg, bmp_ppo_cfg

##
# Register Gym environments.
##

gym.register(
    id="beyondAMP-PunchHitTask-G1-PPO",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": g1_punch_amp_only_env_cfg.G1FlatEnvCfg,
        "rsl_rl_cfg_entry_point": base_ppo_cfg.G1FlatPPORunnerCfg,
    },
)

gym.register(
    id="beyondAMP-PunchHitTask-G1-AMPBasic",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": g1_punch_amp_only_env_cfg.G1FlatEnvCfg,
        "rsl_rl_cfg_entry_point": amp_ppo_cfg.G1FlatAMPRunnerCfg,
    },
)

gym.register(
    id="beyondAMP-PunchHitTask-G1-BMPBasic",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": g1_punch_amp_only_env_cfg.G1FlatEnvCfg,
        "rsl_rl_cfg_entry_point": bmp_ppo_cfg.G1FlatBMPRunnerCfg,
    },
)
