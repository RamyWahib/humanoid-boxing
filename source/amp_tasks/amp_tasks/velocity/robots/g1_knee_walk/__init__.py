import gymnasium as gym

from .agents import base_ppo_cfg, amp_ppo_cfg, bmp_ppo_cfg

from . import g1_knee_walk_velocity_env_cfg

##
# Register Gym environments.
##


gym.register(
    id="beyondAMP-Velocity-KneeWalk-G1-PPO",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": g1_knee_walk_velocity_env_cfg.G1KneeWalkVelocityEnvCfg,
        "rsl_rl_cfg_entry_point": base_ppo_cfg.G1FlatPPORunnerCfg,
    },
)

gym.register(
    id="beyondAMP-Velocity-KneeWalk-G1-AMPBasic",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": g1_knee_walk_velocity_env_cfg.G1KneeWalkVelocityEnvCfg,
        "rsl_rl_cfg_entry_point": amp_ppo_cfg.G1FlatAMPRunnerCfg,
    },
)

gym.register(
    id="beyondAMP-Velocity-KneeWalk-G1-BMPBasic",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": g1_knee_walk_velocity_env_cfg.G1KneeWalkVelocityEnvCfg,
        "rsl_rl_cfg_entry_point": bmp_ppo_cfg.G1FlatBMPRunnerCfg,
    },
)
