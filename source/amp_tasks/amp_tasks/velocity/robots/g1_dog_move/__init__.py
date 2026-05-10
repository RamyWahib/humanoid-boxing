import gymnasium as gym

from .agents import base_ppo_cfg, amp_ppo_cfg

from . import g1_dog_move_velocity_env_cfg

##
# Register Gym environments.
##


gym.register(
    id="beyondAMP-Velocity-DogMove-G1-PPO",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": g1_dog_move_velocity_env_cfg.G1DogMoveVelocityEnvCfg,
        "rsl_rl_cfg_entry_point": base_ppo_cfg.G1FlatPPORunnerCfg,
    },
)

gym.register(
    id="beyondAMP-Velocity-DogMove-G1-AMPBasic",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": g1_dog_move_velocity_env_cfg.G1DogMoveVelocityEnvCfg,
        "rsl_rl_cfg_entry_point": amp_ppo_cfg.G1FlatAMPRunnerCfg,
    },
)
