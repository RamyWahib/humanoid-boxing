"""RL config for Unitree G1 velocity AMP task (mjlab)."""

from __future__ import annotations

from beyondAMP.mjlab.obs_groups import AMPObsBaiscTerms
from beyondAMP.mjlab.rsl_rl import (
  AMPPPOAlgorithmCfg,
  AMPRunnerCfg,
  RslRlPpoActorCriticCfg,
)
from beyondAMP.motion.motion_dataset import MotionDatasetCfg

from ..amp_env_cfg import G1_ANCHOR_NAME, G1_KEY_BODY_NAMES


# Default motion-clip path. Override at run-time via
# ``--agent.amp-data.motion-files=[...]`` on the CLI.
G1_DEFAULT_MOTION_FILES: list[str] = []


def unitree_g1_amp_runner_cfg() -> AMPRunnerCfg:
  return AMPRunnerCfg(
    num_steps_per_env=24,
    max_iterations=5_000,
    save_interval=500,
    experiment_name="g1_velocity_amp",
    run_name="amp",
    empirical_normalization=True,
    policy=RslRlPpoActorCriticCfg(
      init_noise_std=1.0,
      actor_hidden_dims=[512, 256, 128],
      critic_hidden_dims=[512, 256, 128],
      activation="elu",
    ),
    algorithm=AMPPPOAlgorithmCfg(
      class_name="AMPPPO",
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
      entropy_coef=0.005,
      num_learning_epochs=5,
      num_mini_batches=4,
      learning_rate=1.0e-3,
      schedule="adaptive",
      gamma=0.99,
      lam=0.95,
      desired_kl=0.01,
      max_grad_norm=1.0,
    ),
    amp_data=MotionDatasetCfg(
      motion_files=G1_DEFAULT_MOTION_FILES,
      body_names=G1_KEY_BODY_NAMES,
      amp_obs_terms=AMPObsBaiscTerms,
      anchor_name=G1_ANCHOR_NAME,
    ),
    amp_discr_hidden_dims=[256, 256],
    amp_reward_coef=0.5,
    amp_task_reward_lerp=0.3,
  )
