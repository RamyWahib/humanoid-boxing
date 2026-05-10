"""AMP wrapper for mjlab. Mirrors :mod:`beyondAMP.isaaclab.rsl_rl.amp_wrapper`."""

from __future__ import annotations

import torch

from mjlab.envs import ManagerBasedRlEnv

from beyondAMP.motion.motion_dataset import MotionDataset
from beyondAMP.motion.weighted_motion_dataset import WeightedMotionDataset

from .vecenv_wrapper import RslRlVecEnvWrapper


_AMP_GROUP = "amp"


class AMPEnvWrapper(RslRlVecEnvWrapper):
  """mjlab AMP wrapper.

  Adds a third ``amp`` observation group, an ``AMP-aware`` step that returns
  the seven-tuple expected by :class:`AMPOnPolicyRunner`, and exposes
  :attr:`motion_dataset` / :attr:`dof_pos_limits`.
  """

  def __init__(
    self,
    env: ManagerBasedRlEnv,
    clip_actions: float | None = None,
    *,
    motion_dataset: MotionDataset | dict[str, MotionDataset] | None = None,
    amp_group: str = _AMP_GROUP,
  ) -> None:
    super().__init__(env, clip_actions=clip_actions)
    self._amp_group = amp_group
    self.rewards_shape = self.unwrapped.reward_manager._step_reward.shape[-1]

    if isinstance(motion_dataset, MotionDataset) or motion_dataset is None:
      self.motion_dataset = motion_dataset
    else:
      self.motion_dataset = WeightedMotionDataset(
        motion_dataset, self.unwrapped, self.unwrapped.device
      )
    # Stash on env so MDP terms (rewards / events) can reach it.
    self.unwrapped.motion_dataset = self.motion_dataset

  # AMP observations.

  def get_amp_observations(self) -> torch.Tensor:
    return self.unwrapped.observation_manager.compute()[self._amp_group]

  # AMP-aware step.

  def step(  # pyright: ignore[reportIncompatibleMethodOverride]
    self,
    actions: torch.Tensor,
    *,
    not_amp: bool = True,
    **kwargs,
  ):
    if not_amp:
      return super().step(actions, **kwargs)

    if self.clip_actions is not None:
      actions = torch.clamp(actions, -self.clip_actions, self.clip_actions)
    obs_dict, rew, terminated, truncated, extras = self.env.step(actions)
    dones = (terminated | truncated).to(dtype=torch.long)

    obs = obs_dict[self._actor_group].clamp(-500, 500)
    privileged_obs = obs_dict.get(self._critic_group, obs).clamp(-500, 500)
    terminal_amp_states = obs_dict.get(self._amp_group, obs).clamp(-500, 500)
    extras["observations"] = obs_dict

    reset_env_ids = torch.where(dones)[0]
    if not self.cfg.is_finite_horizon:
      extras["time_outs"] = truncated

    return (
      obs,
      privileged_obs,
      rew,
      dones,
      extras,
      reset_env_ids,
      terminal_amp_states[reset_env_ids],
    )

  @property
  def dof_pos_limits(self) -> torch.Tensor:
    return self.unwrapped.scene["robot"].data.joint_pos_limits
