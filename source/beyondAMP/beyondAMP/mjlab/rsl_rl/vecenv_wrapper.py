"""rsl-rl wrapper for mjlab that mirrors :class:`beyondAMP.isaaclab.rsl_rl.RslRlVecEnvWrapper`.

mjlab's own wrapper returns a TensorDict and exposes only ``num_actions``.
The AMP runner expects flat tensors plus ``num_obs`` / ``num_privileged_obs``,
so this subclass adapts the interface without changing mjlab itself.
"""

from __future__ import annotations

import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl.vecenv_wrapper import RslRlVecEnvWrapper as _MjlabBaseWrapper


_ACTOR_GROUP = "actor"
_CRITIC_GROUP = "critic"


class RslRlVecEnvWrapper(_MjlabBaseWrapper):
  """mjlab wrapper exposing IsaacLab-style flat observations.

  Unlike the upstream mjlab wrapper, this one returns the policy/critic
  observation tensors directly instead of a ``TensorDict``. It also
  publishes ``num_obs`` and ``num_privileged_obs`` so that rsl-rl-style
  runners can size their buffers.
  """

  def __init__(
    self,
    env: ManagerBasedRlEnv,
    clip_actions: float | None = None,
    *,
    actor_group: str = _ACTOR_GROUP,
    critic_group: str = _CRITIC_GROUP,
  ) -> None:
    super().__init__(env, clip_actions=clip_actions)
    self._actor_group = actor_group
    self._critic_group = critic_group

    obs_dim = self.unwrapped.observation_manager.group_obs_dim
    self.num_obs = int(obs_dim[actor_group][0])
    self.num_privileged_obs: int | None = (
      int(obs_dim[critic_group][0]) if critic_group in obs_dim else None
    )

  # Observations (flat tensors, not TensorDict).

  def get_observations(self) -> torch.Tensor:
    return self.unwrapped.observation_manager.compute()[self._actor_group]

  def get_privileged_observations(self) -> torch.Tensor | None:
    obs = self.unwrapped.observation_manager.compute()
    return obs.get(self._critic_group)

  # Step / reset return raw tensors.

  def reset(self) -> tuple[torch.Tensor, dict]:
    obs_dict, extras = self.env.reset()
    return obs_dict[self._actor_group], {"observations": obs_dict}

  def step(  # pyright: ignore[reportIncompatibleMethodOverride]
    self, actions: torch.Tensor, **kwargs
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
    if self.clip_actions is not None:
      actions = torch.clamp(actions, -self.clip_actions, self.clip_actions)
    obs_dict, rew, terminated, truncated, extras = self.env.step(actions)
    dones = (terminated | truncated).to(dtype=torch.long)
    extras["observations"] = obs_dict
    if not self.cfg.is_finite_horizon:
      extras["time_outs"] = truncated
    return obs_dict[self._actor_group], rew, dones, extras
