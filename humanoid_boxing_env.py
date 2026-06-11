from __future__ import annotations
from numpy import info
import torch
from isaaclab.envs import ManagerBasedRLEnv
from .humanoid_boxing_env_cfg import HumanoidBoxingEnvCfg


class HumanoidBoxingEnv(ManagerBasedRLEnv):
    cfg: HumanoidBoxingEnvCfg

    def __init__(self, cfg: HumanoidBoxingEnvCfg, **kwargs):
        super().__init__(cfg, **kwargs)
        
        # target position per env: (num_envs, 3)
        self._target_pos = torch.zeros(self.num_envs, 3, device=self.device)
        self._joint_ids = self.scene["robot"].find_joints(".*")[0]
        self._actions = torch.zeros(self.num_envs, self.scene["robot"].num_joints, device=self.device)
        self._randomize_target()

    def _randomize_target(self, env_ids=None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)

        n = len(env_ids)
        # random angle in front of robot: ±40 degrees
        angle = (torch.rand(n, device=self.device) - 0.5) * 2 * 0.698  # ±40° in radians
        dist  = torch.rand(n, device=self.device) * 0.4 + 0.4           # 0.4–0.8 m
        height = torch.rand(n, device=self.device) * 0.4 + 1.2          # 1.2–1.6 m

        self._target_pos[env_ids, 0] = dist * torch.cos(angle)
        self._target_pos[env_ids, 1] = dist * torch.sin(angle)
        self._target_pos[env_ids, 2] = height

        # write into sim
        target = self.scene["target"]
        root_state = target.data.default_root_state.clone()
        root_state[env_ids, :3] = (
            self.scene.env_origins[env_ids] + self._target_pos[env_ids]
        )
        target.write_root_state_to_sim(root_state)

    def _pre_physics_step(self, actions):
        self._actions = actions.clamp(-1.0, 1.0)

    def _apply_action(self):
        self.scene["robot"].set_joint_effort_target(
        self._actions, joint_ids=self._joint_ids
    )

    def step(self, action):
        obs, rew, terminated, truncated, info = super().step(action)
        if "policy" in obs:
            obs["policy"] = torch.nan_to_num(obs["policy"], nan=0.0, posinf=0.0, neginf=0.0)
        rew = torch.nan_to_num(rew, nan=0.0, posinf=0.0, neginf=0.0)
        reset_ids = (terminated | truncated).nonzero(as_tuple=False).squeeze(-1)
        if len(reset_ids) > 0:
            self._randomize_target(reset_ids)
        return obs, rew, terminated, truncated, info