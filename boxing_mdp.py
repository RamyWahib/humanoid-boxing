from __future__ import annotations
import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg

def hand_to_target_vec(env, asset_cfg):
    robot = env.scene["robot"]
    target = env.scene["target"]
    body_idx = asset_cfg.body_ids
    hand_pos = robot.data.body_pos_w[:, body_idx, :]
    target_pos = target.data.root_pos_w.unsqueeze(1)
    vec = target_pos - hand_pos
    return torch.nan_to_num(vec.reshape(env.num_envs, -1), nan=0.0)

def hand_lin_vel(env, asset_cfg):
    robot = env.scene["robot"]
    body_idx = asset_cfg.body_ids
    vel = robot.data.body_lin_vel_w[:, body_idx, :]
    vel = torch.nan_to_num(vel, nan=0.0, posinf=0.0, neginf=0.0)
    return vel.reshape(env.num_envs, -1)

def target_pos_in_robot_frame(env):
    robot = env.scene["robot"]
    target = env.scene["target"]
    rel = target.data.root_pos_w - robot.data.root_pos_w
    return torch.nan_to_num(rel, nan=0.0)

def compute_hit_data(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, hit_threshold: float = 0.20, speed_threshold: float = 0.5) -> dict:
    robot = env.scene["robot"]
    target = env.scene["target"]

    body_idx = asset_cfg.body_ids

    # hand positions: (num_envs, 2, 3)
    hand_pos = torch.nan_to_num(robot.data.body_pos_w[:, body_idx, :], nan=0.0, posinf=0.0, neginf=0.0)

    # target position: (num_envs, 1, 3)
    target_pos = target.data.root_pos_w.unsqueeze(1)

    # vector from hand to target: (num_envs, 2, 3)
    diff = target_pos - hand_pos

    # distance from each hand to target: (num_envs, 2)
    dist = torch.norm(diff, dim=-1)

    # unit vector pointing from hand toward target: (num_envs, 2, 3)
    unit_vec = diff / (dist.unsqueeze(-1) + 1e-6)

    # hand velocities: (num_envs, 2, 3)
    hand_vel = torch.nan_to_num(robot.data.body_lin_vel_w[:, body_idx, :], nan=0.0, posinf=0.0, neginf=0.0)

    # speed of each hand toward target: (num_envs, 2)
    speed_toward = (hand_vel * unit_vec).sum(dim=-1)

    # hit if either hand is close AND moving fast toward target: (num_envs,)
    hit_mask = ((dist < hit_threshold) & (speed_toward > speed_threshold)).any(dim=-1)

    return {
        "hit_mask": hit_mask,
        "speed_toward": speed_toward,
        "dist": dist,
    }

def reward_hit_bonus(env, asset_cfg):
    data = compute_hit_data(env, asset_cfg)
    robot = env.scene["robot"]
    # gate by upright — no boxing reward if falling
    upright = robot.data.projected_gravity_b[:, 2].clamp(0.0, 1.0)
    return data["hit_mask"].float() * upright

def reward_strike(env, asset_cfg, proximity_threshold=0.35):
    data = compute_hit_data(env, asset_cfg)
    robot = env.scene["robot"]
    upright = robot.data.projected_gravity_b[:, 2].clamp(0.0, 1.0)
    in_range = (data["dist"] < proximity_threshold).float()
    strike = (data["speed_toward"].clamp(min=0.0) * in_range).max(dim=-1).values
    return strike.clamp(max=5.0) * upright

def reward_efficiency(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    robot = env.scene["robot"]
    body_idx = asset_cfg.body_ids
    hand_vel = robot.data.body_lin_vel_w[:, body_idx, :]
    hand_vel = torch.nan_to_num(hand_vel, nan=0.0, posinf=0.0, neginf=0.0)  # ← kill bad values
    speed = torch.norm(hand_vel, dim=-1)
    return -speed.sum(dim=-1).clamp(min=-5.0, max=0.0)  # ← tight clamp    

def reward_stay_in_place(env: ManagerBasedRLEnv) -> torch.Tensor:
    robot = env.scene["robot"]
    root_pos = torch.nan_to_num(robot.data.root_pos_w[:, :2], nan=0.0)
    origin = env.scene.env_origins[:, :2]
    dist_from_origin = torch.norm(root_pos - origin, dim=-1)
    return -dist_from_origin.clamp(max=10.0)  # clamp to avoid explosions

def reward_face_target(env: ManagerBasedRLEnv) -> torch.Tensor:
    robot = env.scene["robot"]
    target = env.scene["target"]

    to_target = target.data.root_pos_w[:, :2] - robot.data.root_pos_w[:, :2]
    to_target = torch.nan_to_num(torch.nn.functional.normalize(to_target, dim=-1), nan=0.0)

    forward = robot.data.root_quat_w
    w, x, y, z = forward[:, 0], forward[:, 1], forward[:, 2], forward[:, 3]
    robot_forward = torch.stack([1 - 2*(y**2 + z**2), 2*(x*y + w*z)], dim=-1)
    robot_forward = torch.nan_to_num(torch.nn.functional.normalize(robot_forward, dim=-1), nan=0.0)

    facing = (robot_forward * to_target).sum(dim=-1)
    return torch.nan_to_num(facing, nan=0.0)

def reward_proximity(env, asset_cfg):
    data = compute_hit_data(env, asset_cfg)
    min_dist = data["dist"].min(dim=-1).values
    return torch.nan_to_num(torch.exp(-2.0 * min_dist), nan=0.0)  # less aggressive decay

def reward_default_pose(env: ManagerBasedRLEnv) -> torch.Tensor:
    robot = env.scene["robot"]
    # penalize deviation from default joint positions
    joint_pos = robot.data.joint_pos
    default_pos = robot.data.default_joint_pos
    diff = torch.norm(joint_pos - default_pos, dim=-1)
    return -diff

def reward_joint_vel_penalty(env: ManagerBasedRLEnv) -> torch.Tensor:
    robot = env.scene["robot"]
    joint_vel = robot.data.joint_vel
    return -torch.norm(joint_vel, dim=-1)  # (num_envs,)
  