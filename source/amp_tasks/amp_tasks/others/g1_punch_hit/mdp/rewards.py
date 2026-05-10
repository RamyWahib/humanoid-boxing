import torch

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.assets import Articulation
    from .commands import UniformPunchCommand


def punch_dist_reward_raw(
    env: "ManagerBasedRLEnv",
    command_name: str
):
    command: "UniformPunchCommand" = env.command_manager.get_term(command_name)
    hand_pos_w = command.hand_pos_w
    target_pos_w = command.target_pos_w
    dist = torch.norm(hand_pos_w - target_pos_w, dim=-1)
    return torch.exp(-dist)
def punch_dist_reward(
    env: "ManagerBasedRLEnv",
    command_name: str
):
    raw = punch_dist_reward_raw(env, command_name)
    command: "UniformPunchCommand" = env.command_manager.get_term(command_name)
    return (raw * command.is_punch_action_time.float()).reshape(-1)


def punch_velocity_reward_raw(
    env: "ManagerBasedRLEnv",
    command_name: str,
):
    command: "UniformPunchCommand" = env.command_manager.get_term(command_name)
    hand_pos_w = command.hand_pos_w
    hand_vel_w = command.hand_lin_vel_w
    target_pos_w = command.target_pos_w
    dir_vec = target_pos_w - hand_pos_w
    dir_unit = dir_vec / dir_vec.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    vel_proj = torch.sum(hand_vel_w * dir_unit, dim=-1)
    return torch.clamp(vel_proj, min=0.0)
def punch_velocity_reward(
    env: "ManagerBasedRLEnv",
    command_name: str,
):
    raw = punch_velocity_reward_raw(env, command_name)
    command: "UniformPunchCommand" = env.command_manager.get_term(command_name)
    return (raw * command.is_punch_action_time.float()).reshape(-1)


def punch_on_time_hit_reward(
    env: "ManagerBasedRLEnv",
    command_name: str,
):
    command: "UniformPunchCommand" = env.command_manager.get_term(command_name)
    on_time_hit = command.is_expected_hit_time.float()
    hand_vel_w = command.hand_lin_vel_w
    return (on_time_hit * torch.norm(hand_vel_w, dim=-1)).reshape(-1)


def punch_post_hit_stability_reward_raw(
    env: "ManagerBasedRLEnv",
    command_name: str,
    asset_name: str = "robot",
):
    robot: "Articulation" = env.scene[asset_name]
    base_ang_vel = robot.data.root_ang_vel_b
    return torch.exp(-torch.norm(base_ang_vel, dim=-1))
def punch_post_hit_stability_reward(
    env: "ManagerBasedRLEnv",
    command_name: str,
    asset_name: str = "robot",
):
    raw = punch_post_hit_stability_reward_raw(env, command_name, asset_name)
    command: "UniformPunchCommand" = env.command_manager.get_term(command_name)
    return (raw * command.is_post_action.float()).reshape(-1)
