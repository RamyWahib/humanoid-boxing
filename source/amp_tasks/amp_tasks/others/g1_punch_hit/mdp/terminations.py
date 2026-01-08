import torch

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.assets import Articulation
    from .commands import UniformPunchCommand

def hit_target(
    env: "ManagerBasedRLEnv",
    command_name: str,
):
    command: "UniformPunchCommand" = env.command_manager.get_command(command_name)
    pos_eps      = command.cfg.hit_check.pos_eps 
    vel_min      = command.cfg.hit_check.vel_min 
    hand_pos_w   = command.hand_pos_w
    hand_vel_w   = command.hand_lin_vel_w
    target_pos_w = command.target_pos_w
    
    dist = torch.norm(hand_pos_w - target_pos_w, dim=-1)

    dir_vec = target_pos_w - hand_pos_w
    dir_unit = dir_vec / dir_vec.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    vel_proj = torch.sum(hand_vel_w * dir_unit, dim=-1)

    geom_hit = (dist < pos_eps) & (vel_proj > vel_min)
    return geom_hit
