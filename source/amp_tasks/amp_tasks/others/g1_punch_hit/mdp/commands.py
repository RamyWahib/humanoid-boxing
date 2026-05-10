import torch
from dataclasses import MISSING

from isaaclab.utils import configclass
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.assets import Articulation
from isaaclab.utils.math import quat_apply, quat_apply_inverse
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.markers.config import FRAME_MARKER_CFG

class UniformPunchCommand(CommandTerm):
    """Uniform punch target command (position + time + hand)."""

    cfg: "UniformPunchCommandCfg"

    def __init__(self, cfg: "UniformPunchCommandCfg", env):
        super().__init__(cfg, env)

        self.robot: Articulation = env.scene[cfg.asset_name]

        # hand body indices (0: left, 1: right)
        self.hand_ids = torch.tensor(
            self.robot.find_bodies(cfg.hand_names, preserve_order=True)[0],
            device=self.device,
            dtype=torch.long,
        )

        # command buffers
        self._p_target_b    = torch.zeros((self.num_envs, 3), device=self.device)
        self._p_target_w    = torch.zeros((self.num_envs, 3), device=self.device)
        self._hand_id       = torch.zeros((self.num_envs, 1), device=self.device)

        # internal clock (time since command was sampled)
        self._elapsed_time = torch.zeros((self.num_envs, 1), device=self.device)
        
        self.on_time_hit = torch.zeros((self.num_envs, ), device=self.device).bool()
        self.geom_hit = torch.zeros((self.num_envs, ), device=self.device).bool()
        self._is_punch_action_time = torch.zeros((self.num_envs, ), device=self.device).bool()
        self._is_expected_hit_time = torch.zeros((self.num_envs, ), device=self.device).bool()
        self._is_pre_action = torch.ones((self.num_envs, ), device=self.device).bool()
        self._is_post_action = torch.zeros((self.num_envs, ), device=self.device).bool()

    # --------------------------------------------------------------------- #
    # Sampling
    # --------------------------------------------------------------------- #

    def _resample_command(self, env_ids: torch.Tensor):
        """Sample a new punch command."""
        cfg = self.cfg.ranges

        # target position in base frame
        self._p_target_b[env_ids, 0].uniform_(*cfg.pos_x)
        self._p_target_b[env_ids, 1].uniform_(*cfg.pos_y)
        self._p_target_b[env_ids, 2].uniform_(*cfg.pos_z)

        # base (anchor) pose in world
        base_pos_w = self.robot.data.root_pos_w[env_ids]        # (N, 3)
        base_quat_w = self.robot.data.root_quat_w[env_ids]      # (N, 4)
        # convert to world frame and store persistently
        self._p_target_w[env_ids] = quat_apply(base_quat_w, self._p_target_b[env_ids]) + base_pos_w


        if self.cfg.use_duel_hand:
            # which hand to use: 0 = left, 1 = right
            self._hand_id[env_ids, 0] = torch.randint(
                low=0, high=2, size=(len(env_ids),), device=self.device
            )

        # reset timer
        self._elapsed_time[env_ids, 0] = 0.0
        self.on_time_hit[env_ids] = False
        self.geom_hit[env_ids] = False
        self._is_pre_action[env_ids] = True
        self._is_post_action[env_ids] = False

    def _update_command(self):
        """
        Update internal timer.
        Resampling is event-driven (typically by task or reset),
        so we do NOT auto-resample here.
        """
        self._elapsed_time += self._env.sim.cfg.dt

        if self.cfg.update_base_frame_target:
            # recompute target in current base frame from fixed world target
            base_pos_w = self.robot.data.root_pos_w        # (N, 3)
            base_quat_w = self.robot.data.root_quat_w      # (N, 4)
            self._p_target_b = quat_apply_inverse(base_quat_w, self._p_target_w - base_pos_w)
        
        # update time-based flags
        t = self._elapsed_time.squeeze(-1)
        self._is_punch_action_time = (
            self.cfg.ranges.punch_action_time_range[0] <= t
        ) & (
            t <= self.cfg.ranges.punch_action_time_range[1]
        )
        self._is_expected_hit_time = (
            self.cfg.ranges.punch_expected_time_range[0] <= t
        ) & (
            t <= self.cfg.ranges.punch_expected_time_range[1]
        )
        self._is_pre_action = t < self.cfg.ranges.punch_action_time_range[0]
        self._is_post_action = t > self.cfg.ranges.punch_action_time_range[1]
        
        self._update_hit_info()

    def _update_metrics(self):
        # compute hand-to-target position error
        error_vec = self.hand_pos_w - self.target_pos_w
        self.metrics["punch/hand_pos_error"] = torch.norm(error_vec, dim=-1)
        
        # compute hand linear speed
        self.metrics["punch/hand_speed"] = torch.norm(self.hand_lin_vel_w, dim=-1)
        
        # log elapsed time
        # self.metrics["punch/elapsed_time"] = self._elapsed_time.squeeze(-1)
        
        # log hit-related flags (cast to float)
        # self.metrics["punch/geom_hit"] = self.geom_hit.float()
        # self.metrics["punch/on_time_hit"] = self.on_time_hit.float()
        # self.metrics["punch/is_action_time"] = self._is_punch_action_time.float()
        # self.metrics["punch/is_expected_hit_time"] = self._is_expected_hit_time.float()

    def _update_hit_info(self):
        pos_eps      = self.cfg.hit_check.pos_eps 
        vel_min      = self.cfg.hit_check.vel_min 
        hand_pos_w   = self.hand_pos_w
        hand_vel_w   = self.hand_lin_vel_w
        target_pos_w = self.target_pos_w

        # geometry
        dist = torch.norm(hand_pos_w - target_pos_w, dim=-1)

        dir_vec = target_pos_w - hand_pos_w
        dir_unit = dir_vec / dir_vec.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        vel_proj = torch.sum(hand_vel_w * dir_unit, dim=-1)

        geom_hit = (dist < pos_eps) & (vel_proj > vel_min)
        on_time = geom_hit & self._is_expected_hit_time
        
        self.geom_hit = geom_hit
        self.on_time_hit = on_time

    # --------------------------------------------------------------------- #
    # Interface
    # --------------------------------------------------------------------- #

    @property
    def hand_indices(self):
        hand_id = self._hand_id[:, 0].long()
        hand_indices = self.hand_ids[hand_id]
        return hand_indices
    
    @property
    def hand_pos_w(self):
        return self.robot.data.body_pos_w[torch.tensor(range(self.num_envs), device=self.device), self.hand_indices]

    @property
    def hand_lin_vel_w(self):
        return self.robot.data.body_lin_vel_w[torch.tensor(range(self.num_envs), device=self.device), self.hand_indices]

    @property
    def command(self) -> torch.Tensor:
        """
        Returns:
            (N, 5) tensor:
            [p_target_b (3), _t_hit (1), hand_id (1)]
        """
        return torch.cat([self._p_target_b, self.skill_progress, self._hand_id], dim=-1)

    @property
    def skill_progress(self):
        std = self.cfg.ranges.punch_expected_time_range[1] - self.cfg.ranges.punch_expected_time_range[0]
        mean = (self.cfg.ranges.punch_expected_time_range[1] + self.cfg.ranges.punch_expected_time_range[0]) / 2
        return ((self._elapsed_time - mean) / std).clamp(0, 1)
        
    @property
    def is_punch_action_time(self) -> torch.Tensor:
        """Boolean flag indicating if currently in punch action time window."""
        return self._is_punch_action_time

    @property
    def is_expected_hit_time(self) -> torch.Tensor:
        """Boolean flag indicating if currently in expected hit time window."""
        return self._is_expected_hit_time

    @property
    def is_pre_action(self):
        return self._is_pre_action

    @property
    def is_post_action(self):
        return self._is_post_action

    @property
    def target_pos_b(self) -> torch.Tensor:
        """Target position in base frame."""
        return self._p_target_b
    
    @property
    def target_pos_w(self) -> torch.Tensor:
        """Target position in world frame (for visualization / debug)."""
        return self._p_target_w

    # --------------------------------------------------------------------- #
    # Debug visualization
    # --------------------------------------------------------------------- #

    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "goal_pose_visualizer"):
                self.goal_pose_visualizer = VisualizationMarkers(self.cfg.goal_pose_visualizer_cfg)
                self.current_pose_visualizer = VisualizationMarkers(self.cfg.current_pose_visualizer_cfg)
            self.goal_pose_visualizer.set_visibility(True)
            self.current_pose_visualizer.set_visibility(True)
        else:
            if hasattr(self, "goal_pose_visualizer"):
                self.goal_pose_visualizer.set_visibility(False)
                self.current_pose_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        # robot might be de-initialized
        if not self.robot.is_initialized:
            return

        # visualize punch target (position only, no orientation)
        p_w = self.target_pos_w
        q_identity = torch.zeros((self.num_envs, 4), device=self.device)
        q_identity[:, 0] = 1.0

        self.goal_pose_visualizer.visualize(p_w, q_identity)

        # visualize selected hand current pose
        hand_pose_w = self.robot.data.body_link_pose_w[torch.tensor(range(self.num_envs), device=self.device), self.hand_indices]

        self.current_pose_visualizer.visualize(hand_pose_w[:, :3], hand_pose_w[:, 3:7])


@configclass
class UniformPunchCommandCfg(CommandTermCfg):
    """Uniform punch command generator."""

    class_type: type = UniformPunchCommand

    asset_name: str = MISSING
    """Robot asset name."""

    hand_names: tuple[str, str] = MISSING
    """(left_hand, right_hand) body names."""
    
    use_duel_hand: bool = False
    """Whether using two hand to hit"""
    
    update_base_frame_target: bool = False
    """Whether using the static target or the target may change"""

    @configclass
    class HitCheck:
        pos_eps     : float = 0.08
        vel_min     : float = 0.5
        time_eps    : float = 0.15

    @configclass
    class Ranges:
        """Sampling ranges in base frame."""

        pos_x: tuple[float, float] = (0.4, 0.8)
        pos_y: tuple[float, float] = (-0.3, 0.3)
        pos_z: tuple[float, float] = (0.9, 1.3)
        
        # Action execution window (when punch-like motion is allowed)
        punch_action_time_range: tuple[float, float] = (0.0, 1.1)
        
        # Expected hit timing window (for on-time evaluation)
        punch_expected_time_range: tuple[float, float] = (0.5, 1.0)

    hit_check: HitCheck = HitCheck()
    ranges: Ranges = Ranges()

    visualize: bool = True
    """Whether to visualize punch target."""

    goal_pose_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(
        prim_path="/Visuals/Command/punch_target"
    )
    current_pose_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(
        prim_path="/Visuals/Command/hand_pose"
    )
