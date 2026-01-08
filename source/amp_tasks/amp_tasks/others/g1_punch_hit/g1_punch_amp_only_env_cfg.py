from isaaclab.utils import configclass
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import EventTermCfg  as EventTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm

from robotlib.beyondMimic.robots.g1 import G1_ACTION_SCALE, G1_CYLINDER_CFG
from robotlib.robot_keys.g1_29d import g1_key_body_names

from amp_tasks.others.amp_env_cfg import AMPEnvCfg, RewardsCfg, EventCfg, ObservationsCfg
from beyondAMP.obs_groups import AMPObsBaiscCfg, AMPObsBodySoftTrackCfg, AMPObsBodyHardTrackCfg

from . import mdp as punch_mdp
from isaaclab.envs import mdp

@configclass
class G1PunchCommands:
    punch_command = punch_mdp.UniformPunchCommandCfg(
        asset_name="robot",
        hand_names=["right_wrist_roll_link"],
        resampling_time_range = (), 
    )

@configclass
class G1PunchObservationsCfg(ObservationsCfg):
    @configclass
    class PolicyCfg(ObservationsCfg.PolicyCfg):
        punch_command   = ObsTerm(func=mdp.generated_commands, params={"command_name": "punch_command"})
    
    @configclass
    class PrivilegedCfg(ObservationsCfg.PrivilegedCfg):
        punch_command   = ObsTerm(func=mdp.generated_commands, params={"command_name": "punch_command"})
    
    policy = PolicyCfg()
    critic = PrivilegedCfg()
        
@configclass
class G1PunchRewardsCfg(RewardsCfg):
    punch_dist_reward = RewTerm(
        punch_mdp.punch_dist_reward,
        params={"command_name": "punch_command"},
        weight=0.3
    )
    punch_velocity_reward = RewTerm(
        punch_mdp.punch_velocity_reward,
        params={"command_name": "punch_command"},
        weight=0.3
    )
    punch_on_time_hit_reward = RewTerm(
        punch_mdp.punch_on_time_hit_reward,
        params={"command_name": "punch_command"},
        weight=0.3
    )
    punch_post_hit_stability_reward = RewTerm(
        punch_mdp.punch_post_hit_stability_reward,
        params={"command_name": "punch_command"},
        weight=0.3
    )

@configclass
class G1PunchEvents(EventCfg):
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        },
    )
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (-1.0, 1.0),
            "velocity_range": (-1.0, 1.0),
        },
    )
    def __post_init__(self):
        super().__post_init__()
        self.reset_to_ref_motion_dataset = None

@configclass
class G1FlatEnvCfg(AMPEnvCfg):
    observations = G1PunchObservationsCfg()
    rewards = G1PunchRewardsCfg()
    events = G1PunchEvents()
    commands = G1PunchCommands()
    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = G1_CYLINDER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.actions.joint_pos.scale = G1_ACTION_SCALE
        
