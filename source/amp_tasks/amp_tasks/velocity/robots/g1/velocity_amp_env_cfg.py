
from isaaclab.utils import configclass
from .velocity_env_cfg import RobotEnvCfg

from beyondAMP.amp_obs_grp import AMPObsBaiscCfg, AMPObsSoftTrackCfg, AMPObsHardTrackCfg


@configclass
class G1VelocityAMPEnvCfg(RobotEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.observations.amp = AMPObsBaiscCfg()