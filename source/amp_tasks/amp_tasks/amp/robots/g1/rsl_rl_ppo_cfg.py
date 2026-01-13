from isaaclab.utils import configclass
from beyondAMP.isaaclab.configs.rl_cfg import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg

from beyondAMP.isaaclab.configs.amp_cfg import AMPDataCfg, AMPObsBaiscCfg, AMPPPOAlgorithmCfg, AMPRunnerCfg

from beyondAMP.amp_obs_grp import AMPObsBaiscTerms, AMPObsSoftTrackTerms, AMPObsHardTrackTerms

from .config import g1_key_body_names

@configclass
class G1FlatPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 30000
    save_interval = 500
    experiment_name = "g1_flat"
    empirical_normalization = True
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
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
    )

@configclass
class G1FlatAMPRunnerCfg(AMPRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 5000
    save_interval = 500
    experiment_name = "g1_flat"
    empirical_normalization = True
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = AMPPPOAlgorithmCfg(
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
    )
    amp_data = AMPDataCfg(
        motion_files=[
            "data/demo/punch_000.npz"
        ],
        body_names = g1_key_body_names,
        amp_obs_terms = None
    )
    amp_discr_hidden_dims = [256, 256]
    amp_reward_coef = 0.5
    amp_task_reward_lerp = 0.3


@configclass
class G1FlatAMPBaiscCfg(G1FlatAMPRunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.amp_data.amp_obs_terms = AMPObsBaiscTerms
        self.run_name = "basic"


@configclass
class G1FlatAMPSoftTrackCfg(G1FlatAMPRunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.amp_data.amp_obs_terms = AMPObsSoftTrackTerms
        self.run_name = "soft_track"
        self.amp_data.motion_files = [
            "data/datasets/MocapG1Full/LAFAN/walk1_subject1.npz",
            "data/datasets/MocapG1Full/LAFAN/walk1_subject2.npz",
            "data/datasets/MocapG1Full/LAFAN/walk1_subject5.npz",
            "data/datasets/MocapG1Full/LAFAN/walk2_subject1.npz",
            "data/datasets/MocapG1Full/LAFAN/walk2_subject3.npz",
            "data/datasets/MocapG1Full/LAFAN/walk2_subject4.npz",
            "data/datasets/MocapG1Full/LAFAN/walk3_subject1.npz",
            "data/datasets/MocapG1Full/LAFAN/walk3_subject2.npz",
            "data/datasets/MocapG1Full/LAFAN/walk3_subject3.npz",
            "data/datasets/MocapG1Full/LAFAN/walk3_subject4.npz",
            "data/datasets/MocapG1Full/LAFAN/walk3_subject5.npz",
            "data/datasets/MocapG1Full/LAFAN/walk4_subject1.npz",
        ]

@configclass
class G1FlatAMPHardTrackCfg(G1FlatAMPRunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.amp_data.amp_obs_terms = AMPObsHardTrackTerms
        self.run_name = "hard_track"