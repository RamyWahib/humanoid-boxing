from isaaclab.utils import configclass
from beyondAMP.isaaclab.rsl_rl.configs.rl_cfg import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg
from beyondAMP.isaaclab.rsl_rl.configs.amp_cfg import \
    MotionDatasetCfg, AMPObsBaiscCfg, AMPPPOAlgorithmCfg, AMPRunnerCfg, AMPPPOWeightedAlgorithmCfg
from beyondAMP.obs_groups import AMPObsBaiscTerms, AMPObsSoftTrackTerms, AMPObsHardTrackTerms

from . import general

@configclass
class G1FlatAMPRunnerCfg(AMPRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 10000
    save_interval = 500
    experiment_name = general.experiment_name
    run_name = "amp"
    empirical_normalization = True
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = AMPPPOWeightedAlgorithmCfg(
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
        rescore_interval=100,
    )
    amp_data = MotionDatasetCfg(
        motion_files=general.amp_data_file,
        body_names = general.g1_key_body_names,
        anchor_name = general.g1_anchor_name,
        amp_obs_terms = AMPObsBaiscTerms,
    )
    amp_discr_hidden_dims = [256, 256]
    amp_reward_coef = 1.0
    amp_task_reward_lerp = 0.1
