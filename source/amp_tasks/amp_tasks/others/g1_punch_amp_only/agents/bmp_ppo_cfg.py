from isaaclab.utils import configclass
from beyondAMP.isaaclab.rsl_rl.configs.rl_cfg import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg
from beyondAMP.isaaclab.rsl_rl.configs.bmp_cfg import \
    MotionDatasetCfg, AMPObsBaiscCfg, BMPOnPolicyRunner, BMPPPOAlgorithmCfg, BMPRunnerCfg
from beyondAMP.obs_groups import AMPObsBaiscTerms, AMPObsSoftTrackTerms, AMPObsHardTrackTerms
from robotlib.robot_keys.g1_29d import g1_key_body_names, g1_anchor_name
from amp_tasks.amp_task_demo_data_cfg import file_punch

@configclass
class G1FlatBMPRunnerCfg(BMPRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 10000
    save_interval = 500
    experiment_name = "g1_flat_demo"
    run_name = "bmp"
    empirical_normalization = True
    policy = RslRlPpoActorCriticCfg(
        class_name= "ActorCriticMulti",
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = BMPPPOAlgorithmCfg(
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
    amp_data = MotionDatasetCfg(
        motion_files=[file_punch],
        body_names = g1_key_body_names,
        anchor_name = g1_anchor_name,
        amp_obs_terms = AMPObsBaiscTerms,
    )
    amp_discr_hidden_dims = [256, 256]
    amp_reward_coef = 0.5
    amp_task_reward_lerp = 0.7
    