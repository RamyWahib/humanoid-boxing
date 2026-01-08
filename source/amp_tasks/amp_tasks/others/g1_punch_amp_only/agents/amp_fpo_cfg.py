from isaaclab.utils import configclass
from beyondAMP.isaaclab.rsl_rl.configs.rl_cfg import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg
from beyondAMP.isaaclab.rsl_rl.configs.amp_cfg import \
    MotionDatasetCfg, AMPObsBaiscCfg, AMPPPOAlgorithmCfg, AMPRunnerCfg, AMPPPOWeightedAlgorithmCfg, \
    AMPFPORunnerCfg, AMPFPOAlgorithmCfgNew, ActorCriticFPOCfg

from beyondAMP.obs_groups import AMPObsBaiscTerms, AMPObsSoftTrackTerms, AMPObsHardTrackTerms

from robotlib.robot_keys.g1_29d import g1_key_body_names, g1_anchor_name
from amp_tasks.amp_task_demo_data_cfg import file_punch

@configclass
class G1FlatAMPFPORunnerCfg(AMPFPORunnerCfg):
    """Base configuration for AMPFPO (FPO-style region-wise optimization)."""
    num_steps_per_env = 24
    max_iterations = 30000
    save_interval = 500
    experiment_name = "g1_flat_demo"
    run_name = "fpo"
    empirical_normalization = True
    policy = ActorCriticFPOCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],  # Used for both task and repair critics
        activation="elu",
        use_recover_critic=True,  # Default is False, same as previous successful run
        recover_critic_hidden_dims=[512, 256, 128],
    )
    algorithm = AMPFPOAlgorithmCfgNew(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,  # FIXED: Back to 0.005 (experiment 1 value) - 0.015 was too high and caused noise explosion
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        # FPO-specific parameters
        feasible_threshold=0.3,  # FIXED: Back to 0.3 (experiment 1 value) - 0.2 was too permissive
        disc_prob_fn="amp_reward",
        repair_alpha=5.0,  # Strength coefficient for repair reward
        repair_power=1.0,  # Power for violation-based repair reward
        # CRITICAL FIX: Add positive style reward for in-region transitions
        # This compensates for negative task rewards, similar to AMP reward mixing in Basic
        # Basic uses amp_reward_coef=1.0 with lerp=0.05, so AMP contributes ~0.95 per step
        # Setting this to 0.5-1.0 provides similar positive signal
        in_region_style_reward_coef=0.5,  # Positive reward for being in feasible set (in-region only)
        use_recover_critic=True,
    )
    amp_data = MotionDatasetCfg(
        motion_files=[file_punch],
        body_names = g1_key_body_names,
        anchor_name = g1_anchor_name,
        amp_obs_terms = None,
    )
    amp_discr_hidden_dims = [256, 256]
    amp_reward_coef = 0.5  # Used for discriminator initialization, not for reward mixing
    amp_task_reward_lerp = 0.0  # Not used in FPO (task reward is kept separate)