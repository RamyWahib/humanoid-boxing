# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass
from beyondAMP.isaaclab.configs.rl_cfg import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg
from beyondAMP.isaaclab.configs.amp_cfg import AMPDataCfg, AMPObsBaiscCfg, AMPPPOAlgorithmCfg, AMPRunnerCfg
from beyondAMP.amp_obs_grp import AMPObsSoftTrackTerms

from .config import g1_key_body_names

from beyondAMP.amp_obs_grp import AMPObsBaiscTerms

@configclass
class BasePPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 50000
    save_interval = 250
    experiment_name = "g1_loco"  # same as task name
    run_name = "cliped_with_lin"
    empirical_normalization = False
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
        entropy_coef=0.01,
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
    max_iterations = 30000
    save_interval = 500
    experiment_name = "g1_loco"
    run_name = "amp"
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
            "data/datasets/AMASS7/B4_-_Stand_to_Walk_backwards_stageii.npz",
            "data/datasets/AMASS7/B9_-__Walk_turn_left_90_stageii.npz",
            "data/datasets/AMASS7/B10_-__Walk_turn_left_45_stageii.npz",
            "data/datasets/AMASS7/B11_-__Walk_turn_left_135_stageii.npz",
            "data/datasets/AMASS7/B13_-__Walk_turn_right_90_stageii.npz",
            "data/datasets/AMASS7/B14_-__Walk_turn_right_45_t2_stageii.npz",
            "data/datasets/AMASS7/B15_-__Walk_turn_around_stageii.npz",
            "data/datasets/AMASS7/B22_-__side_step_left_stageii.npz",
            "data/datasets/AMASS7/B23_-__side_step_right_stageii.npz",
            "data/datasets/AMASS7/C1_-_stand_to_run_stageii.npz",
            "data/datasets/AMASS7/C3_-_run_stageii.npz",
            "data/datasets/AMASS7/C4_-_run_to_walk_a_stageii.npz",
            "data/datasets/AMASS7/C5_-_walk_to_run_stageii.npz",
            "data/datasets/AMASS7/C6_-_stand_to_run_backwards_stageii.npz",
            "data/datasets/AMASS7/C8_-_run_backwards_to_stand_stageii.npz",
            "data/datasets/AMASS7/C9_-_run_backwards_turn_run_forward_stageii.npz",
            "data/datasets/AMASS7/C11_-_run_turn_left_90_stageii.npz",
            "data/datasets/AMASS7/C12_-_run_turn_left_45_stageii.npz",
            "data/datasets/AMASS7/C13_-_run_turn_left_135_stageii.npz",
            "data/datasets/AMASS7/C14_-_run_turn_right_90_stageii.npz",
            "data/datasets/AMASS7/C15_-_run_turn_right_45_stageii.npz",
            "data/datasets/AMASS7/C16_-_run_turn_right_135_stageii.npz",
            "data/datasets/AMASS7/C17_-_run_change_direction_stageii.npz",
            "data/datasets/AMASS7/Walk_B4_-_Stand_to_Walk_Back_stageii.npz",
            "data/datasets/AMASS7/Walk_B10_-_Walk_turn_left_45_stageii.npz",
            "data/datasets/AMASS7/Walk_B13_-_Walk_turn_right_45_stageii.npz",
            "data/datasets/AMASS7/Walk_B15_-_Walk_turn_around_stageii.npz",
            "data/datasets/AMASS7/Walk_B16_-_Walk_turn_change_stageii.npz",
            "data/datasets/AMASS7/Walk_B22_-_Side_step_left_stageii.npz",
            "data/datasets/AMASS7/Walk_B23_-_Side_step_right_stageii.npz",
        ],
        body_names = g1_key_body_names,
        amp_obs_terms = AMPObsBaiscTerms
    )
    amp_discr_hidden_dims = [256, 256]
    amp_reward_coef = 0.6   # 风格奖励缩放系数（0.5为基准值）
    amp_task_reward_lerp = 0.35  # 40%任务 + 60%风格
