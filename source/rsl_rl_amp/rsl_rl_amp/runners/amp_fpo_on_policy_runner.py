import os
import statistics
import time
from collections import deque

import torch
from torch.utils.tensorboard import SummaryWriter

from beyondAMP.isaaclab.rsl_rl.amp_wrapper import AMPEnvWrapper
from rsl_rl_amp.algorithms import AMPFPO
from rsl_rl_amp.modules import ActorCriticFPO, ActorCriticRecurrent
from rsl_rl_amp.modules.amp_discriminator import AMPDiscriminator
from rsl_rl_amp.utils.utils import Normalizer


class AMPFPOOnPolicyRunner:
    """On-policy runner for AMPFPO.

    Differences vs `AMPOnPolicyRunner`:
      - Keeps env task reward as-is (no reward mixing).
      - Uses discriminator output to define feasibility/constraint (h).
      - FPO-style region-wise objective inside `AMPFPO.update()`.
    """

    def __init__(self, env, train_cfg, log_dir=None, device="cpu"):
        self.cfg = train_cfg
        self.alg_cfg = train_cfg["algorithm"]
        self.policy_cfg = train_cfg["policy"]
        self.amp_data_cfg = train_cfg["amp_data"]
        self.device = device
        self.env: AMPEnvWrapper = env

        if self.env.num_privileged_obs is not None:
            num_critic_obs = self.env.num_privileged_obs
        else:
            num_critic_obs = self.env.num_obs
        num_actor_obs = self.env.num_obs

        actor_critic_class = eval(self.policy_cfg["class_name"])
        actor_critic = actor_critic_class(
            num_actor_obs=num_actor_obs,
            num_critic_obs=num_critic_obs,
            num_actions=self.env.num_actions,
            **self.policy_cfg,
        ).to(self.device)

        # AMP discriminator (state, next_state) features
        amp_obs_dim = env.get_amp_observations().shape[-1]
        amp_normalizer = Normalizer(amp_obs_dim)
        discriminator = AMPDiscriminator(
            amp_obs_dim * 2,
            train_cfg["amp_reward_coef"],
            train_cfg["amp_discr_hidden_dims"],
            device,
            train_cfg.get("amp_task_reward_lerp", 0.0),
        ).to(self.device)

        # algorithm
        alg_class = eval(self.alg_cfg["class_name"])
        min_std = (
            torch.tensor(self.cfg["amp_min_normalized_std"], device=self.device)
            * (torch.abs(self.env.dof_pos_limits[0, :, 1] - self.env.dof_pos_limits[0, :, 0]))
        )
        self.alg: AMPFPO = alg_class(
            actor_critic,
            discriminator,
            env.motion_dataset,
            amp_normalizer,
            device=self.device,
            min_std=min_std,
            **self.alg_cfg,
        )

        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        self.save_interval = self.cfg["save_interval"]

        # init storage
        self.alg.init_storage(
            self.env.num_envs,
            self.num_steps_per_env,
            [num_actor_obs],
            [self.env.num_privileged_obs],
            [self.env.num_actions],
        )

        self.log_dir = log_dir
        self.writer = None
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0

        _, _ = self.env.reset()

    def learn(self, num_learning_iterations, init_at_random_ep_len=False):
        if self.log_dir is not None and self.writer is None:
            self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )

        obs = self.env.get_observations()
        privileged_obs = self.env.get_privileged_observations()
        amp_obs = self.env.get_amp_observations()
        critic_obs = privileged_obs if privileged_obs is not None else obs
        obs, critic_obs, amp_obs = obs.to(self.device), critic_obs.to(self.device), amp_obs.to(self.device)

        self.alg.actor_critic.train()
        self.alg.discriminator.train()

        ep_infos = []
        # NOTE: mean_episode_length can look extremely jittery if computed over too few episodes,
        # especially when there is a mixture of early terminations (e.g., bad_orientation) vs timeouts.
        # Use a larger buffer to reduce pure reporting noise.
        rewbuffer = deque(maxlen=1000)
        lenbuffer = deque(maxlen=1000)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        # Stage-wise training: track disc_prob and violation for entropy scheduling
        # When disc_prob_mean is high and violation_mean is low (stable), reduce entropy
        # This is more reliable than in_region_ratio since it's based on discriminator output directly
        original_entropy_coef = self.alg.entropy_coef
        entropy_reduced = False
        disc_prob_history = deque(maxlen=50)  # Track last 50 iterations
        violation_history = deque(maxlen=50)  # Track last 50 iterations
        
        # Value loss explosion detection: track for early stopping
        value_loss_task_history = deque(maxlen=10)  # Track last 10 iterations
        value_loss_explosion_threshold = 1e3  # If value loss > 1e3, likely explosion
        value_loss_explosion_detected = False

        tot_iter = self.current_learning_iteration + num_learning_iterations
        for it in range(self.current_learning_iteration, tot_iter):
            start = time.time()
            episodes_finished_iter = 0
            # per-iteration feasibility stats (averaged over env steps)
            disc_prob_means = []
            violation_means = []
            in_ratio_means = []
            with torch.inference_mode():
                for _ in range(self.num_steps_per_env):
                    actions = self.alg.act(obs, critic_obs, amp_obs)
                    obs, privileged_obs, task_rewards, dones, infos, reset_env_ids, terminal_amp_states = self.env.step(
                        actions, not_amp=False
                    )
                    next_amp_obs = self.env.get_amp_observations()

                    critic_obs = privileged_obs if privileged_obs is not None else obs
                    obs = obs.to(self.device)
                    critic_obs = critic_obs.to(self.device)
                    next_amp_obs = next_amp_obs.to(self.device)
                    task_rewards = task_rewards.to(self.device)
                    dones = dones.to(self.device)

                    # terminal AMP states handling
                    next_amp_obs_with_term = torch.clone(next_amp_obs)
                    next_amp_obs_with_term[reset_env_ids] = terminal_amp_states

                    # store transition (task reward + disc-constraint)
                    disc_p_mean, viol_mean, in_ratio_mean = self.alg.process_env_step(
                        task_rewards, dones, infos, next_amp_obs_with_term
                    )
                    disc_prob_means.append(disc_p_mean)
                    violation_means.append(viol_mean)
                    in_ratio_means.append(in_ratio_mean)

                    amp_obs = torch.clone(next_amp_obs)

                    if self.log_dir is not None:
                        if "episode" in infos:
                            ep_infos.append(infos["episode"])
                        if "log" in infos:
                            ep_infos.append(infos["log"])
                        cur_reward_sum += task_rewards
                        cur_episode_length += 1
                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        episodes_finished_iter += int(new_ids.numel())
                        rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                        lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                        cur_reward_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0

                stop = time.time()
                collection_time = stop - start
                start = stop
                self.alg.compute_returns(critic_obs)

            (
                mean_value_loss_task,
                mean_value_loss_repair,
                mean_surrogate_loss,
                mean_amp_loss,
                mean_grad_pen_loss,
                mean_policy_pred,
                mean_expert_pred,
                mean_in_ratio,
                mean_task_adv,
                std_task_adv,
            ) = self.alg.update()

            # ===== IMPROVEMENT 0: Adaptive feasible_threshold scheduling =====
            # Keep in_region_ratio in a trainable band to avoid critic starvation.
            if in_ratio_means:
                current_in_ratio = statistics.mean(in_ratio_means)
                old_thr = float(getattr(self.alg, "feasible_threshold", 0.0))
                new_thr = float(self.alg.maybe_update_feasible_threshold(current_in_ratio))
                if self.writer is not None:
                    self.writer.add_scalar("Train/feasible_threshold", new_thr, it)
                if abs(new_thr - old_thr) > 1e-8:
                    print(
                        f"[ADAPT-THRESH] iter={it} in_region_ratio={current_in_ratio:.3f} "
                        f"feasible_threshold: {old_thr:.3f} -> {new_thr:.3f}"
                    )

            stop = time.time()
            learn_time = stop - start

            # ===== IMPROVEMENT 1: Value loss explosion detection =====
            value_loss_task_history.append(mean_value_loss_task)
            if len(value_loss_task_history) >= 3:
                recent_max = max(list(value_loss_task_history)[-3:])
                if recent_max > value_loss_explosion_threshold and not value_loss_explosion_detected:
                    value_loss_explosion_detected = True
                    print(f"\n{'='*80}")
                    print(f"[CRITICAL] Value loss explosion detected at iteration {it}!")
                    print(f"  Recent max value_loss_task: {recent_max:.2e} (threshold: {value_loss_explosion_threshold:.2e})")
                    print(f"  This usually indicates training instability.")
                    print(f"  Consider: stopping training, rolling back to previous checkpoint,")
                    print(f"  or reducing learning rate / entropy coefficient.")
                    print(f"{'='*80}\n")
                    if self.writer is not None:
                        self.writer.add_scalar("Train/value_loss_explosion_detected", 1.0, it)

            # ===== IMPROVEMENT 2: Stage-wise entropy reduction =====
            # Use disc_prob_mean + violation_mean trends instead of in_region_ratio
            # This is more reliable since disc_prob/violation directly reflect style quality,
            # while in_region_ratio depends on discriminator threshold which may change
            if len(disc_prob_means) > 0 and len(violation_means) > 0:
                current_disc_prob = statistics.mean(disc_prob_means)
                current_violation = statistics.mean(violation_means)
                disc_prob_history.append(current_disc_prob)
                violation_history.append(current_violation)
                
                if (not entropy_reduced and 
                    len(disc_prob_history) >= 20 and 
                    it >= 1000):  # Only after some initial training
                    # Check if disc_prob is high and stable (good style similarity)
                    recent_disc_probs = list(disc_prob_history)[-20:]
                    mean_disc_prob = statistics.mean(recent_disc_probs)
                    std_disc_prob = statistics.stdev(recent_disc_probs) if len(recent_disc_probs) > 1 else 0.0
                    
                    # Check if violation is low and stable (few constraint violations)
                    recent_violations = list(violation_history)[-20:]
                    mean_violation = statistics.mean(recent_violations)
                    std_violation = statistics.stdev(recent_violations) if len(recent_violations) > 1 else 0.0
                    
                    # Criteria: disc_prob >= 0.6 (good style), violation <= 0.2 (low violations),
                    # both stable (std < 0.05)
                    disc_prob_good = mean_disc_prob >= 0.6 and std_disc_prob < 0.05
                    violation_low = mean_violation <= 0.2 and std_violation < 0.05
                    
                    if disc_prob_good and violation_low:
                        # Reduce entropy coefficient by 50% to make policy more deterministic
                        new_entropy_coef = original_entropy_coef * 0.5
                        self.alg.entropy_coef = new_entropy_coef
                        entropy_reduced = True
                        print(f"\n[STAGE-WISE TRAINING] Reducing entropy coefficient at iteration {it}")
                        print(f"  disc_prob_mean: {mean_disc_prob:.3f} ± {std_disc_prob:.3f} (stable, good)")
                        print(f"  violation_mean: {mean_violation:.3f} ± {std_violation:.3f} (stable, low)")
                        print(f"  entropy_coef: {original_entropy_coef:.6f} -> {new_entropy_coef:.6f}")
                        if self.writer is not None:
                            self.writer.add_scalar("Train/entropy_coef", new_entropy_coef, it)
                            self.writer.add_scalar("Train/stage_entropy_reduced", 1.0, it)

            if self.log_dir is not None:
                # Feasibility statistics (more actionable than in_region_ratio alone)
                if self.writer is not None and disc_prob_means:
                    self.writer.add_scalar("Train/episodes_finished_iter", float(episodes_finished_iter), it)
                    self.writer.add_scalar("Train/disc_prob_mean", statistics.mean(disc_prob_means), it)
                    self.writer.add_scalar("Train/violation_mean", statistics.mean(violation_means), it)
                    self.writer.add_scalar("Train/in_region_ratio_rollout", statistics.mean(in_ratio_means), it)
                self.log(locals())
            if it % self.save_interval == 0:
                self.save(os.path.join(self.log_dir, f"model_{it}.pt"))
            ep_infos.clear()

        self.current_learning_iteration += num_learning_iterations
        self.save(os.path.join(self.log_dir, f"model_{self.current_learning_iteration}.pt"))

    def log(self, locs, width=80, pad=35):
        self.tot_timesteps += self.num_steps_per_env * self.env.num_envs
        self.tot_time += locs["collection_time"] + locs["learn_time"]
        iteration_time = locs["collection_time"] + locs["learn_time"]

        ep_string = ""
        if locs["ep_infos"]:
            for key in locs["ep_infos"][0]:
                infotensor = torch.tensor([], device=self.device)
                for ep_info in locs["ep_infos"]:
                    if not isinstance(ep_info[key], torch.Tensor):
                        ep_info[key] = torch.Tensor([ep_info[key]])
                    if len(ep_info[key].shape) == 0:
                        ep_info[key] = ep_info[key].unsqueeze(0)
                    infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
                value = torch.mean(infotensor)
                self.writer.add_scalar("Episode/" + key, value, locs["it"])
                ep_string += f"""{f"Mean episode {key}:":>{pad}} {value:.4f}\n"""

        mean_std = self.alg.actor_critic.std.mean()
        fps = int(
            self.num_steps_per_env
            * self.env.num_envs
            / (locs["collection_time"] + locs["learn_time"])
        )

        self.writer.add_scalar("Loss/value_function_task", locs["mean_value_loss_task"], locs["it"])
        self.writer.add_scalar("Loss/value_function_repair", locs["mean_value_loss_repair"], locs["it"])
        self.writer.add_scalar("Loss/surrogate", locs["mean_surrogate_loss"], locs["it"])
        self.writer.add_scalar("Loss/AMP", locs["mean_amp_loss"], locs["it"])
        self.writer.add_scalar("Loss/AMP_grad", locs["mean_grad_pen_loss"], locs["it"])
        self.writer.add_scalar("Loss/learning_rate", self.alg.learning_rate, locs["it"])
        self.writer.add_scalar("Policy/mean_noise_std", mean_std.item(), locs["it"])
        self.writer.add_scalar("Train/in_region_ratio", locs["mean_in_ratio"], locs["it"])
        
        # Diagnostic metrics for episode_length issue
        self.writer.add_scalar("Train/task_advantage_mean", locs.get("mean_task_adv", 0.0), locs["it"])
        self.writer.add_scalar("Train/task_advantage_std", locs.get("std_task_adv", 0.0), locs["it"])
        self.writer.add_scalar("Train/task_critic_recovery_active", 
                               1.0 if getattr(self.alg, "_task_critic_recovery_active", False) else 0.0, 
                               locs["it"])
        
        # IMPROVEMENT: Log value loss explosion warning if detected
        if locs.get("value_loss_explosion_detected", False):
            self.writer.add_scalar("Train/value_loss_explosion_warning", 1.0, locs["it"])

        self.writer.add_scalar("Perf/total_fps", fps, locs["it"])
        self.writer.add_scalar("Perf/collection time", locs["collection_time"], locs["it"])
        self.writer.add_scalar("Perf/learning_time", locs["learn_time"], locs["it"])

        if len(locs["rewbuffer"]) > 0:
            self.writer.add_scalar("Train/mean_reward", statistics.mean(locs["rewbuffer"]), locs["it"])
            self.writer.add_scalar("Train/mean_episode_length", statistics.mean(locs["lenbuffer"]), locs["it"])

        title = f" \033[1m Learning iteration {locs['it']}/{self.current_learning_iteration + locs['num_learning_iterations']} \033[0m "

        # IMPROVEMENT: Highlight feasibility metrics (more important than reward for FPO)
        vf_task_str = f"{locs['mean_value_loss_task']:.4f}"
        if locs.get("value_loss_explosion_detected", False):
            vf_task_str = f"{vf_task_str} [EXPLOSION WARNING!]"
        
        log_string = (f"""{'#' * width}\n"""
                      f"""{title.center(width, ' ')}\n\n"""
                      f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs['collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                      f"""{'Value loss (task):':>{pad}} {vf_task_str}\n"""
                      f"""{'Value loss (repair):':>{pad}} {locs['mean_value_loss_repair']:.4f}\n"""
                      f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                      f"""{'AMP loss:':>{pad}} {locs['mean_amp_loss']:.4f}\n"""
                      f"""{'AMP grad pen loss:':>{pad}} {locs['mean_grad_pen_loss']:.4f}\n"""
                      f"""{'AMP mean policy pred:':>{pad}} {locs['mean_policy_pred']:.4f}\n"""
                      f"""{'AMP mean expert pred:':>{pad}} {locs['mean_expert_pred']:.4f}\n"""
                      f"""{'In-region ratio:':>{pad}} {locs['mean_in_ratio']:.3f} [KEY METRIC]\n"""
                      f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f} [KEY METRIC]\n""")

        if len(locs["rewbuffer"]) > 0:
            log_string += (
                f"""{'Mean task reward:':>{pad}} {statistics.mean(locs['rewbuffer']):.2f}\n"""
                f"""{'Mean episode length:':>{pad}} {statistics.mean(locs['lenbuffer']):.2f}\n"""
            )

        log_string += ep_string
        log_string += (
            f"""{'-' * width}\n"""
            f"""{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"""
            f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
            f"""{'Total time:':>{pad}} {self.tot_time:.2f}s\n"""
            f"""{'ETA:':>{pad}} {self.tot_time / (locs['it'] + 1) * (locs['num_learning_iterations'] - locs['it']):.1f}s\n"""
        )
        print(log_string)

    def save(self, path, infos=None):
        torch.save(
            {
                "model_state_dict": self.alg.actor_critic.state_dict(),
                "optimizer_state_dict": self.alg.optimizer.state_dict(),
                "discriminator_state_dict": self.alg.discriminator.state_dict(),
                "amp_normalizer": self.alg.amp_normalizer,
                "iter": self.current_learning_iteration,
                "infos": infos,
            },
            path,
        )

    def load(self, path, load_optimizer=True, reduce_lr_on_resume=True, lr_reduction_factor=5.0):
        """Load checkpoint with improved stability for resume training.
        
        Args:
            path: Path to checkpoint file
            load_optimizer: Whether to load optimizer state (default: True)
            reduce_lr_on_resume: If True and load_optimizer=False, reduce learning rate
                to prevent instability from mismatched optimizer state (default: True)
            lr_reduction_factor: Factor to reduce learning rate by if reduce_lr_on_resume=True
                (default: 5.0, i.e., reduce to 1/5 of original)
        """
        loaded_dict = torch.load(path, weights_only=False)
        self.alg.actor_critic.load_state_dict(loaded_dict["model_state_dict"])
        self.alg.discriminator.load_state_dict(loaded_dict["discriminator_state_dict"])
        self.alg.amp_normalizer = loaded_dict["amp_normalizer"]
        
        if load_optimizer:
            self.alg.optimizer.load_state_dict(loaded_dict["optimizer_state_dict"])
        elif reduce_lr_on_resume:
            # Reduce learning rate to prevent instability from mismatched optimizer state
            original_lr = self.alg.learning_rate
            new_lr = original_lr / lr_reduction_factor
            self.alg.learning_rate = new_lr
            for param_group in self.alg.optimizer.param_groups:
                param_group["lr"] = new_lr
            print(f"[RESUME] Reduced learning rate from {original_lr:.6f} to {new_lr:.6f} "
                  f"(factor: {lr_reduction_factor:.1f}x) to improve stability")
        
        # Restore current_learning_iteration from checkpoint
        if "iter" in loaded_dict:
            self.current_learning_iteration = loaded_dict["iter"]
            print(f"[RESUME] Restored learning iteration: {self.current_learning_iteration}")
        
        return loaded_dict.get("infos", None)

    def get_inference_policy(self, device=None):
        self.alg.actor_critic.eval()
        if device is not None:
            self.alg.actor_critic.to(device)
        return self.alg.actor_critic.act_inference




