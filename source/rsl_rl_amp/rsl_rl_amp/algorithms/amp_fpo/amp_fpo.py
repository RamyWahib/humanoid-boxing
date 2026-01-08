from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim

from rsl_rl_amp.modules.amp_discriminator import AMPDiscriminator
from rsl_rl_amp.storage.fpo_rollout_storage import FPORolloutStorage, masked_mean
from rsl_rl_amp.storage.replay_buffer import ReplayBuffer


class AMPFPO:
    """AMP + PPO with FPO-style region-wise objective (single shared actor).

    Key changes vs traditional AMP (reward-mixing):
      - Discriminator defines feasibility via a constraint score (no manual reward weighting).
      - In-Set transitions: optimize task return only.
      - Out-Set transitions: optimize repair/style recovery only.
      - Actor is shared, with a region-wise advantage `adv = I_in*Adv_task + I_out*Adv_repair`.
    """

    def __init__(
        self,
        actor_critic,
        discriminator: AMPDiscriminator,
        amp_data,
        amp_normalizer,
        # PPO
        num_learning_epochs: int = 1,
        num_mini_batches: int = 1,
        clip_param: float = 0.2,
        gamma: float = 0.998,
        lam: float = 0.95,
        value_loss_coef: float = 1.0,
        repair_value_loss_coef: float | None = None,
        entropy_coef: float = 0.0,
        learning_rate: float = 1e-3,
        max_grad_norm: float = 1.0,
        use_clipped_value_loss: bool = True,
        schedule: str = "fixed",
        desired_kl: float = 0.01,
        device: str = "cpu",
        # AMP replay
        amp_replay_buffer_size: int = 100000,
        # feasibility / constraint
        feasible_threshold: float = 0.7,
        # How to convert discriminator output `d` (unbounded logit) into a bounded feasibility score p∈[0,1].
        # IMPORTANT: AMP discriminator is trained with MSE targets expert=+1 / policy=-1, but `d` can be unbounded.
        # We must NOT treat large d>1 as "more expert-like" (often indicates discriminator instability/overshoot).
        # We therefore use the same shaping as AMP's reward conversion: clamp(1 - 1/4*(d-1)^2, 0, 1).
        disc_prob_fn: str = "amp_reward", 
        # adaptive thresholding (runner can call maybe_update_feasible_threshold())
        adaptive_feasible_threshold: bool = False,
        target_in_ratio_low: float = 0.2,
        target_in_ratio_high: float = 0.6,
        threshold_adjust_rate: float = 0.01,
        feasible_threshold_min: float = 0.05,
        feasible_threshold_max: float = 0.95,
        # repair reward shaping (adaptive constraint strength)
        repair_alpha: float = 1.0,
        repair_power: float = 1.0,
        gamma_repair: float | None = None,
        lam_repair: float | None = None,
        # repair stream semantics (only "shaping" mode supported for switch advantage)
        repair_reward_mode: str = "shaping",  # "shaping" only (weighted_fpo removed)
        # recover critic (hard violation recovery)
        use_recover_critic: bool = False,
        recover_critic_lr: float = 1e-3,
        recover_value_loss_coef: float | None = None,
        hard_violation_threshold: float = 0.1,  # disc_prob < this => hard violation
        recover_reward_coef: float = 1.0,
        gamma_recover: float | None = None,
        lam_recover: float | None = None,
        # in-region style reward (positive reward for being in feasible set)
        # This provides positive signal when task_reward is mostly negative
        in_region_style_reward_coef: float = 0.0,  # If > 0, add positive reward based on disc_prob for in-region transitions
        # value loss stabilization (avoid region starvation early)
        value_mix_warmup_iters: int = 0,
        value_mix_in_ratio_low: float = 0.15,
        value_mix_in_ratio_high: float = 0.85,
        value_mix_task_out_weight: float = 0.1,
        value_mix_repair_in_weight: float = 0.1,
        min_region_fraction: float = 0.02,
        # task critic recovery: when in_region_ratio rises, force full-sample training
        task_critic_recovery_threshold: float = 0.4,  # Trigger recovery when in_ratio >= this
        task_critic_recovery_iters: int = 100,  # How many iterations to use full-sample training
        # discriminator update schedule (reduce coupling/oscillation)
        disc_update_start_iter: int = 0,
        disc_update_every: int = 1,
        min_std=None,
        **kwargs,
    ):
        if kwargs:
            # keep config-compatible; ignore unknown keys
            print("AMPFPO.__init__ got unexpected arguments (ignored): " + str(list(kwargs.keys())))

        self.device = device
        self.desired_kl = desired_kl
        self.schedule = schedule
        self.learning_rate = learning_rate
        self.min_std = min_std

        self.clip_param = clip_param
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.value_loss_coef = value_loss_coef
        self.repair_value_loss_coef = repair_value_loss_coef if repair_value_loss_coef is not None else value_loss_coef
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.lam = lam
        self.gamma_repair = gamma_repair if gamma_repair is not None else gamma
        self.lam_repair = lam_repair if lam_repair is not None else lam
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss

        # Feasibility / constraint
        self.feasible_threshold = float(feasible_threshold)
        self.disc_prob_fn = disc_prob_fn
        self.adaptive_feasible_threshold = bool(adaptive_feasible_threshold)
        self.target_in_ratio_low = float(target_in_ratio_low)
        self.target_in_ratio_high = float(target_in_ratio_high)
        self.threshold_adjust_rate = float(threshold_adjust_rate)
        self.feasible_threshold_min = float(feasible_threshold_min)
        self.feasible_threshold_max = float(feasible_threshold_max)

        self.repair_alpha = float(repair_alpha)
        self.repair_power = float(repair_power)
        self.in_region_style_reward_coef = float(in_region_style_reward_coef)
        self.repair_reward_mode = str(repair_reward_mode)
        if self.repair_reward_mode != "shaping":
            raise ValueError(f"repair_reward_mode must be 'shaping' (weighted_fpo removed, only switch mode supported)")

        # Recover critic (optional)
        self.use_recover_critic = bool(use_recover_critic)
        self.hard_violation_threshold = float(hard_violation_threshold)
        self.recover_reward_coef = float(recover_reward_coef)
        self.gamma_recover = gamma_recover if gamma_recover is not None else self.gamma_repair
        self.lam_recover = lam_recover if lam_recover is not None else self.lam_repair
        self.recover_value_loss_coef = (
            float(recover_value_loss_coef) if recover_value_loss_coef is not None else float(value_loss_coef)
        )

        # Value loss stabilization
        self.value_mix_warmup_iters = int(value_mix_warmup_iters)
        self.value_mix_in_ratio_low = float(value_mix_in_ratio_low)
        self.value_mix_in_ratio_high = float(value_mix_in_ratio_high)
        self.value_mix_task_out_weight = float(value_mix_task_out_weight)
        self.value_mix_repair_in_weight = float(value_mix_repair_in_weight)
        self.min_region_fraction = float(min_region_fraction)
        
        # Task critic recovery
        self.task_critic_recovery_threshold = float(task_critic_recovery_threshold)
        self.task_critic_recovery_iters = int(task_critic_recovery_iters)
        self._task_critic_recovery_active = False
        self._task_critic_recovery_start_iter = None

        # Discriminator schedule
        self.disc_update_start_iter = int(disc_update_start_iter)
        self.disc_update_every = int(disc_update_every)
        if self.disc_update_every <= 0:
            raise ValueError("disc_update_every must be >= 1")

        # Internal counters
        self._update_iter = 0

        # Networks
        self.actor_critic = actor_critic.to(self.device)
        if not (hasattr(self.actor_critic, "evaluate_task") and hasattr(self.actor_critic, "evaluate_repair")):
            raise ValueError(
                "AMPFPO requires an actor-critic with `evaluate_task()` and `evaluate_repair()` "
                "(use rsl_rl_amp.modules.ActorCriticFPO)."
            )
        self.discriminator = discriminator.to(self.device)
        self.amp_data = amp_data
        self.amp_normalizer = amp_normalizer

        # AMP replay buffer (policy transitions)
        self.amp_transition = FPORolloutStorage.Transition()
        self.amp_storage = ReplayBuffer(discriminator.input_dim // 2, amp_replay_buffer_size, device)

        # PPO rollout storage
        self.storage: FPORolloutStorage | None = None
        self.transition = FPORolloutStorage.Transition()

        # Optimizer: actor_critic + discriminator (same as AMPPPO for simplicity)
        # If we use a separate recover critic optimizer, exclude recover_critic params from the main optimizer to avoid
        # double-updates.
        recover_param_ids = set()
        if self.use_recover_critic and hasattr(self.actor_critic, "recover_critic") and self.actor_critic.recover_critic is not None:
            recover_param_ids = {id(p) for p in self.actor_critic.recover_critic.parameters()}

        actor_critic_main_params = [p for p in self.actor_critic.parameters() if id(p) not in recover_param_ids]
        # Keep a stable reference for correct grad clipping (avoid including recover_critic grads in PPO update).
        self._actor_critic_main_params = actor_critic_main_params

        params = [
            {"params": actor_critic_main_params, "name": "actor_critic"},
            {"params": self.discriminator.trunk.parameters(), "weight_decay": 10e-4, "name": "amp_trunk"},
            {"params": self.discriminator.amp_linear.parameters(), "weight_decay": 10e-2, "name": "amp_head"},
        ]
        self.optimizer = optim.Adam(params, lr=learning_rate)

        self.recover_optimizer = None
        if self.use_recover_critic:
            if not hasattr(self.actor_critic, "evaluate_recover"):
                raise ValueError(
                    "AMPFPO(use_recover_critic=True) requires ActorCriticFPO with recover_critic enabled "
                    "(use_recover_critic=True in policy config)."
                )
            if hasattr(self.actor_critic, "recover_critic") and self.actor_critic.recover_critic is not None:
                self.recover_optimizer = optim.Adam(self.actor_critic.recover_critic.parameters(), lr=float(recover_critic_lr))

    # ---------------------------
    # Storage / mode
    # ---------------------------
    def init_storage(self, num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, action_shape):
        self.storage = FPORolloutStorage(
            num_envs,
            num_transitions_per_env,
            actor_obs_shape,
            critic_obs_shape,
            action_shape,
            self.device,
        )

    def test_mode(self):
        self.actor_critic.eval()
        self.discriminator.eval()

    def train_mode(self):
        self.actor_critic.train()
        self.discriminator.train()

    # ---------------------------
    # Feasibility helpers
    # ---------------------------
    def _disc_to_prob(self, d: torch.Tensor) -> torch.Tensor:

        if self.disc_prob_fn in ("amp_reward"):
            return torch.clamp(1.0 - 0.25 * torch.square(d - 1.0), 0.0, 1.0)

    @torch.no_grad()
    def _compute_disc_prob(self, amp_state: torch.Tensor, amp_next_state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # normalize AMP features if needed
        if self.amp_normalizer is not None:
            amp_state = self.amp_normalizer.normalize_torch(amp_state, self.device)
            amp_next_state = self.amp_normalizer.normalize_torch(amp_next_state, self.device)
        inp = torch.cat([amp_state, amp_next_state], dim=-1)
        self.discriminator.eval()
        d = self.discriminator(inp)
        p = self._disc_to_prob(d)
        self.discriminator.train()
        return p.squeeze(-1), d.squeeze(-1)

    def _repair_reward_from_prob(self, p: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (repair_reward, violation). Out-set only, in-set => 0."""
        # Map to bounded cost/violation signal for stable scaling.
        # cost in [0, 1]: 0 => feasible, 1 => strongly infeasible.
        cost = torch.clamp(self.feasible_threshold - p, min=0.0, max=1.0)
        # Repair reward: negative shaping, magnitude controlled by repair_alpha and repair_power.
        # power=1 => -alpha*cost, power=2 => -alpha*cost^2
        repair_reward = -self.repair_alpha * torch.pow(cost + 1e-8, self.repair_power)
        return repair_reward, cost

    def maybe_update_feasible_threshold(self, in_region_ratio: float) -> float:
        """Optionally adapt feasible_threshold to keep in_region_ratio in a trainable band.

        This is intentionally simple (OmniSafe-like threshold adjust). The runner should pass
        an iteration-level in_region_ratio (e.g., mean over rollout steps).

        Returns the (possibly updated) feasible_threshold.
        """
        if not self.adaptive_feasible_threshold:
            return self.feasible_threshold

        r = float(in_region_ratio)
        thr = self.feasible_threshold
        if r < self.target_in_ratio_low:
            # Too few in-region samples -> relax feasibility (lower threshold since higher p is better).
            thr = thr - self.threshold_adjust_rate
        elif r > self.target_in_ratio_high:
            # Too many in-region samples -> tighten feasibility.
            thr = thr + self.threshold_adjust_rate

        thr = float(max(self.feasible_threshold_min, min(self.feasible_threshold_max, thr)))
        self.feasible_threshold = thr
        return thr

    # ---------------------------
    # Rollout API
    # ---------------------------
    def act(self, obs, critic_obs, amp_obs):
        if getattr(self.actor_critic, "is_recurrent", False):
            raise NotImplementedError("AMPFPO currently supports non-recurrent policies only.")

        aug_obs = obs.detach()
        aug_critic_obs = critic_obs.detach()

        self.transition.actions = self.actor_critic.act(aug_obs).detach()
        self.transition.values_task = self.actor_critic.evaluate_task(aug_critic_obs).detach()
        self.transition.values_repair = self.actor_critic.evaluate_repair(aug_critic_obs).detach()
        if self.use_recover_critic:
            self.transition.values_recover = self.actor_critic.evaluate_recover(aug_critic_obs).detach()
        else:
            self.transition.values_recover = torch.zeros_like(self.transition.values_task)

        self.transition.actions_log_prob = self.actor_critic.get_actions_log_prob(self.transition.actions).detach()
        self.transition.action_mean = self.actor_critic.action_mean.detach()
        self.transition.action_sigma = self.actor_critic.action_std.detach()

        # record obs before env.step
        self.transition.observations = obs
        self.transition.critic_observations = critic_obs

        # record AMP features for discriminator transition
        self.amp_transition.observations = amp_obs

        return self.transition.actions

    def process_env_step(self, task_rewards, dones, infos, next_amp_obs):
        """
        Process environment step and store transition.
        
        ===== KEY DIFFERENCE FROM AMP PPO =====
        FPO uses PURE task reward (no mixing with AMP reward):
        - task_rewards: Raw task reward from environment (can be positive or negative)
        - In AMP PPO: rewards are mixed via discriminator.predict_amp_reward()
        - In FPO: task_rewards are used directly (no mixing)
        
        Task reward composition (from tracking_env_cfg.py):
        POSITIVE terms (when tracking is good):
        - motion_global_anchor_pos: weight=0.5, max ~0.5
        - motion_global_anchor_ori: weight=0.5, max ~0.5
        - motion_body_pos: weight=1.0, max ~1.0
        - motion_body_ori: weight=1.0, max ~1.0
        - motion_body_lin_vel: weight=1.0, max ~1.0
        - motion_body_ang_vel: weight=1.0, max ~1.0
        → Maximum positive: ~4.0 (when tracking is perfect)
        
        NEGATIVE terms (penalties):
        - action_rate_l2: weight=-0.1
        - joint_limit: weight=-10.0 (LARGE penalty if joints exceed limits)
        - undesired_contacts: weight=-0.1
        
        Why task_reward might be negative:
        1. Training early: Policy hasn't learned good tracking → positive terms ≈ 0
        2. Joint limit violations: -10.0 penalty dominates
        3. Poor tracking: Positive terms are small, penalties accumulate
        
        Why task_reward should become positive as training progresses:
        - Policy learns better tracking → positive terms increase (toward ~4.0)
        - Policy avoids violations → penalties decrease
        - Net result: task_reward becomes positive
        
        NOTE: In FPO, task_reward is ONLY used for in-region transitions.
        Out-region transitions use repair_reward (computed from discriminator).
        """
        assert self.storage is not None, "Call init_storage() first."

        # IMPORTANT: Keep reward/done tensors consistently shaped as [N, 1] to avoid silent broadcasting bugs.
        # Some envs return [N] and others [N,1].
        task_rewards = task_rewards.view(-1, 1)
        dones = dones.view(-1, 1)
        self.transition.task_rewards = task_rewards.clone()
        self.transition.dones = dones

        # Add AMP transition to replay buffer for discriminator update.
        self.amp_storage.insert(self.amp_transition.observations, next_amp_obs)

        # Compute feasibility label + repair reward from discriminator probability.
        disc_prob, _disc_raw = self._compute_disc_prob(self.amp_transition.observations, next_amp_obs)
        in_region = disc_prob >= self.feasible_threshold
        hard_violation = disc_prob < self.hard_violation_threshold

        # Repair stream: out-region-only negative shaping
        repair_reward, violation = self._repair_reward_from_prob(disc_prob)
        # Only apply repair shaping on out-set (in-set -> 0)
        repair_r = torch.where(in_region, torch.zeros_like(repair_reward), repair_reward)
        if self.in_region_style_reward_coef > 0:
            style_reward = self.in_region_style_reward_coef * disc_prob
            style_reward = torch.where(in_region, style_reward, torch.zeros_like(style_reward))
            self.transition.task_rewards = self.transition.task_rewards + style_reward.view(-1, 1)

        # Bootstrap on timeouts (task stream)
        if "time_outs" in infos:
            to = infos["time_outs"].view(-1, 1).to(self.device)
            self.transition.task_rewards += self.gamma * (self.transition.values_task * to)

        # Bootstrap on timeouts (repair/cost stream)
        if "time_outs" in infos:
            to = infos["time_outs"].view(-1, 1).to(self.device)
            repair_r = repair_r + self.gamma_repair * (self.transition.values_repair * to).view(-1)
        self.transition.repair_rewards = repair_r.view(-1, 1)
        self.transition.in_region = in_region.view(-1, 1)
        self.transition.hard_violation = hard_violation.view(-1, 1)
        self.transition.disc_prob = disc_prob.view(-1, 1)

        # Recover stream (optional): only meaningful for hard-violation transitions.
        # IMPORTANT: Recover reward should be NEGATIVE (penalty) to encourage recovery from hard violations.
        # Use -(1 - disc_prob) so that hard violations (low disc_prob) get larger penalties.
        # This ensures recover advantage is negative, encouraging policy to improve disc_prob.
        # Similar to repair_reward design: negative reward encourages recovery.
        if self.use_recover_critic:
            # recover_r = self.recover_reward_coef * disc_prob
            recover_r = -self.recover_reward_coef * (1.0 - disc_prob)
            recover_r = torch.where(hard_violation, recover_r, torch.zeros_like(recover_r))
            if "time_outs" in infos:
                to = infos["time_outs"].view(-1, 1).to(self.device)
                recover_r = recover_r + self.gamma_recover * (self.transition.values_recover * to).view(-1)
            self.transition.recover_rewards = recover_r.view(-1, 1)
        else:
            self.transition.recover_rewards = None

        # Record transition
        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.amp_transition.clear()

        self.actor_critic.reset(dones)

        # (for logging; return values in case runner wants them)
        return disc_prob.mean().item(), violation.mean().item(), in_region.float().mean().item()

    def compute_returns(self, last_critic_obs):
        assert self.storage is not None, "Call init_storage() first."
        aug_last = last_critic_obs.detach()
        last_task = self.actor_critic.evaluate_task(aug_last).detach()
        last_repair = self.actor_critic.evaluate_repair(aug_last).detach()
        last_recover = self.actor_critic.evaluate_recover(aug_last).detach() if self.use_recover_critic else None
        self.storage.compute_returns(
            last_task,
            last_repair,
            self.gamma,
            self.lam,
            self.gamma_repair,
            self.lam_repair,
            last_values_recover=last_recover,
            gamma_recover=self.gamma_recover,
            lam_recover=self.lam_recover,
        )

    # ---------------------------
    # Update
    # ---------------------------
    def update(self):
        assert self.storage is not None, "Call init_storage() first."

        mean_value_loss_task = 0.0
        mean_value_loss_repair = 0.0
        mean_surrogate_loss = 0.0
        mean_amp_loss = 0.0
        mean_grad_pen_loss = 0.0
        mean_policy_pred = 0.0
        mean_expert_pred = 0.0
        mean_in_ratio = 0.0
        mean_task_adv = 0.0
        std_task_adv = 0.0

        # Decide discriminator update (reduce coupling between feasibility boundary and actor updates).
        do_disc_update = (self._update_iter >= self.disc_update_start_iter) and (
            (self._update_iter - self.disc_update_start_iter) % self.disc_update_every == 0
        )

        # Rollout-level in/out statistics to stabilize critic training in sparse regions.
        rollout_in_ratio = self.storage.in_region.float().mean().item()

        generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)

        if do_disc_update:
            amp_policy_generator = self.amp_storage.feed_forward_generator(
                self.num_learning_epochs * self.num_mini_batches,
                self.storage.num_envs * self.storage.num_transitions_per_env // self.num_mini_batches,
            )
            amp_expert_generator = self.amp_data.feed_forward_generator(
                self.num_learning_epochs * self.num_mini_batches,
                self.storage.num_envs * self.storage.num_transitions_per_env // self.num_mini_batches,
            )
            loop_iter = zip(generator, amp_policy_generator, amp_expert_generator)
        else:
            loop_iter = ((sample, None, None) for sample in generator)

        # If we skip discriminator updates, prevent weight_decay-only updates from a shared Adam optimizer
        # by setting discriminator param-group lr to 0 for this update call.
        saved_lrs = None
        if not do_disc_update:
            saved_lrs = [pg.get("lr", None) for pg in self.optimizer.param_groups]
            for pg in self.optimizer.param_groups:
                if pg.get("name") in ("amp_trunk", "amp_head"):
                    pg["lr"] = 0.0

        for sample, sample_amp_policy, sample_amp_expert in loop_iter:
            (
                obs_batch,
                critic_obs_batch,
                actions_batch,
                old_values_task_batch,
                old_values_repair_batch,
                old_values_recover_batch,
                returns_task_batch,
                returns_repair_batch,
                returns_recover_batch,
                advantages_batch,
                advantages_task_batch,
                advantages_repair_batch,
                advantages_recover_batch,
                in_region_batch,
                hard_violation_batch,
                old_logp_batch,
                old_mu_batch,
                old_sigma_batch,
                _hid_states_batch,
                _masks_batch,
            ) = sample

            aug_obs_batch = obs_batch.detach()
            self.actor_critic.act(aug_obs_batch)
            actions_log_prob_batch = self.actor_critic.get_actions_log_prob(actions_batch)

            aug_critic_obs_batch = critic_obs_batch.detach()
            value_task_batch = self.actor_critic.evaluate_task(aug_critic_obs_batch)
            value_repair_batch = self.actor_critic.evaluate_repair(aug_critic_obs_batch)
            value_recover_batch = self.actor_critic.evaluate_recover(aug_critic_obs_batch) if self.use_recover_critic else None

            mu_batch = self.actor_critic.action_mean
            sigma_batch = self.actor_critic.action_std
            entropy_batch = self.actor_critic.entropy

            # KL scheduling (same as PPO)
            if self.desired_kl is not None and self.schedule == "adaptive":
                with torch.inference_mode():
                    kl = torch.sum(
                        torch.log(sigma_batch / old_sigma_batch + 1e-5)
                        + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch))
                        / (2.0 * torch.square(sigma_batch))
                        - 0.5,
                        axis=-1,
                    )
                    kl_mean = torch.mean(kl)
                    if kl_mean > self.desired_kl * 2.0:
                        self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                    elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                        self.learning_rate = min(1e-2, self.learning_rate * 1.5)
                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.learning_rate

            # PPO surrogate (shared actor)
            # Switch mode: use precomputed (and already normalized) region-wise advantages from storage.
            # NOTE: hard_violation override (if enabled) is applied inside storage.compute_returns() BEFORE normalization.
            adv_for_policy = torch.squeeze(advantages_batch)

            ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_logp_batch))
            surrogate = -adv_for_policy * ratio
            surrogate_clipped = -adv_for_policy * torch.clamp(
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
            )
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            # Value losses
            in_mask = in_region_batch.bool().view(-1, 1)
            out_mask = ~in_mask
            hv_mask = hard_violation_batch.bool().view(-1, 1)

            if self.use_clipped_value_loss:
                vt_clipped = old_values_task_batch + (value_task_batch - old_values_task_batch).clamp(
                    -self.clip_param, self.clip_param
                )
                vt_losses = (value_task_batch - returns_task_batch).pow(2)
                vt_losses_clipped = (vt_clipped - returns_task_batch).pow(2)
                value_loss_task_all = torch.max(vt_losses, vt_losses_clipped)

                vr_clipped = old_values_repair_batch + (value_repair_batch - old_values_repair_batch).clamp(
                    -self.clip_param, self.clip_param
                )
                vr_losses = (value_repair_batch - returns_repair_batch).pow(2)
                vr_losses_clipped = (vr_clipped - returns_repair_batch).pow(2)
                value_loss_repair_all = torch.max(vr_losses, vr_losses_clipped)
            else:
                value_loss_task_all = (returns_task_batch - value_task_batch).pow(2)
                value_loss_repair_all = (returns_repair_batch - value_repair_batch).pow(2)

            # Stabilize critic learning when one region is too sparse (early training / threshold mismatch).
            in_frac = in_mask.float().mean().item()
            out_frac = 1.0 - in_frac

            # Task critic recovery: when in_region_ratio rises above threshold, force full-sample training
            if rollout_in_ratio >= self.task_critic_recovery_threshold and not self._task_critic_recovery_active:
                self._task_critic_recovery_active = True
                self._task_critic_recovery_start_iter = self._update_iter
                print(f"[TASK-CRITIC-RECOVERY] Activated at iter {self._update_iter}, "
                      f"rollout_in_ratio={rollout_in_ratio:.3f} >= {self.task_critic_recovery_threshold:.3f}")
            
            # Check if recovery period has ended
            if self._task_critic_recovery_active:
                recovery_iters_elapsed = self._update_iter - self._task_critic_recovery_start_iter
                if recovery_iters_elapsed >= self.task_critic_recovery_iters:
                    self._task_critic_recovery_active = False
                    print(f"[TASK-CRITIC-RECOVERY] Completed at iter {self._update_iter} "
                          f"(ran for {recovery_iters_elapsed} iterations)")

            mix_region_losses = (
                (self._update_iter < self.value_mix_warmup_iters)
                or (rollout_in_ratio < self.value_mix_in_ratio_low)
                or (rollout_in_ratio > self.value_mix_in_ratio_high)
                or self._task_critic_recovery_active  # Force mixing during recovery
            )

            if in_frac < self.min_region_fraction:
                value_loss_task = value_loss_task_all.mean()
            else:
                # During recovery, use full-sample training (weight=1.0 means all samples)
                if self._task_critic_recovery_active:
                    # Use all samples equally (full-sample training)
                    value_loss_task = value_loss_task_all.mean()
                elif mix_region_losses and out_frac >= self.min_region_fraction:
                    w = self.value_mix_task_out_weight
                    value_loss_task = (1.0 - w) * masked_mean(value_loss_task_all, in_mask) + w * masked_mean(
                        value_loss_task_all, out_mask
                    )
                else:
                    value_loss_task = masked_mean(value_loss_task_all, in_mask)

            # Repair critic training: train mainly on out-region with optional mixing for stability.
            if out_frac < self.min_region_fraction:
                value_loss_repair = value_loss_repair_all.mean()
            else:
                if mix_region_losses and in_frac >= self.min_region_fraction:
                    w = self.value_mix_repair_in_weight
                    value_loss_repair = (1.0 - w) * masked_mean(value_loss_repair_all, out_mask) + w * masked_mean(
                        value_loss_repair_all, in_mask
                    )
                else:
                    value_loss_repair = masked_mean(value_loss_repair_all, out_mask)

            # Recover critic update (separate optimizer; only on hard-violation samples)
            if self.use_recover_critic and self.recover_optimizer is not None and value_recover_batch is not None:
                if self.use_clipped_value_loss:
                    vrec_clipped = old_values_recover_batch + (value_recover_batch - old_values_recover_batch).clamp(
                        -self.clip_param, self.clip_param
                    )
                    vrec_losses = (value_recover_batch - returns_recover_batch).pow(2)
                    vrec_losses_clipped = (vrec_clipped - returns_recover_batch).pow(2)
                    value_loss_recover_all = torch.max(vrec_losses, vrec_losses_clipped)
                else:
                    value_loss_recover_all = (returns_recover_batch - value_recover_batch).pow(2)

                # If no hard-violation samples in this batch, skip update.
                if hv_mask.float().sum().item() >= 1.0:
                    value_loss_recover = masked_mean(value_loss_recover_all, hv_mask)
                    self.recover_optimizer.zero_grad()
                    (self.recover_value_loss_coef * value_loss_recover).backward()
                    nn.utils.clip_grad_norm_(self.actor_critic.recover_critic.parameters(), self.max_grad_norm)
                    self.recover_optimizer.step()
                    # IMPORTANT: clear recover_critic grads so they don't pollute the main PPO grad-norm clipping.
                    for p in self.actor_critic.recover_critic.parameters():
                        p.grad = None

            # Discriminator update loss (same as AMPPPO), optionally skipped for stability.
            if do_disc_update:
                policy_state, policy_next_state = sample_amp_policy
                expert_state, expert_next_state = sample_amp_expert
                if self.amp_normalizer is not None:
                    with torch.no_grad():
                        policy_state = self.amp_normalizer.normalize_torch(policy_state, self.device)
                        policy_next_state = self.amp_normalizer.normalize_torch(policy_next_state, self.device)
                        expert_state = self.amp_normalizer.normalize_torch(expert_state, self.device)
                        expert_next_state = self.amp_normalizer.normalize_torch(expert_next_state, self.device)

                policy_d = self.discriminator(torch.cat([policy_state, policy_next_state], dim=-1))
                expert_d = self.discriminator(torch.cat([expert_state, expert_next_state], dim=-1))
                expert_loss = torch.nn.MSELoss()(expert_d, torch.ones(expert_d.size(), device=self.device))
                policy_loss = torch.nn.MSELoss()(policy_d, -1 * torch.ones(policy_d.size(), device=self.device))
                amp_loss = 0.5 * (expert_loss + policy_loss)
                grad_pen_loss = self.discriminator.compute_grad_pen(*sample_amp_expert, lambda_=10)
            else:
                amp_loss = torch.zeros((), device=self.device)
                grad_pen_loss = torch.zeros((), device=self.device)
                policy_d = torch.zeros((), device=self.device)
                expert_d = torch.zeros((), device=self.device)

            loss = (
                surrogate_loss
                + self.value_loss_coef * value_loss_task
                + self.repair_value_loss_coef * value_loss_repair
                - self.entropy_coef * entropy_batch.mean()
                + amp_loss
                + grad_pen_loss
            )

            self.optimizer.zero_grad()
            loss.backward()
            # IMPORTANT: only clip params actually updated by the main optimizer
            # (exclude recover_critic to avoid scaling down PPO gradients).
            nn.utils.clip_grad_norm_(self._actor_critic_main_params, self.max_grad_norm)
            self.optimizer.step()

            # clamp std if desired
            if not getattr(self.actor_critic, "fixed_std", False) and self.min_std is not None:
                self.actor_critic.std.data = self.actor_critic.std.data.clamp(min=self.min_std)

            # update normalizer stats (only when we actually sampled for discriminator update)
            if self.amp_normalizer is not None and do_disc_update:
                self.amp_normalizer.update(policy_state.cpu().numpy())
                self.amp_normalizer.update(expert_state.cpu().numpy())

            mean_value_loss_task += value_loss_task.item()
            mean_value_loss_repair += value_loss_repair.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_amp_loss += amp_loss.item()
            mean_grad_pen_loss += grad_pen_loss.item()
            mean_policy_pred += policy_d.mean().item()
            mean_expert_pred += expert_d.mean().item()
            mean_in_ratio += in_mask.float().mean().item()
            
            # Diagnostic: task advantage statistics (for debugging episode_length issue)
            with torch.no_grad():
                adv_flat = advantages_batch.view(-1)
                in_mask_flat = in_mask.view(-1)
                if in_mask_flat.sum() > 0:
                    task_adv_batch = adv_flat[in_mask_flat]
                    mean_task_adv += task_adv_batch.mean().item()
                    std_task_adv += task_adv_batch.std().item()
                else:
                    # Fallback: use all advantages if no in-region samples
                    mean_task_adv += adv_flat.mean().item()
                    std_task_adv += adv_flat.std().item()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss_task /= num_updates
        mean_value_loss_repair /= num_updates
        mean_surrogate_loss /= num_updates
        mean_amp_loss /= num_updates
        mean_grad_pen_loss /= num_updates
        mean_policy_pred /= num_updates
        mean_expert_pred /= num_updates
        mean_in_ratio /= num_updates
        mean_task_adv /= num_updates
        std_task_adv /= num_updates

        self.storage.clear()

        # Restore lrs if we temporarily froze discriminator param groups.
        if saved_lrs is not None:
            for pg, lr in zip(self.optimizer.param_groups, saved_lrs):
                if lr is not None:
                    pg["lr"] = lr

        self._update_iter += 1

        return (
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
        )


