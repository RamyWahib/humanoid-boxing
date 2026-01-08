import torch


def masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean over elements where mask==True. Returns 0 if mask is empty."""
    mask_f = mask.float()
    denom = torch.clamp(mask_f.sum(), min=1.0)
    return (x * mask_f).sum() / denom


class FPORolloutStorage:
    """Rollout storage for region-wise (FPO-style) objectives.

    Stores:
      - task reward/value/returns/advantages
      - repair reward/value/returns/advantages (style feasibility recovery)
      - per-transition in-region mask derived from discriminator constraint
    """

    class Transition:
        def __init__(self):
            self.observations = None
            self.critic_observations = None
            self.actions = None

            self.task_rewards = None
            self.repair_rewards = None
            self.recover_rewards = None
            self.dones = None

            self.values_task = None
            self.values_repair = None
            self.values_recover = None

            self.actions_log_prob = None
            self.action_mean = None
            self.action_sigma = None

            self.in_region = None  # bool
            self.hard_violation = None  # bool (subset of out-region)
            self.disc_prob = None  # float in [0,1]

            self.hidden_states = None

        def clear(self):
            self.__init__()

    def __init__(
        self,
        num_envs: int,
        num_transitions_per_env: int,
        obs_shape,
        privileged_obs_shape,
        actions_shape,
        device="cpu",
    ):
        self.device = device
        self.obs_shape = obs_shape
        self.privileged_obs_shape = privileged_obs_shape
        self.actions_shape = actions_shape

        # Core
        self.observations = torch.zeros(num_transitions_per_env, num_envs, *obs_shape, device=self.device)
        if privileged_obs_shape[0] is not None:
            self.privileged_observations = torch.zeros(
                num_transitions_per_env, num_envs, *privileged_obs_shape, device=self.device
            )
        else:
            self.privileged_observations = None
        self.actions = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=self.device)
        self.dones = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device).byte()

        # Region label + discriminator score
        self.in_region = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device, dtype=torch.bool)
        self.hard_violation = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device, dtype=torch.bool)
        self.disc_prob = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)

        # Task stream
        self.task_rewards = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.values_task = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.returns_task = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.advantages_task = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)

        # Repair stream
        self.repair_rewards = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.values_repair = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.returns_repair = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.advantages_repair = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)

        # Recover stream (optional; used when hard_violation is True)
        self.recover_rewards = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.values_recover = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.returns_recover = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.advantages_recover = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)

        # Combined advantage for the shared actor update
        self.advantages = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)

        # PPO bookkeeping
        self.actions_log_prob = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.mu = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=self.device)
        self.sigma = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=self.device)

        self.num_transitions_per_env = num_transitions_per_env
        self.num_envs = num_envs
        self.step = 0

        # (kept for API parity; we don't support recurrent here)
        self.saved_hidden_states_a = None
        self.saved_hidden_states_c = None

    def clear(self):
        self.step = 0

    def add_transitions(self, transition: Transition):
        if self.step >= self.num_transitions_per_env:
            raise AssertionError("Rollout buffer overflow")

        self.observations[self.step].copy_(transition.observations)
        if self.privileged_observations is not None:
            self.privileged_observations[self.step].copy_(transition.critic_observations)
        self.actions[self.step].copy_(transition.actions)
        self.dones[self.step].copy_(transition.dones.view(-1, 1))

        self.task_rewards[self.step].copy_(transition.task_rewards.view(-1, 1))
        self.repair_rewards[self.step].copy_(transition.repair_rewards.view(-1, 1))
        self.values_task[self.step].copy_(transition.values_task)
        self.values_repair[self.step].copy_(transition.values_repair)

        self.in_region[self.step].copy_(transition.in_region.view(-1, 1))
        if transition.hard_violation is not None:
            self.hard_violation[self.step].copy_(transition.hard_violation.view(-1, 1))
        else:
            self.hard_violation[self.step].zero_()
        self.disc_prob[self.step].copy_(transition.disc_prob.view(-1, 1))

        if transition.recover_rewards is not None:
            self.recover_rewards[self.step].copy_(transition.recover_rewards.view(-1, 1))
        else:
            self.recover_rewards[self.step].zero_()
        if transition.values_recover is not None:
            self.values_recover[self.step].copy_(transition.values_recover)
        else:
            self.values_recover[self.step].zero_()

        self.actions_log_prob[self.step].copy_(transition.actions_log_prob.view(-1, 1))
        self.mu[self.step].copy_(transition.action_mean)
        self.sigma[self.step].copy_(transition.action_sigma)

        self.step += 1

    def compute_returns(
        self,
        last_values_task: torch.Tensor,
        last_values_repair: torch.Tensor,
        gamma_task: float,
        lam_task: float,
        gamma_repair: float,
        lam_repair: float,
        last_values_recover: torch.Tensor | None = None,
        gamma_recover: float | None = None,
        lam_recover: float | None = None,
    ):
        # --- task GAE ---
        adv = 0.0
        for step in reversed(range(self.num_transitions_per_env)):
            next_values = last_values_task if step == self.num_transitions_per_env - 1 else self.values_task[step + 1]
            next_not_terminal = 1.0 - self.dones[step].float()
            delta = self.task_rewards[step] + next_not_terminal * gamma_task * next_values - self.values_task[step]
            adv = delta + next_not_terminal * gamma_task * lam_task * adv
            self.returns_task[step] = adv + self.values_task[step]
        self.advantages_task = self.returns_task - self.values_task

        # --- repair GAE ---
        adv = 0.0
        for step in reversed(range(self.num_transitions_per_env)):
            next_values = (
                last_values_repair if step == self.num_transitions_per_env - 1 else self.values_repair[step + 1]
            )
            next_not_terminal = 1.0 - self.dones[step].float()
            delta = (
                self.repair_rewards[step] + next_not_terminal * gamma_repair * next_values - self.values_repair[step]
            )
            adv = delta + next_not_terminal * gamma_repair * lam_repair * adv
            self.returns_repair[step] = adv + self.values_repair[step]
        self.advantages_repair = self.returns_repair - self.values_repair

        # --- recover GAE (optional) ---
        if last_values_recover is not None:
            g_rec = gamma_recover if gamma_recover is not None else gamma_repair
            l_rec = lam_recover if lam_recover is not None else lam_repair
            adv = 0.0
            for step in reversed(range(self.num_transitions_per_env)):
                next_values = (
                    last_values_recover if step == self.num_transitions_per_env - 1 else self.values_recover[step + 1]
                )
                next_not_terminal = 1.0 - self.dones[step].float()
                delta = self.recover_rewards[step] + next_not_terminal * g_rec * next_values - self.values_recover[step]
                adv = delta + next_not_terminal * g_rec * l_rec * adv
                self.returns_recover[step] = adv + self.values_recover[step]
            self.advantages_recover = self.returns_recover - self.values_recover
        else:
            self.returns_recover.zero_()
            self.advantages_recover.zero_()

        # --- region-wise combined advantage (shared actor) ---
        self.advantages = torch.where(self.in_region, self.advantages_task, self.advantages_repair)
        # If recover stream is enabled, hard-violation samples override the combined advantage.
        # This keeps normalization consistent (override happens BEFORE normalize).
        if last_values_recover is not None:
            self.advantages = torch.where(self.hard_violation, self.advantages_recover, self.advantages)

        # normalize combined advantages (avoid manual weighting issues)
        adv_flat = self.advantages.view(-1)
        self.advantages = (self.advantages - adv_flat.mean()) / (adv_flat.std() + 1e-8)

    def get_region_statistics(self) -> dict[str, float]:
        in_ratio = self.in_region.float().mean().item()
        disc_mean = self.disc_prob.mean().item()
        return {"in_region_ratio": in_ratio, "disc_prob_mean": disc_mean}

    def mini_batch_generator(self, num_mini_batches: int, num_epochs: int = 8):
        batch_size = self.num_envs * self.num_transitions_per_env
        mini_batch_size = batch_size // num_mini_batches
        indices = torch.randperm(num_mini_batches * mini_batch_size, requires_grad=False, device=self.device)

        observations = self.observations.flatten(0, 1)
        critic_observations = (
            self.privileged_observations.flatten(0, 1) if self.privileged_observations is not None else observations
        )
        actions = self.actions.flatten(0, 1)
        old_logp = self.actions_log_prob.flatten(0, 1)
        adv = self.advantages.flatten(0, 1)
        in_region = self.in_region.flatten(0, 1)
        hard_violation = self.hard_violation.flatten(0, 1)

        # task stream
        old_values_task = self.values_task.flatten(0, 1)
        returns_task = self.returns_task.flatten(0, 1)
        adv_task = self.advantages_task.flatten(0, 1)

        # repair stream
        old_values_repair = self.values_repair.flatten(0, 1)
        returns_repair = self.returns_repair.flatten(0, 1)
        adv_repair = self.advantages_repair.flatten(0, 1)

        # recover stream
        old_values_recover = self.values_recover.flatten(0, 1)
        returns_recover = self.returns_recover.flatten(0, 1)
        adv_recover = self.advantages_recover.flatten(0, 1)

        old_mu = self.mu.flatten(0, 1)
        old_sigma = self.sigma.flatten(0, 1)

        for _ in range(num_epochs):
            for i in range(num_mini_batches):
                start = i * mini_batch_size
                end = (i + 1) * mini_batch_size
                batch_idx = indices[start:end]

                yield (
                    observations[batch_idx],
                    critic_observations[batch_idx],
                    actions[batch_idx],
                    old_values_task[batch_idx],
                    old_values_repair[batch_idx],
                    old_values_recover[batch_idx],
                    returns_task[batch_idx],
                    returns_repair[batch_idx],
                    returns_recover[batch_idx],
                    adv[batch_idx],
                    adv_task[batch_idx],
                    adv_repair[batch_idx],
                    adv_recover[batch_idx],
                    in_region[batch_idx],
                    hard_violation[batch_idx],
                    old_logp[batch_idx],
                    old_mu[batch_idx],
                    old_sigma[batch_idx],
                    (None, None),
                    None,
                )


