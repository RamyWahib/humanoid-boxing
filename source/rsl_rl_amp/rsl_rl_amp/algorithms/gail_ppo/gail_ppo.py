import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from rsl_rl_amp.modules import ActorCritic
from rsl_rl_amp.storage import RolloutStorage
from rsl_rl_amp.storage.replay_buffer import ReplayBuffer
from beyondAMP.motion.motion_dataset import MotionDataset
from rsl_rl_amp.modules.amp_discriminator import AMPDiscriminator  # Reusable AMP-style discriminator

class GAILPPO:
    """
    GAIL + PPO baseline designed for AMP-style setups where the discriminator
    takes (state, next_state) as input.

    Fully compatible with the AMPPPO interface, including:
        - init_storage
        - act
        - process_env_step
        - compute_returns
        - update

    Important hyperparameters:
        - lambda_gail: scaling factor for the GAIL reward (typically 1.0 or smaller)
        - disc_steps: number of discriminator updates per minibatch
        - grad_pen_lambda: weight for gradient penalty if the discriminator
                           provides compute_grad_pen()
    """
    def __init__(self,
                 actor_critic: ActorCritic,
                 discriminator: AMPDiscriminator,
                 amp_data: MotionDataset,
                 amp_normalizer=None,
                 lambda_gail=1.0,
                 disc_steps=1,
                 grad_pen_lambda=10.0,
                 policy_lr=1e-3,
                 disc_lr=1e-3,
                 num_learning_epochs=1,
                 num_mini_batches=1,
                 clip_param=0.2,
                 gamma=0.998,
                 lam=0.95,
                 value_loss_coef=1.0,
                 entropy_coef=0.0,
                 max_grad_norm=1.0,
                 use_clipped_value_loss=True,
                 schedule="fixed",
                 desired_kl=0.01,
                 device='cpu',
                 amp_replay_buffer_size=100000,
                 min_std=None,
                 **kwargs):
        self.device = device

        # Core models
        self.actor_critic = actor_critic.to(self.device)
        self.discriminator = discriminator.to(self.device)

        # AMP dataset and optional normalization
        self.amp_data = amp_data
        self.amp_normalizer = amp_normalizer
        # Stores policy-generated (state, next_state) transitions
        self.amp_storage = ReplayBuffer(discriminator.input_dim // 2,
                                        amp_replay_buffer_size,
                                        device)

        # GAIL hyperparameters
        self.lambda_gail = lambda_gail
        self.disc_steps = disc_steps
        self.grad_pen_lambda = grad_pen_lambda
        self.min_std = min_std

        # PPO settings
        self.clip_param = clip_param
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.lam = lam
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss
        self.schedule = schedule
        self.desired_kl = desired_kl

        # Rollout storage
        self.storage = None
        self.transition = RolloutStorage.Transition()
        self.amp_transition = RolloutStorage.Transition()

        # Optimizers: policy and discriminator managed separately
        self.policy_optimizer = optim.Adam(self.actor_critic.parameters(), lr=policy_lr)
        disc_params = (list(self.discriminator.trunk.parameters()) +
                       list(self.discriminator.amp_linear.parameters()))
        self.disc_optimizer = optim.Adam(disc_params, lr=disc_lr)

        self.bce_loss = nn.BCEWithLogitsLoss()

    def init_storage(self, num_envs, num_transitions_per_env, actor_obs_shape,
                     critic_obs_shape, action_shape):
        self.storage = RolloutStorage(
            num_envs, num_transitions_per_env,
            actor_obs_shape, critic_obs_shape,
            action_shape, self.device)

    def test_mode(self):
        self.actor_critic.test()

    def train_mode(self):
        self.actor_critic.train()

    def act(self, obs, critic_obs, amp_obs):
        # Save hidden states if RNN is used
        if self.actor_critic.is_recurrent:
            self.transition.hidden_states = self.actor_critic.get_hidden_states()

        # Standard policy forward pass
        aug_obs = obs.detach()
        aug_critic_obs = critic_obs.detach()

        self.transition.actions = self.actor_critic.act(aug_obs).detach()
        self.transition.values = self.actor_critic.evaluate(aug_critic_obs).detach()
        self.transition.actions_log_prob = self.actor_critic.get_actions_log_prob(
            self.transition.actions).detach()
        self.transition.action_mean = self.actor_critic.action_mean.detach()
        self.transition.action_sigma = self.actor_critic.action_std.detach()
        self.transition.observations = obs
        self.transition.critic_observations = critic_obs

        # AMP transition: store current state
        self.amp_transition.observations = amp_obs
        return self.transition.actions

    def process_env_step(self, rewards, dones, infos, amp_obs):
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones

        # Handle time_out bootstrap if provided
        if 'time_outs' in infos:
            self.transition.rewards += (
                self.gamma *
                torch.squeeze(self.transition.values *
                              infos['time_outs'].unsqueeze(1).to(self.device), 1)
            )

        # Insert (state, next_state) for discriminator training
        self.amp_storage.insert(self.amp_transition.observations, amp_obs)

        # Standard PPO rollout storage
        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.amp_transition.clear()

        self.actor_critic.reset(dones)

    def compute_returns(self, last_critic_obs):
        last_values = self.actor_critic.evaluate(last_critic_obs.detach()).detach()
        self.storage.compute_returns(last_values, self.gamma, self.lam)

    def _gail_reward(self, logits):
        # Numerical-stable form: -log(1 - sigmoid(x)) = -logsigmoid(-x)
        return -F.logsigmoid(-logits)

    def update(self):
        """
        Main update loop.
        Iterates over PPO minibatches and aligns:
            - PPO rollout samples
            - policy-generated (state, next_state)
            - expert (state, next_state)

        The discriminator is updated disc_steps times per minibatch.

        The GAIL reward is added to returns and advantages with scaling λ.
        """
        if self.actor_critic.is_recurrent:
            generator = self.storage.reccurent_mini_batch_generator(
                self.num_mini_batches, self.num_learning_epochs)
        else:
            generator = self.storage.mini_batch_generator(
                self.num_mini_batches, self.num_learning_epochs)

        amp_policy_generator = self.amp_storage.feed_forward_generator(
            self.num_learning_epochs * self.num_mini_batches,
            self.storage.num_envs *
            self.storage.num_transitions_per_env //
            self.num_mini_batches)

        amp_expert_generator = self.amp_data.feed_forward_generator(
            self.num_learning_epochs * self.num_mini_batches,
            self.storage.num_envs *
            self.storage.num_transitions_per_env //
            self.num_mini_batches)

        mean_val_loss = 0.0
        mean_surr_loss = 0.0
        mean_disc_loss = 0.0
        mean_policy_pred = 0.0
        mean_expert_pred = 0.0
        updates = 0

        for sample, sample_amp_policy, sample_amp_expert in zip(
                generator, amp_policy_generator, amp_expert_generator):
            (obs_batch, critic_obs_batch, actions_batch, target_values_batch,
             advantages_batch, returns_batch, old_actions_log_prob_batch,
             old_mu_batch, old_sigma_batch, hid_states_batch,
             masks_batch) = sample

            # ------------------------------------------------------------
            # Discriminator update
            # ------------------------------------------------------------
            policy_state, policy_next_state = sample_amp_policy
            expert_state, expert_next_state = sample_amp_expert

            if self.amp_normalizer is not None:
                with torch.no_grad():
                    policy_state = self.amp_normalizer.normalize_torch(policy_state, self.device)
                    policy_next_state = self.amp_normalizer.normalize_torch(policy_next_state, self.device)
                    expert_state = self.amp_normalizer.normalize_torch(expert_state, self.device)
                    expert_next_state = self.amp_normalizer.normalize_torch(expert_next_state, self.device)

            for _ in range(self.disc_steps):
                policy_logits = self.discriminator(
                    torch.cat([policy_state, policy_next_state], dim=-1))
                expert_logits = self.discriminator(
                    torch.cat([expert_state, expert_next_state], dim=-1))

                expert_labels = torch.ones_like(expert_logits, device=self.device)
                policy_labels = torch.zeros_like(policy_logits, device=self.device)

                expert_loss = self.bce_loss(expert_logits, expert_labels)
                policy_loss = self.bce_loss(policy_logits, policy_labels)
                d_loss = 0.5 * (expert_loss + policy_loss)

                # Gradient penalty if provided
                if hasattr(self.discriminator, 'compute_grad_pen'):
                    gp = self.discriminator.compute_grad_pen(
                        *sample_amp_expert, lambda_=self.grad_pen_lambda)
                    d_loss = d_loss + gp

                self.disc_optimizer.zero_grad()
                d_loss.backward()
                self.disc_optimizer.step()

            # Discriminator statistics
            mean_disc_loss += d_loss.item()
            mean_policy_pred += torch.sigmoid(policy_logits).mean().item()
            mean_expert_pred += torch.sigmoid(expert_logits).mean().item()

            # ------------------------------------------------------------
            # PPO forward pass
            # ------------------------------------------------------------
            aug_obs_batch = obs_batch.detach()
            self.actor_critic.act(
                aug_obs_batch, masks=masks_batch, hidden_states=hid_states_batch[0])
            actions_log_prob_batch = self.actor_critic.get_actions_log_prob(actions_batch)
            value_batch = self.actor_critic.evaluate(
                critic_obs_batch.detach(), masks=masks_batch,
                hidden_states=hid_states_batch[1])

            mu_batch = self.actor_critic.action_mean
            sigma_batch = self.actor_critic.action_std
            entropy_batch = self.actor_critic.entropy

            # Adaptive KL schedule (optional)
            if self.desired_kl is not None and self.schedule == 'adaptive':
                with torch.inference_mode():
                    kl = torch.sum(
                        torch.log(sigma_batch / old_sigma_batch + 1.e-5) +
                        (torch.square(old_sigma_batch) +
                         torch.square(old_mu_batch - mu_batch)) /
                        (2.0 * torch.square(sigma_batch)) - 0.5,
                        axis=-1)
                    kl_mean = torch.mean(kl)
                    if kl_mean > self.desired_kl * 2.0:
                        for pg in self.policy_optimizer.param_groups:
                            pg['lr'] = max(1e-5, pg['lr'] / 1.5)
                    elif self.desired_kl / 2.0 < kl_mean:
                        for pg in self.policy_optimizer.param_groups:
                            pg['lr'] = min(1e-2, pg['lr'] * 1.5)

            # ------------------------------------------------------------
            # GAIL reward: discriminator-based shaping
            # ------------------------------------------------------------
            with torch.no_grad():
                policy_logits_for_reward = self.discriminator(
                    torch.cat([policy_state, policy_next_state], dim=-1))
                gail_r = self._safe_gail_reward(policy_logits_for_reward).squeeze()

                if returns_batch.dim() > gail_r.dim():
                    gail_r = gail_r.unsqueeze(1)

                returns_batch = returns_batch + self.lambda_gail * gail_r
                advantages_batch = advantages_batch + self.lambda_gail * gail_r

            # ------------------------------------------------------------
            # PPO losses
            # ------------------------------------------------------------
            ratio = torch.exp(actions_log_prob_batch -
                              torch.squeeze(old_actions_log_prob_batch))

            surrogate = -torch.squeeze(advantages_batch) * ratio
            surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param)
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            if self.use_clipped_value_loss:
                value_clipped = target_values_batch + (
                    value_batch - target_values_batch
                ).clamp(-self.clip_param, self.clip_param)
                value_losses = (value_batch - returns_batch).pow(2)
                value_losses_clipped = (value_clipped - returns_batch).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (returns_batch - value_batch).pow(2).mean()

            total_policy_loss = (
                surrogate_loss
                + self.value_loss_coef * value_loss
                - self.entropy_coef * entropy_batch.mean()
            )

            # Policy update
            self.policy_optimizer.zero_grad()
            total_policy_loss.backward()
            nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
            self.policy_optimizer.step()

            # Optional std clamping
            if not self.actor_critic.fixed_std and self.min_std is not None:
                self.actor_critic.std.data.clamp_(min=self.min_std)

            mean_val_loss += value_loss.item()
            mean_surr_loss += surrogate_loss.item()
            updates += 1

        # Normalize metrics
        if updates == 0:
            return 0, 0, 0, 0, 0

        mean_val_loss /= updates
        mean_surr_loss /= updates
        denom = max(1, self.num_learning_epochs * self.num_mini_batches)
        mean_disc_loss /= denom
        mean_policy_pred /= denom
        mean_expert_pred /= denom

        # Reset PPO storage
        self.storage.clear()
        return mean_val_loss, mean_surr_loss, mean_disc_loss, mean_policy_pred, mean_expert_pred

    def _safe_gail_reward(self, logits):
        # Numerically stable -log(1 - sigmoid(x))
        return -F.logsigmoid(-logits)
