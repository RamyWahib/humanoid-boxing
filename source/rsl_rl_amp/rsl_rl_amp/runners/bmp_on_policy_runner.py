import time
import os
from collections import deque
import statistics

import numpy as np
from torch.utils.tensorboard import SummaryWriter
import torch

from rsl_rl_amp.algorithms.bmp_ppo import BMPPPO
from rsl_rl_amp.modules import ActorCritic, ActorCriticRecurrent, ActorCriticMulti
from rsl_rl_amp.env import VecEnv
from rsl_rl_amp.modules.amp_discriminator import AMPDiscriminator
from rsl_rl_amp.utils.utils import Normalizer
from beyondAMP.isaaclab.rsl_rl.amp_wrapper import AMPEnvWrapper

from beyondAMP.motion.motion_dataset import MotionDataset

from .amp_on_policy_runner import AMPOnPolicyRunner

class BMPOnPolicyRunner(AMPOnPolicyRunner):

    def __init__(self,
                 env: VecEnv,
                 train_cfg,
                 log_dir=None,
                 device='cpu'):

        self.cfg=train_cfg
        self.alg_cfg = train_cfg["algorithm"]
        self.policy_cfg = train_cfg["policy"]
        self.amp_data_cfg = train_cfg["amp_data"]
        self.device = device
        self.env:AMPEnvWrapper = env
        if self.env.num_privileged_obs is not None:
            num_critic_obs = self.env.num_privileged_obs 
        else:
            num_critic_obs = self.env.num_obs
        actor_critic_class = eval(self.policy_cfg["class_name"]) # ActorCritic
        num_actor_obs = self.env.num_obs
        actor_critic: ActorCritic = actor_critic_class( num_actor_obs=num_actor_obs,
                                                        num_critic_obs=num_critic_obs,
                                                        num_actions=self.env.num_actions,
                                                        **self.policy_cfg).to(self.device)

        amp_data = env.motion_dataset
        amp_obs_dim = env.get_amp_observations().shape[-1] # amp_data.observation_dim
        amp_normalizer = Normalizer(amp_obs_dim)
        discriminator = AMPDiscriminator(
            amp_obs_dim * 2,
            train_cfg['amp_reward_coef'],
            train_cfg['amp_discr_hidden_dims'], device,
            train_cfg['amp_task_reward_lerp']).to(self.device)

        # self.discr: AMPDiscriminator = AMPDiscriminator()
        alg_class = eval(self.alg_cfg["class_name"]) # PPO
        min_std = (
            torch.tensor(self.cfg["amp_min_normalized_std"], device=self.device) *
            (torch.abs(self.env.dof_pos_limits[0, :, 1] - self.env.dof_pos_limits[0, :, 0])))
        self.alg: BMPPPO = alg_class(actor_critic, discriminator, amp_data, amp_normalizer, device=self.device, min_std=min_std, **self.alg_cfg)
        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        self.save_interval = self.cfg["save_interval"]

        # init storage and model
        self.alg.init_storage(self.env.num_envs, self.num_steps_per_env, [num_actor_obs], [self.env.num_privileged_obs], [self.env.num_actions])

        # Log
        self.log_dir = log_dir
        self.writer = None
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0

        _, _ = self.env.reset()
    
    def learn(self, num_learning_iterations, init_at_random_ep_len=False):
        # initialize writer
        if self.log_dir is not None and self.writer is None:
            self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(self.env.episode_length_buf, high=int(self.env.max_episode_length))
        obs = self.env.get_observations()
        privileged_obs = self.env.get_privileged_observations()
        amp_obs = self.env.get_amp_observations()
        critic_obs = privileged_obs if privileged_obs is not None else obs
        obs, critic_obs, amp_obs = obs.to(self.device), critic_obs.to(self.device), amp_obs.to(self.device)
        self.alg.actor_critic.train() # switch to train mode (for dropout for example)
        self.alg.discriminator.train()

        ep_infos = []
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        tot_iter = self.current_learning_iteration + num_learning_iterations
        for it in range(self.current_learning_iteration, tot_iter):
            start = time.time()
            # Rollout
            with torch.inference_mode():
                for i in range(self.num_steps_per_env):
                    actions = self.alg.act(obs, critic_obs, amp_obs)
                    obs, privileged_obs, task_rewards, dones, infos, reset_env_ids, terminal_amp_states = self.env.step(actions, not_amp=False)
                    next_amp_obs = self.env.get_amp_observations()

                    critic_obs = privileged_obs if privileged_obs is not None else obs
                    obs, critic_obs, next_amp_obs, task_rewards, dones = obs.to(self.device), critic_obs.to(self.device), next_amp_obs.to(self.device), task_rewards.to(self.device), dones.to(self.device)

                    # Account for terminal states.  
                    next_amp_obs_with_term = torch.clone(next_amp_obs)
                    next_amp_obs_with_term[reset_env_ids] = terminal_amp_states

                    lerp_rewards, d_logits, amp_rewards = self.alg.discriminator.predict_amp_reward(
                        amp_obs, next_amp_obs_with_term, task_rewards, normalizer=self.alg.amp_normalizer)
                    amp_obs = torch.clone(next_amp_obs)
                    full_rewards = torch.cat([task_rewards.reshape(-1,1), amp_rewards.reshape(-1,1)], dim=-1)
                    self.alg.process_env_step(full_rewards, dones, infos, next_amp_obs_with_term)
                    
                    if self.log_dir is not None:
                        # Book keeping
                        if 'episode' in infos:
                            ep_infos.append(infos['episode'])
                        if 'log' in infos:
                            ep_infos.append(infos['log'])
                        cur_reward_sum += lerp_rewards
                        cur_episode_length += 1
                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                        lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                        cur_reward_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0

                stop = time.time()
                collection_time = stop - start

                # Learning step
                start = stop
                self.alg.compute_returns(critic_obs)
            
            mean_value_loss, mean_surrogate_loss, \
            mean_amp_loss, mean_grad_pen_loss, \
            mean_policy_pred, mean_expert_pred, alpha_coef = \
                self.alg.update()
                
            stop = time.time()
            learn_time = stop - start
            if self.log_dir is not None:
                self.log(locals())
            if it % self.save_interval == 0:
                self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(it)))
            ep_infos.clear()
        
        self.current_learning_iteration += num_learning_iterations
        self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(self.current_learning_iteration)))

    def log(self, locs, width=80, pad=35):
        self.writer.add_scalar('Loss/alpha_coef', locs['alpha_coef'], locs['it'])
        return super().log(locs, width, pad)