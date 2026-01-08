import torch
import torch.nn as nn
from torch.distributions import Normal

from .actor_critic import get_activation


class ActorCriticFPO(nn.Module):
    """Shared actor with two critics (task critic + repair/style critic).

    This is a minimal extension of the original `ActorCritic` to support
    FPO-style region-wise objectives:
      - In-Set: optimize task return (task critic used for GAE/value loss)
      - Out-Set: optimize repair/style return (repair critic used for GAE/value loss)
    """

    is_recurrent = False

    def __init__(
        self,
        num_actor_obs: int,
        num_critic_obs: int,
        num_actions: int,
        actor_hidden_dims=(256, 256, 256),
        critic_hidden_dims=(256, 256, 256),
        # Optional recover critic (for hard-violation recovery)
        use_recover_critic: bool = False,
        recover_critic_hidden_dims=None,
        activation: str = "elu",
        init_noise_std: float = 1.0,
        fixed_std: bool = False,
        **kwargs,
    ):
        if kwargs:
            print(
                "ActorCriticFPO.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )
        super().__init__()

        activation_layer = get_activation(activation)

        # --------------------
        # Actor
        # --------------------
        actor_layers = [nn.Linear(num_actor_obs, actor_hidden_dims[0]), activation_layer]
        for l in range(len(actor_hidden_dims)):
            if l == len(actor_hidden_dims) - 1:
                actor_layers.append(nn.Linear(actor_hidden_dims[l], num_actions))
            else:
                actor_layers.append(nn.Linear(actor_hidden_dims[l], actor_hidden_dims[l + 1]))
                actor_layers.append(activation_layer)
        self.actor = nn.Sequential(*actor_layers)

        # --------------------
        # Task critic
        # --------------------
        task_critic_layers = [nn.Linear(num_critic_obs, critic_hidden_dims[0]), activation_layer]
        for l in range(len(critic_hidden_dims)):
            if l == len(critic_hidden_dims) - 1:
                task_critic_layers.append(nn.Linear(critic_hidden_dims[l], 1))
            else:
                task_critic_layers.append(nn.Linear(critic_hidden_dims[l], critic_hidden_dims[l + 1]))
                task_critic_layers.append(activation_layer)
        self.task_critic = nn.Sequential(*task_critic_layers)

        # --------------------
        # Repair/style critic
        # --------------------
        repair_critic_layers = [nn.Linear(num_critic_obs, critic_hidden_dims[0]), activation_layer]
        for l in range(len(critic_hidden_dims)):
            if l == len(critic_hidden_dims) - 1:
                repair_critic_layers.append(nn.Linear(critic_hidden_dims[l], 1))
            else:
                repair_critic_layers.append(nn.Linear(critic_hidden_dims[l], critic_hidden_dims[l + 1]))
                repair_critic_layers.append(activation_layer)
        self.repair_critic = nn.Sequential(*repair_critic_layers)

        # --------------------
        # Recover critic
        # --------------------
        self.use_recover_critic = bool(use_recover_critic)
        if self.use_recover_critic:
            r_dims = recover_critic_hidden_dims if recover_critic_hidden_dims is not None else critic_hidden_dims
            recover_layers = [nn.Linear(num_critic_obs, r_dims[0]), activation_layer]
            for l in range(len(r_dims)):
                if l == len(r_dims) - 1:
                    recover_layers.append(nn.Linear(r_dims[l], 1))
                else:
                    recover_layers.append(nn.Linear(r_dims[l], r_dims[l + 1]))
                    recover_layers.append(activation_layer)
            self.recover_critic = nn.Sequential(*recover_layers)
        else:
            self.recover_critic = None

        # --------------------
        # Gaussian policy std
        # --------------------
        self.fixed_std = fixed_std
        std = init_noise_std * torch.ones(num_actions)
        self.std = torch.tensor(std) if fixed_std else nn.Parameter(std)
        self.distribution = None
        Normal.set_default_validate_args = False

    def reset(self, dones=None):
        # Non-recurrent by design.
        return

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def update_distribution(self, observations: torch.Tensor) -> None:
        mean = self.actor(observations)
        std = self.std.to(mean.device)
        self.distribution = Normal(mean, mean * 0.0 + std)

    def act(self, observations: torch.Tensor, **kwargs) -> torch.Tensor:
        self.update_distribution(observations)
        return self.distribution.sample()

    def get_actions_log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, observations: torch.Tensor) -> torch.Tensor:
        return self.actor(observations)

    # Backward-compatible: treat `evaluate()` as task value.
    def evaluate(self, critic_observations: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.evaluate_task(critic_observations, **kwargs)

    def evaluate_task(self, critic_observations: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.task_critic(critic_observations)

    def evaluate_repair(self, critic_observations: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.repair_critic(critic_observations)

    def evaluate_recover(self, critic_observations: torch.Tensor, **kwargs) -> torch.Tensor:
        if not self.use_recover_critic or self.recover_critic is None:
            raise RuntimeError(
                "ActorCriticFPO.evaluate_recover() called but recover_critic is disabled. "
                "Enable it by passing use_recover_critic=True in the policy config."
            )
        return self.recover_critic(critic_observations)















