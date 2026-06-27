# G1 Punch-Hit: Teaching a Unitree G1 to Throw a Punch

This project trains a Unitree G1 humanoid to punch a target sphere placed in front of it, comparing a pure-PPO policy against an AMP-regularized policy trained on motion-capture punching data.

Built on top of [**beyondAMP**](https://github.com/Renforce-Dynamics/beyondAMP) by Ziang Zheng, which provides the IsaacLab + AMP integration (discriminator, motion dataset, on-policy runner). See [Credit](#credit) below.

## What's here

* `source/amp_tasks/amp_tasks/others/g1_punch_hit/g1_punch_amp_only_env_cfg.py` — environment and task config for the G1 punch task
* `source/amp_tasks/amp_tasks/others/g1_punch_hit/mdp/commands.py` — punch command: samples a target position in front of the robot, tracks hit timing, and drives the red sphere marker
* `source/amp_tasks/amp_tasks/others/g1_punch_hit/mdp/rewards.py` — reward terms, including `punch_efficiency_penalty`, which penalizes hand speed outside the punch-action time window so the policy learns to keep the arm still between punches instead of flailing for reward
* Registered tasks: `beyondAMP-PunchHitTask-G1-PPO` (pure RL baseline) and `beyondAMP-PunchHitTask-G1-AMPBasic` (AMP-regularized)

## Setup

```bash
bash scripts/setup_ext.sh
```

## Train

```bash
python scripts/factoryIsaac/train.py --task beyondAMP-PunchHitTask-G1-PPO --headless
python scripts/factoryIsaac/train.py --task beyondAMP-PunchHitTask-G1-AMPBasic --headless
```

To continue training a checkpoint for more iterations, bump `max_iterations` in `source/amp_tasks/amp_tasks/others/g1_punch_hit/agents/{base_ppo_cfg,amp_ppo_cfg}.py`, then:

```bash
python scripts/factoryIsaac/train.py --task beyondAMP-PunchHitTask-G1-PPO --resume --checkpoint <path/to/model_N.pt> --headless
```

Checkpoints land under `logs/rsl_rl/g1_punch_hit/<timestamp>_<ppo|amp>/model_<iteration>.pt`.

## Play / visualize

```bash
python scripts/factoryIsaac/play.py --target <path/to/model_N.pt> --num_envs 1
```

Always cap `--num_envs` when playing — the env config defaults to thousands of parallel environments for training, and trying to render that many in the GUI looks like a freeze.

## Credit

This work builds directly on [beyondAMP](https://github.com/Renforce-Dynamics/beyondAMP) by Ziang Zheng (Renforce Dynamics), which provides the underlying AMP-in-IsaacLab pipeline this task is built on top of. See the [beyondAMP repository](https://github.com/Renforce-Dynamics/beyondAMP) for the full framework, mjlab backend, dataset tooling, and citation details.
