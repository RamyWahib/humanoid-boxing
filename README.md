# beyondAMP: One-Step Integration of AMP into IsaacLab & mjlab

## Overview

**beyondAMP** provides a unified pipeline to integrate Adversarial Motion Priors (AMP) into any IsaacLab robot setup, with minimal modifications and full compatibility with custom robot designs. The same AMP machinery (discriminator, motion dataset, observation groups, on-policy runner) is also available against the [mjlab](https://github.com/mujocolab/mjlab) (MuJoCo-Warp) backend — see [mjlab Backend](#-mjlab-backend) below. [中文README](./README_cn.md)

**Supported backends:**

| Backend | Tasks package | Train entry | Reference task |
| :--- | :--- | :--- | :--- |
| IsaacLab (PhysX) | `source/amp_tasks` | `scripts/factoryIsaac/train.py` | `beyondAMP-DemoPunch-G1-BasicAMP` |
| mjlab (MuJoCo-Warp) | `source/amp_tasks_mjlab` | `scripts/factoryMjlab/train.py` | `Mjlab-AMP-Velocity-Flat-Unitree-G1` |

## 🚀 Fast Setup

```bash
cd beyondAMP
bash scripts/setup_ext.sh
# Downloads assets, robot configs, and installs dependencies
```

Optional VSCode workspace setup:

```bash
python scripts/setup_vscode.py
```

## 📌 How to Use

### Quick Start

* Basic environment: `source/amp_tasks/amp_tasks/amp`
* PPO config for G1 robot: `source/amp_tasks/amp_tasks/amp/robots/g1/rsl_rl_ppo_cfg.py`

Training can be launched with:
```bash
python scripts/factoryIsaac/train.py --task beyondAMP-DemoPunch-G1-BasicAMP --headless
# python scripts/factoryIsaac/train.py --task beyondAMP-DemoPunch-G1-SoftAMPTrack --headless
# python scripts/factoryIsaac/train.py --task beyondAMP-DemoPunch-G1-HardAMPTrack --headless
```

To evaluate or visualize a trained checkpoint:
```bash
python scripts/factoryIsaac/play.py --headless --target <path to your ckpt.pt> --video --num_envs 32
```

|  AMP Punch (3k)     | Motion Tracking Punck (30k)         | AMP Dog Move | AMP Knee Walk |
| :------------------------ | :---------------------------- |:------------ |:------------- |
| <img src="data/demo/punch/amp.gif" alt="3k" style="max-height: 200px; width: auto;"> | <img src="data/demo/punch/tracking.gif" alt="30k" style="max-height: 200px; width: auto;"> | <img src="data/demo/dog_move/dog_move.gif" alt="dog_move" style="max-height: 200px; width: auto;"> | <img src="data/demo/knee_walk/knee_walk.gif" alt="knee_walk" style="max-height: 200px; width: auto;"> |

### Dataset Preparation

The dataset follows the same structure and conventions used in BeyondMimic(whole_body_tracking). All motion sequences should be stored as *.npz files and placed under data/datasets/, maintaining a consistent directory layout with the reference pipeline.

For motion retargeting and preprocessing, GMR is recommended for generating high-quality retargeted mocap data. TrackerLab may be used to perform forward kinematics checks and robot-specific adjustments, ensuring the motions remain physically plausible for your robot model.

With these tools, the dataset organization naturally aligns with the conventions established in BeyondMimic(whole_body_tracking), enabling seamless integration with the AMP training pipeline.

> Following the dataset pipeline of **BeyondMimic**:
> 
> * Motion files: place `*.npz` into `data/datasets/`
> * Recommended tools:
>   * **GMR** for retargeted motion
>   * **TrackerLab** for FK validation & robot-specific preprocessing

### G1 Punch-Hit Task (custom, on top of beyondAMP)

Built on the beyondAMP/IsaacLab pipeline above, `source/amp_tasks/amp_tasks/others/g1_punch_hit/` trains a Unitree G1 to punch a target sphere, comparing a pure-PPO policy against an AMP-regularized one.

* Env + task config: `g1_punch_amp_only_env_cfg.py`
* Punch command (target sampling, hit detection, sphere visualization): `mdp/commands.py`
* Rewards, including the punch-efficiency term: `mdp/rewards.py`
* Registered tasks: `beyondAMP-PunchHitTask-G1-PPO` (pure RL) and `beyondAMP-PunchHitTask-G1-AMPBasic` (AMP)

The `punch_efficiency_penalty` reward (`mdp/rewards.py`) penalizes hand speed outside the punch-action time window, so the policy learns to keep the arm still between punches instead of flailing for free reward.

Train both policies:
```bash
python scripts/factoryIsaac/train.py --task beyondAMP-PunchHitTask-G1-PPO --headless
python scripts/factoryIsaac/train.py --task beyondAMP-PunchHitTask-G1-AMPBasic --headless
```

Resume training for more iterations from a checkpoint:
```bash
python scripts/factoryIsaac/train.py --task beyondAMP-PunchHitTask-G1-PPO --resume --checkpoint <path/to/model_N.pt> --headless
```

Play a trained checkpoint (always cap `--num_envs` for GUI playback, or the scene will try to render the full training env count and appear to freeze):
```bash
python scripts/factoryIsaac/play.py --target logs/rsl_rl/g1_punch_hit/<run_name>/model_<iter>.pt --num_envs 1
```

Trained checkpoints live under `logs/rsl_rl/g1_punch_hit/<timestamp>_<ppo|amp>/model_<iteration>.pt`.

### AMP Integration Details

* AMP observation group added via a new `amp` observation config
* RSL-RL integration: `source/rsl_rl/rsl_rl/env/isaaclab/amp_wrapper.py`
* Default transition builder: `source/beyondAMP/beyondAMP/amp_obs.py`

> For full tutorial and customization, see `docs/tutorial.md`.

## 🧩 mjlab Backend

beyondAMP also ships an [mjlab](https://github.com/mujocolab/mjlab)-backed pipeline that mirrors the IsaacLab path. mjlab adopts IsaacLab's manager-based API on top of MuJoCo-Warp, so the AMP discriminator, motion dataset, and on-policy runner are reused — only the env wrapper and observation hooks differ.

### Layout

| Path | Purpose |
| :--- | :--- |
| `source/amp_tasks_mjlab/` | mjlab-side AMP task package (registers `Mjlab-AMP-*` tasks) |
| `source/beyondAMP/beyondAMP/mjlab/` | mjlab-specific obs groups, mdp helpers, and rsl_rl wrappers |
| `scripts/factoryMjlab/` | Training entry point for mjlab AMP tasks (see its [README](scripts/factoryMjlab/README.md)) |

### Install

`scripts/setup_ext.sh` currently installs the IsaacLab task package only. For the mjlab backend, additionally install:

```bash
pip install mjlab                       # MuJoCo-Warp backend
pip install -e source/amp_tasks_mjlab   # registers Mjlab-AMP-* tasks
```

### Train

```bash
uv run python scripts/factoryMjlab/train.py \
    Mjlab-AMP-Velocity-Flat-Unitree-G1 \
    --agent.amp-data.motion-files data/demo/punch_000.npz
```

Tasks are looked up via `mjlab.tasks.registry`; the env is wrapped with `beyondAMP.mjlab.rsl_rl.AMPEnvWrapper` and trained with `rsl_rl_amp.runners.AMPOnPolicyRunner`. Registered out of the box:

* `Mjlab-AMP-Velocity-Flat-Unitree-G1`
* `Mjlab-AMP-Velocity-Rough-Unitree-G1`

To add your own mjlab AMP task, follow the pattern in `source/amp_tasks_mjlab/amp_tasks_mjlab/velocity/g1/__init__.py` and call `register_mjlab_task(..., runner_cls=None)` (the AMP wrapper replaces mjlab's stock `RslRlVecEnvWrapper`).


<details>
<summary><strong>Additional Notes</strong></summary>

* Fully modular AMP observation builder
* Compatible with IsaacLab 4.5+
* Designed for rapid experimentation across robot morphologies

</details>

## 🙏 Acknowledgement

### Referenced Repositories

| Repository                                                           | Purpose                               |
| -------------------------------------------------------------------- | ------------------------------------- |
| [robotlib](https://github.com/Renforce-Dynamics/robotlib)            | Robot configurations                  |
| [assetslib](https://github.com/Renforce-Dynamics/assetslib)          | Asset storage                         |
| [TrackerLab](https://github.com/Renforce-Dynamics/trackerLab)        | Data organization & retargeting tools |
| [AMP_for_hardware](https://github.com/escontra/AMP_for_hardware)     | AMP implementation reference          |
| [BeyondMimic](https://github.com/HybridRobotics/whole_body_tracking) | Dataset format & tracking comparison  |

---

## 📘 Citation

```bibtex
@software{zheng2025@beyondAMP,
  author = {Ziang Zheng},
  title = {beyondAMP: One step unify IsaacLab with AMP.},
  url = {https://github.com/Renforce-Dynamics/beyondAMP},
  year = {2025}
}
```

```bibtex
@software{zheng2025@trackerLab,
  author = {Ziang Zheng},
  title = {TrackerLab: One step unify IsaacLab with multi-mode whole-body control.},
  url = {https://github.com/interval-package/trackerLab},
  year = {2025}
}

@INPROCEEDINGS{Escontrela@amphardware,
  author={Escontrela, Alejandro and Peng, Xue Bin and Yu, Wenhao and Zhang, Tingnan and Iscen, Atil and Goldberg, Ken and Abbeel, Pieter},
  booktitle={2022 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)}, 
  title={Adversarial Motion Priors Make Good Substitutes for Complex Reward Functions}, 
  year={2022}
  }

@article{liao2025beyondmimic,
  title={Beyondmimic: From motion tracking to versatile humanoid control via guided diffusion},
  author={Liao, Qiayuan and Truong, Takara E and Huang, Xiaoyu and Tevet, Guy and Sreenath, Koushil and Liu, C Karen},
  journal={arXiv preprint arXiv:2508.08241},
  year={2025}
}
```

## Community

| Renforce Dynamics | **Join our WeChat Group** |
|:--|:--|
| | <img src="https://github.com/Renforce-Dynamics/.github/blob/main/pics/wechat_group/group.jpg" alt="Renforce Dynamics WeChat Group" height="180"> |
