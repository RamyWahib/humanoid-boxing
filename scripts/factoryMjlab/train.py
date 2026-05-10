"""Train an AMP agent on top of the mjlab simulator backend.

Mirrors :mod:`scripts.factoryIsaac.train` but uses mjlab instead of
IsaacLab. Tasks are looked up via ``mjlab.tasks.registry`` (e.g.
``Mjlab-AMP-Velocity-Flat-Unitree-G1``); the env is wrapped with the
mjlab AMP wrapper, then handed to :class:`AMPOnPolicyRunner`.

Example::

    uv run python scripts/factoryMjlab/train.py \\
        Mjlab-AMP-Velocity-Flat-Unitree-G1 \\
        --agent.amp-data.motion-files data/demo/punch_000.npz
"""

from __future__ import annotations

import os
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Literal

import tyro

from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.scripts._cli import maybe_print_top_level_help
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg
from mjlab.utils.gpu import select_gpus
from mjlab.utils.os import dump_yaml, get_checkpoint_path
from mjlab.utils.torch import configure_torch_backends
from mjlab.utils.wrappers import VideoRecorder

from beyondAMP.mjlab.rsl_rl import AMPEnvWrapper, AMPRunnerCfg
from rsl_rl_amp.runners.amp_on_policy_runner import AMPOnPolicyRunner


@dataclass(frozen=True)
class TrainConfig:
  env: ManagerBasedRlEnvCfg
  agent: AMPRunnerCfg
  video: bool = False
  video_length: int = 200
  video_interval: int = 2000
  log_root: str = "logs/rsl_rl"
  gpu_ids: list[int] | Literal["all"] | None = field(default_factory=lambda: [0])

  @staticmethod
  def from_task(task_id: str) -> "TrainConfig":
    env_cfg = load_env_cfg(task_id)
    agent_cfg = load_rl_cfg(task_id)
    assert isinstance(agent_cfg, AMPRunnerCfg), (
      f"Task '{task_id}' is not an AMP task — expected AMPRunnerCfg, got "
      f"{type(agent_cfg).__name__}. Use mjlab's stock train.py for non-AMP tasks."
    )
    return TrainConfig(env=env_cfg, agent=agent_cfg)


def run_train(task_id: str, cfg: TrainConfig, log_dir: Path) -> None:
  cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
  if cuda_visible == "":
    device = "cpu"
  else:
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    os.environ["MUJOCO_EGL_DEVICE_ID"] = str(local_rank)
    device = f"cuda:{local_rank}"

  configure_torch_backends()

  cfg.agent.seed = cfg.agent.seed
  cfg.env.seed = cfg.agent.seed

  print(f"[INFO] Training AMP on mjlab: task={task_id} device={device}")
  print(f"[INFO] Logging experiment in directory: {log_dir}")

  env = ManagerBasedRlEnv(
    cfg=cfg.env, device=device, render_mode="rgb_array" if cfg.video else None
  )

  if cfg.video:
    env = VideoRecorder(
      env,
      video_folder=Path(log_dir) / "videos" / "train",
      step_trigger=lambda step: step % cfg.video_interval == 0,
      video_length=cfg.video_length,
      disable_logger=True,
    )
    print("[INFO] Recording videos during training.")

  env = AMPEnvWrapper(
    env,
    clip_actions=cfg.agent.clip_actions,
    motion_dataset=cfg.agent.amp_data,
  )

  agent_cfg_dict = asdict(cfg.agent)
  env_cfg_dict = asdict(cfg.env)

  dump_yaml(log_dir / "params" / "env.yaml", env_cfg_dict)
  dump_yaml(log_dir / "params" / "agent.yaml", agent_cfg_dict)

  runner = AMPOnPolicyRunner(
    env, agent_cfg_dict, log_dir=str(log_dir), device=device
  )

  if cfg.agent.resume:
    log_root_path = log_dir.parent
    resume_path = get_checkpoint_path(
      log_root_path, cfg.agent.load_run, cfg.agent.load_checkpoint
    )
    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    runner.load(str(resume_path))

  runner.learn(
    num_learning_iterations=cfg.agent.max_iterations, init_at_random_ep_len=True
  )

  env.close()


def launch_training(task_id: str, args: TrainConfig | None = None) -> None:
  args = args or TrainConfig.from_task(task_id)

  log_root_path = (Path(args.log_root) / args.agent.experiment_name).resolve()
  log_dir_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
  if args.agent.run_name:
    log_dir_name += f"_{args.agent.run_name}"
  log_dir = log_root_path / log_dir_name

  selected_gpus, _ = select_gpus(args.gpu_ids)
  if selected_gpus is None:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
  else:
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, selected_gpus))
  os.environ["MUJOCO_GL"] = "egl"

  run_train(task_id, args, log_dir)


def main() -> None:
  maybe_print_top_level_help("train")

  # Importing the package populates the registry with AMP tasks.
  import amp_tasks_mjlab  # noqa: F401
  import mjlab.tasks  # noqa: F401

  # Restrict task choices to AMP variants so users can't accidentally
  # invoke this script on a non-AMP task.
  amp_tasks = [t for t in list_tasks() if isinstance(load_rl_cfg(t), AMPRunnerCfg)]
  if not amp_tasks:
    raise RuntimeError(
      "No AMP tasks registered. Make sure 'amp_tasks_mjlab' is installed."
    )

  import mjlab

  chosen_task, remaining_args = tyro.cli(
    tyro.extras.literal_type_from_choices(amp_tasks),
    add_help=False,
    return_unknown_args=True,
    config=mjlab.TYRO_FLAGS,
  )

  args = tyro.cli(
    TrainConfig,
    args=remaining_args,
    default=TrainConfig.from_task(chosen_task),
    prog=sys.argv[0] + f" {chosen_task}",
    config=mjlab.TYRO_FLAGS,
  )
  del remaining_args

  launch_training(task_id=chosen_task, args=args)


if __name__ == "__main__":
  main()
