"""Register Unitree G1 velocity AMP tasks (mjlab backend).

Tasks are registered into ``mjlab.tasks.registry`` with ``runner_cls=None``
because the AMP wrapper differs from mjlab's stock ``RslRlVecEnvWrapper``.
Use ``scripts/factoryMjlab/train.py`` (which wraps the env with
:class:`beyondAMP.mjlab.rsl_rl.AMPEnvWrapper` and runs
:class:`rsl_rl_amp.runners.AMPOnPolicyRunner`) instead of mjlab's stock
``train.py``.
"""

from mjlab.tasks.registry import register_mjlab_task

from .agents.amp_ppo_cfg import unitree_g1_amp_runner_cfg
from .amp_env_cfg import unitree_g1_flat_amp_env_cfg, unitree_g1_rough_amp_env_cfg

register_mjlab_task(
  task_id="Mjlab-AMP-Velocity-Flat-Unitree-G1",
  env_cfg=unitree_g1_flat_amp_env_cfg(),
  play_env_cfg=unitree_g1_flat_amp_env_cfg(play=True),
  rl_cfg=unitree_g1_amp_runner_cfg(),
  runner_cls=None,
)

register_mjlab_task(
  task_id="Mjlab-AMP-Velocity-Rough-Unitree-G1",
  env_cfg=unitree_g1_rough_amp_env_cfg(),
  play_env_cfg=unitree_g1_rough_amp_env_cfg(play=True),
  rl_cfg=unitree_g1_amp_runner_cfg(),
  runner_cls=None,
)
