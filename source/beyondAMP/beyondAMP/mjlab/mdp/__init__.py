"""mjlab MDP terms used by AMP observation groups.

Re-exports the velocity task's MDP plus the body-tracking observations that
AMP needs (``body_pos_w``/``body_quat_w``/...). The body-tracking funcs read
mjlab's ``body_link_*`` fields under the IsaacLab-compatible names so the
AMP obs group code can stay backend-agnostic.
"""

from mjlab.envs.mdp import *  # noqa: F401,F403
from mjlab.tasks.velocity.mdp import *  # noqa: F401,F403

from .observations import (  # noqa: F401
  anchor_ang_vel_w,
  anchor_lin_vel_w,
  anchor_pos_w,
  anchor_quat_w,
  base_ang_vel_yaw,
  base_lin_vel_yaw,
  body_ang_vel_w,
  body_lin_vel_w,
  body_pos_w,
  body_quat_w,
)
