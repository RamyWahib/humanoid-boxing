## Temporal Phase Design for Punch Skill

The punch skill is explicitly structured by two time windows defined in the command:

```python
punch_action_time_range    = (t_act_start, t_act_end)
punch_expected_time_range  = (t_hit_start, t_hit_end)
```

These two windows partition the command execution into **five distinct temporal phases**, each with a clear behavioral and optimization objective.

---

### Phase 1: Pre-action Stabilization

**Time:** `t < t_act_start`

* The robot is **not yet allowed to punch**
* Desired behavior:

  * Maintain posture and balance
  * Minimize unnecessary motion
* Reward focus:

  * Stability-related terms
* No punch-specific shaping is active

---

### Phase 2: Action Initiation

**Time:** `t_act_start ≤ t < t_hit_start`

* The punch motion is allowed to begin
* Desired behavior:

  * Initiate movement toward the target
  * Build up hand velocity
* Reward focus:

  * Distance-to-target shaping
  * Velocity alignment toward the target
* Hit is not yet evaluated

---

### Phase 3: Expected Hit Window

**Time:** `t_hit_start ≤ t ≤ t_hit_end`

* This is the **valid punch window**
* Desired behavior:

  * Strike the target with sufficient speed
  * Precise timing
* Reward focus:

  * On-time hit bonus
  * Impact quality
* Any geometric hit in this window is considered a **successful punch**

---

### Phase 4: Late Action / Missed Hit

**Time:** `t_hit_end < t ≤ t_act_end`

* Punch motion is still allowed
* However, the optimal hit window has passed
* Desired behavior:

  * Avoid late or sloppy strikes
* Reward focus:

  * No hit bonus
  * Possibly reduced shaping
* Late hits are treated as failures (task-dependent)

---

### Phase 5: Post-action Recovery

**Time:** `t > t_act_end`

* Punch action is finished
* Desired behavior:

  * Recover balance
  * Reduce residual angular and linear momentum
* Reward focus:

  * Stability and smoothness
* Typically followed by command resampling or episode termination

---

## Design Principles

* **Explicit temporal structure** enables phase-aware learning without hard-coded motion scripts.
* **Action and evaluation windows are decoupled**:

  * The robot can move before and after the hit window.
  * Only hits within the expected window are rewarded.
* **Successful hits do not terminate the episode**:

  * Instead, a new punch command is resampled.
  * This enables continuous striking behaviors.

---

## Intuition

This design mirrors human punching behavior:

* Prepare → initiate → strike → recover
  while providing clean learning signals for:
* *when* to punch
* *how* to punch
* *when a punch is considered successful*