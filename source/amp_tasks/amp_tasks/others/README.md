# AMP tasks for other types

These tasks may not have specific type, hence we denote it as `others` where the rewards is governed by each task itself.

## Notice

- `amp_only`: The AMP Only task is very dangerous where the tasks will give only alive signals and may not give desired guidance. 
    - Policy train in this task is easily to crush.
    - (The task name will contain the `amp_only`)
- Other tasks will be guided with more rewards.
- `reset_to_ref_motion_dataset`: This reset method will reset robot to its reference, which will give a lot information for the policy.