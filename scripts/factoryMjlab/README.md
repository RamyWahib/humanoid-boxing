# factoryMjlab

mjlab-backed training entry for beyondAMP. Mirrors `factoryIsaac/` but
targets the mjlab simulator instead of IsaacLab.

```sh
# Train AMP on the G1 flat-velocity task. Provide motion files via the
# tyro CLI override so the discriminator has reference data:
uv run python scripts/factoryMjlab/train.py \
    Mjlab-AMP-Velocity-Flat-Unitree-G1 \
    --agent.amp-data.motion-files data/demo/punch_000.npz
```

Tasks are looked up via `mjlab.tasks.registry`. The AMP variants are
registered by `amp_tasks_mjlab`, so make sure that package is installed
(`pip install -e source/amp_tasks_mjlab`).
