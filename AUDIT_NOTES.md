# Code audit notes

This reviewed build retains the existing DQN experiment and its saved `log_reward128` files. The active configuration remains `tile_planes_symmetry_v2`, which uses a separate model directory.

## Fixes and safeguards

- Fixed the startup crash by reading replay-buffer capacity from `memory_buffer.capacity`, not the nonexistent `maxlen` attribute.
- Metrics logging falls back to `training_metrics_recovery.csv` if the main CSV is locked, and logging failure does not itself stop training.
- Checkpoint resume now rejects a state-encoding mismatch with a clear error instead of silently loading an incompatible encoding into the current experiment.
- Checkpoint loading rejects unknown encodings and incompatible input/output dimensions early.
- Agent construction rejects misspelled state-encoding names immediately.
- Added coverage for invalid encodings and saving/loading the `tile_planes` training state.

## Validation performed

- Python syntax compilation passed for `ai.py`, `game.py`, `main.py`, and `ui.py`.
- All 16 automated tests passed.
- Symmetry action and legal-mask mappings were checked across all four rotations and both reflection states on 100 random boards.
- The graphical interface was not launched as part of this code audit, so a Windows GUI smoke test remains unverified.
