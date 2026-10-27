# 2048 DQN Lab

A NumPy-based Double-DQN learner for 2048. The agent starts from fresh random weights and discovers strategy from experience. There are **no hard-coded corner bonuses, snake-pattern bonuses, opening scripts, or preferred-move rules**.

## Install and run

Python 3.10+ is recommended.

```bash
python -m pip install -r requirements.txt
python main.py
```

The default `RUN_MODE = 0` trains headlessly. Press Enter in the terminal to open the live showcase/replay menu. Change `RUN_MODE` at the top of `main.py` to use the existing modes:

- `0`: headless training and terminal-triggered showcase/replay
- `1`: multi-board live dashboard
- `2`: slower multi-board live dashboard
- `3`: single-game showcase
- `4`: LARP mode, where the model pilots moves while the interface lets you take the credit

In graphical training modes, `SPACE` opens the live showcase, `R` opens the best saved replay, and `S` saves a checkpoint. Replay controls are Space to pause, left/right arrows to step, up/down arrows to change speed, and Escape to close.

## Learning design

- **Two interchangeable state encodings:** `STATE_ENCODING = "log"` uses 16 logarithmically scaled values; `"tile_planes"` uses 17 one-hot tile categories for each cell (272 inputs). Every board position is treated identically by either encoding.
- **Double DQN:** online network chooses the next action, target network evaluates it, with legal-action masking.
- **Prioritized replay:** fixed-size NumPy arrays avoid rebuilding a Python list on every batch. High-TD-error experiences are sampled more often, with importance-sampling weights.
- **Three-step returns:** reward information can propagate backwards through multiple moves. Terminal episodes flush pending transitions correctly.
- **Stability:** Adam, gradient clipping, a target network and bounded epsilon-greedy exploration.
- **Neutral reward:** actual merge score divided by `REWARD_SCALE`, plus `SURVIVAL_REWARD` per move and `TERMINAL_PENALTY` at game over. No board-shape rewards are used.
- **Evaluation:** quick 4-game checks every 10 episodes, 50-game fixed-seed validation every 100 episodes, and a 100-game unseen-seed final evaluation at shutdown. Evaluation does not train the agent. A random legal-move baseline is recorded at startup.
- **Metrics:** `training_metrics.csv` records training episodes, evaluation scores, tile milestones, epsilon, replay size, TD error and update counts.
- **Checkpoints:** the latest checkpoint includes weights, target weights, Adam state, replay priorities and data, n-step queue, RNG state and exploration progress. The best evaluated model is kept separately.

## Experiments without accidental mixing

Each run writes to `models/<EXPERIMENT_NAME>/`. Keep a unique `EXPERIMENT_NAME` for each experiment and do not reuse a previous checkpoint when comparing configurations.

Recommended comparisons:

1. `STATE_ENCODING = "log"`, `REWARD_SCALE = 128.0`, `EXPERIMENT_NAME = "log_reward128"`.
2. `STATE_ENCODING = "tile_planes"`, `REWARD_SCALE = 128.0`, `EXPERIMENT_NAME = "planes_reward128"`.
3. Repeat the better encoding with `REWARD_SCALE = 256.0` and a new experiment name.
4. If needed, test a different `SURVIVAL_REWARD` or `TERMINAL_PENALTY` separately, changing only one value per experiment.

Use the same `RANDOM_SEED` for comparable runs when reproducibility matters. Do not compare only the best single game; compare 50-game validation and the final held-out test results. Changing reward scaling changes the learning problem, so a higher training reward alone does not mean better play.

## Tests

```bash
python -m pytest -q
```

The tests cover board rules, action masks, position-neutral encodings, prioritized replay, multi-step returns, weight updates and checkpoint restoration. Training is stochastic: more visible activity is not a guarantee of mastery, and late-game strategy requires sustained evaluation rather than one fortunate run.

## Symmetry-augmented experiment (v2)

The current `main.py` defaults to `STATE_ENCODING = "tile_planes"` and
`EXPERIMENT_NAME = "tile_planes_symmetry_v2"`. It starts a separate experiment
folder so it cannot overwrite or accidentally resume the earlier run. The DQN
randomly rotates and reflects replay transitions during learning, while remapping
actions and legal-action masks consistently. This uses only the mathematical
symmetries of the game; it does not reward corners or prescribe a strategy.
Exploration decays over 400,000 environment moves rather than 120,000.

The previous `models/log_reward128` checkpoint and CSV should be kept for
comparison. Evaluate the new run against the same fixed evaluation seeds and
compare average score, average maximum tile, and 2048 success rate.
