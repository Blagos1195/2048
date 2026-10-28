# Self-learning 2048 AI

This is a standalone implementation of a self-learning 2048 agent. It does not depend on the existing DQN, pygame, a graphical interface, a pretrained model, a GPU, or a paid API.

The agent combines:
- **N-tuple temporal-difference learning:** a learned board-value function starts with zero weights and updates from actual game transitions.
- **Expectimax search:** plans through legal moves and random tile spawns, using the learned value function at the search frontier.
- **Self-play:** generates its own games and learns from their observed rewards and outcomes.
- **Exploration:** epsilon-greedy random legal moves reduce gradually during training.
- **Evaluation and checkpoints:** records metrics, saves the latest model, and retains the strongest model found by its periodic evaluation score.

There are no hard-coded corner bonuses, snake patterns, opening scripts, or preferred-move rules. This is a learned evaluator, not just a fixed heuristic. The architecture is intentionally NumPy-only to keep installation simple and avoid the Python-version issues associated with some game and deep-learning packages.

## Install

Python 3.10+ is recommended. Install the standalone dependencies:

\`\`\`bash
python -m pip install -r requirements-selfplay.txt
python -m pytest -q tests/test_selfplay.py
\`\`\`

If NumPy cannot be installed on your Python version, use a currently supported Python release. The agent itself does not require pygame or PyTorch.

## Start training from scratch

\`\`\`bash
python selfplay_ai.py train
\`\`\`

Defaults: 100,000 games, depth-1 search during training for throughput, 20% initial random exploration decaying toward 2%, evaluation every 500 games, and output under \`models/selfplay/\`. Progress is printed live. Training begins with randomly initialised (zero) value weights; no existing checkpoint is required.

A quick smoke run:

\`\`\`bash
python selfplay_ai.py train --games 10 --eval-every 5 --eval-games 3 --depth 1 --eval-depth 1 --out models/smoke
\`\`\`

Depth 2 search is considerably more expensive than depth 1. Raise \`--depth\` when stronger lookahead is worth the extra runtime. The separate \`--eval-depth\` option controls search depth for periodic evaluations.

## Evaluate a saved model

\`\`\`bash
python selfplay_ai.py evaluate --model models/selfplay/best.npz --games 100 --depth 2
\`\`\`

The report includes average and median score, average and best maximum tile, 2048/4096/8192 success rates, and average moves. Use enough games to avoid mistaking a lucky run for a genuine improvement.

## Resume training

\`\`\`bash
python selfplay_ai.py train --resume models/selfplay/latest.npz --games 100000 --out models/selfplay_continued
\`\`\`

This keeps the new run's CSV separate. If you intentionally want to append metrics to an existing \`training.csv\`, use \`--resume-metrics\` and set \`--out\` to that same experiment folder.

## Output files

- \`models/selfplay/latest.npz\`: latest value tables and learning parameters.
- \`models/selfplay/best.npz\`: best model seen at a periodic evaluation checkpoint.
- \`models/selfplay/training.csv\`: per-game scores, maximum tiles, move counts, exploration rate, TD error and update count.

The branch contains the simulator, learning algorithm, search, training/evaluation CLI, checkpointing and tests. No code edits are needed to start it. Training is a compute task, so a fresh checkout does not contain a magically pre-trained model; run the training command to generate one locally.

## Method notes

The value function uses eight overlapping four-cell row and column patterns, with tile values encoded as exponents. A TD(0) update fits the current board value toward the normalised merge reward plus discounted value of the next state. Expectimax models empty-cell spawns as 2 with probability 0.9 and 4 with probability 0.1. The evaluation seed range is separated from the training seed range for reproducibility.

This is a complete working baseline designed for efficient self-learning, not a guarantee of 2048/4096 success before training. Playing strength must be measured from the saved model after a meaningful training run.
