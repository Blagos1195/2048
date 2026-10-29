# Self-learning 2048 AI

This repository's main agent is a **self-learning n-tuple temporal-difference (TD) learner** guided by Expectimax search. It starts with zero learned value weights and improves from real game transitions. The old DQN implementation, its incompatible checkpoints, generated Python caches, and unused UI files have been removed so there is one main learning system rather than two competing ones.

## Install

Python 3.10+ is recommended.

```bash
python -m pip install -r requirements.txt
python -m pytest -q
```

## Run the AI

Headless training is the default:

```bash
python main.py
```

This trains for 100,000 games by default and saves checkpoints and a CSV training log under `models/selfplay/`. To run a shorter test:

```bash
python main.py --games 100
```

The training/evaluation CLI is also available directly:

```bash
python selfplay_ai.py train --games 100000
python selfplay_ai.py evaluate --model models/selfplay/best.npz --games 100 --depth 2
python selfplay_ai.py train --resume models/selfplay/latest.npz --games 100000 --out models/selfplay
```

## View modes

All four graphical modes use the new learned-value agent. They can be launched from `main.py`:

- `python main.py --mode 1 --games 10000`: four-board live training dashboard, faster animation.
- `python main.py --mode 2 --games 10000`: four-board live training dashboard, slower animation.
- `python main.py --mode 3`: watch the saved AI play automatically.
- `python main.py --mode 4`: **LARP mode**. Press any key to let the AI make one move while you receive the credit. Press `R` to restart and `Esc` to quit.

Modes 3 and 4 need a trained checkpoint. Train first, or pass an explicit checkpoint using `--model path/to/model.npz`. The view modes are optional; headless training does not need a display.

## How it learns

- **N-tuple TD learning:** eight overlapping four-cell row/column patterns provide a learned estimate of a board's long-term value. Weights start at zero and update after observed transitions.
- **Expectimax search:** evaluates legal moves and the random appearance of 2 and 4 tiles, using the learned value function rather than hard-coded board-shape bonuses.
- **Self-play and exploration:** the agent generates its own games and uses a decaying epsilon rate to try random legal moves while learning.
- **Evaluation and checkpoints:** periodic evaluation records average score, maximum tiles and 2048/4096/8192 success rates. The latest and best evaluated checkpoints are saved separately.

There are no fixed corner bonuses, snake-pattern bonuses, opening scripts, or preferred-move rules. The search predicts likely consequences, but its decisions are only as good as its learned value estimate and search depth.

## Compute and performance

The learner uses NumPy and runs on CPU; it does not currently use a GPU, distributed workers, or parallel game simulation. Training throughput is the limiting factor on hosted CI runners, so the GitHub workflow trains in resumable chunks instead of relying on a single job that can exceed the runner time limit. More games and deeper search can improve decisions, but also cost more compute. Always compare results across many fixed-seed evaluation games rather than trusting one unusually good run.

## Output files

- `models/selfplay/latest.npz`: latest learned value weights and training metadata.
- `models/selfplay/best.npz`: strongest model found at a periodic evaluation.
- `models/selfplay/training.csv`: per-game metrics and periodic evaluation summaries.

The tests cover move/merge rules, terminal detection, deterministic game seeds, pattern indexing, TD updates, legal action selection, and spawn expectation.
