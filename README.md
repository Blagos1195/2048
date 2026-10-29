# Self-learning 2048 AI

A single self-learning 2048 project. **Run every user-facing command through `main.py`**; `selfplay_ai.py` and `selfplay_ui.py` are implementation modules, not separate entry points.

## Install

Python 3.10+ is recommended. There is one dependency file:

```bash
python -m pip install -r requirements.txt
```

This installs NumPy, **pygame-ce** (imported in Python as `pygame`), and pytest.

Run the tests:

```bash
python -m pytest -q
```

## Train the AI

Train from scratch using the defaults (100,000 games):

```bash
python main.py
```

Or set the training options explicitly:

```bash
python main.py train --games 100000 --depth 1 --workers 0
```

A short smoke run:

```bash
python main.py train --games 10 --eval-every 5 --eval-games 3 --depth 1 --eval-depth 1 --out models/smoke
```

Useful training options:

- `--games`: number of games to train.
- `--depth`: search depth used while collecting training games. Depth 1 is much faster.
- `--workers 0`: automatically select up to four parallel workers. Use `--workers 1` for sequential training or a larger positive number to choose the worker count.
- `--eval-every`: evaluate and save checkpoints every N games; use `0` to disable periodic evaluation.
- `--resume models/selfplay/latest.npz`: continue training from a checkpoint.
- `--resume-metrics`: append to an existing training CSV when resuming into the same output folder.
- `--out models/selfplay`: choose the output folder.

For the full list:

```bash
python main.py train --help
```

## Evaluate and benchmark

Evaluate a saved model:

```bash
python main.py evaluate --model models/selfplay/best.npz --games 100 --depth 2
```

Compare search depths using the same game seeds:

```bash
python main.py benchmark --model models/selfplay/best.npz --games 10 --depths 1 2 --seed 4000000
```

Help for either command:

```bash
python main.py evaluate --help
python main.py benchmark --help
```

## Graphical modes

These modes use pygame-ce and launch from the same entry point:

- `python main.py --mode 1 --games 10000`: four-board training dashboard with faster animation.
- `python main.py --mode 2 --games 10000`: four-board training dashboard with slower animation.
- `python main.py --mode 3 --model models/selfplay/best.npz --depth 2`: watch the trained AI play automatically.
- `python main.py --mode 4 --model models/selfplay/best.npz --depth 2`: LARP mode. Press a key for the AI's next move, `R` to restart, and `Esc` or `Q` to quit.

Modes 3 and 4 need a trained checkpoint. The program looks for `models/selfplay/best.npz` and then `latest.npz` if `--model` is omitted. Headless training does not require a display.

## How it learns

- **N-tuple TD learning:** eight overlapping four-cell row/column patterns estimate the long-term value of a board. The weights start at zero and update from observed transitions.
- **Expectimax search:** evaluates legal moves and random 2/4 tile spawns using the learned value function.
- **Self-play and exploration:** the agent generates games and gradually reduces random legal moves as training progresses.
- **Evaluation and checkpoints:** training records scores, maximum tiles, move counts, exploration rate, TD error and update counts. It saves the latest model and the best model found during periodic evaluation.

There are no fixed corner bonuses, snake-pattern bonuses, opening scripts, or preferred-move rules. The search predicts likely consequences, but its decisions are only as good as its learned value estimate and search depth.

## Compute and performance

The learner uses NumPy on CPU and collects games in parallel worker processes. Workers read a stable snapshot of the weights; the parent applies TD updates between batches. Expectimax caches repeated board/depth states during each move. There is no GPU acceleration because this n-tuple evaluator and branch-heavy search are CPU-oriented.

More training and deeper search can improve decisions, but cost more compute. Depth 2 is much slower than depth 1, so evaluate the trade-off on your machine.

## Output files

- `models/selfplay/latest.npz`: latest learned value weights and metadata.
- `models/selfplay/best.npz`: strongest model found by periodic evaluation.
- `models/selfplay/training.csv`: per-game training metrics.

The tests cover move and merge rules, terminal detection, deterministic seeds, pattern indexing, TD updates, legal action selection, and spawn expectation.
