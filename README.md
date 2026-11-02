# Self-learning 2048 AI

A single self-learning 2048 project. **Run every user-facing command through `main.py`**; `selfplay_ai.py` and `selfplay_ui.py` are implementation modules, not separate entry points.

## Install

Python 3.10+ is recommended. There is one dependency file:

```bash
python -m pip install -r requirements.txt
```

This installs NumPy, **pygame-ce** (imported in Python as `pygame`), Numba for accelerated search, and pytest.

Run the tests:

```bash
python -m pytest -q
```

## Easy launcher

For the normal way to use the project, run:

```bash
python run.py
```

This opens a graphical launcher with buttons for training, continuing training, watching the AI, LARP mode, live training, evaluation, comparisons, and tests. Each activity has its own settings window and confirmation screen, so you can configure it without memorising command-line arguments.

Game modes open the 2048 game window. Closing the game returns you to the launcher. Training, evaluation, comparisons, and tests run as background tasks while their progress and output appear in the terminal. When they finish, the launcher is ready for the next task.

The launcher uses Python's built-in Tkinter interface. On some Linux installations, Tkinter may need to be installed separately (often through the system package named `python3-tk`).

The command-line options below are still available if you want more control.

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

## Performance checks

Run a repeatable comparison of search depths 1, 2, and 3 on the same model and
fixed game seeds. The CSV includes score, tile, milestone-rate, and throughput
metrics:

```bash
python scripts/benchmark_search.py --model models/selfplay/best.npz --games 1 --depths 1 2 3 --seed 4200000 --output artifacts/benchmark.csv
```

Profile expectimax/search with cProfile and save the cumulative-time summary as
plain text:

```bash
python profile_search.py --model models/selfplay/best.npz --depth 1 --moves 8 --output artifacts/profile.txt
```

GitHub Actions runs the test suite, a two-worker training smoke run, and a resume-from-checkpoint check. Performance comparisons are run separately when evaluating code changes.

## Graphical modes

These modes use pygame-ce and launch from the same entry point:

- `python main.py --mode 1 --games 10000`: four-board training dashboard with faster animation.
- `python main.py --mode 2 --games 10000`: four-board training dashboard with slower animation.
- `python main.py --mode 3 --model models/selfplay/best.npz --depth 2`: watch the trained AI play automatically.
- `python main.py --mode 4 --model models/selfplay/best.npz --depth 2`: LARP mode. Press a key for the AI's next move, `R` to restart, and `Esc` or `Q` to quit.
