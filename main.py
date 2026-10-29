"""Single entry point for training, evaluating, benchmarking, and viewing the 2048 AI."""
from __future__ import annotations

import argparse
import sys

AI_COMMANDS = {"train", "evaluate", "benchmark"}


def show_help() -> None:
    parser = argparse.ArgumentParser(
        prog="python main.py",
        description=(
            "Train the self-learning 2048 AI, test a saved model, compare search speeds, "
            "or open a graphical view."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""QUICK START
  python main.py
      Train a new model using the default settings.

  python main.py train --games 100000 --depth 1
      Train for 100,000 games. Depth 1 is the faster starting point.

  python main.py evaluate --model models/selfplay/best.npz --games 100 --depth 2
      Test a saved model over 100 games.

  python main.py benchmark --model models/selfplay/best.npz --depths 1 2
      Compare search depths using the same game seeds.

GRAPHICAL MODES
  python main.py --mode 1
      Show four games while training at the faster animation speed.

  python main.py --mode 2
      Show four games while training at the slower animation speed.

  python main.py --mode 3 --model models/selfplay/best.npz
      Watch a saved model play automatically.

  python main.py --mode 4 --model models/selfplay/best.npz
      Press a key to make the AI play one move. R restarts; Esc or Q quits.

DETAILED OPTIONS
  python main.py train --help
  python main.py evaluate --help
  python main.py benchmark --help
  python main.py --mode 1 --help
""",
    )
    parser.add_argument(
        "--mode",
        type=int,
        choices=range(1, 5),
        metavar="{1,2,3,4}",
        help="open a graphical mode instead of the command-line tools (see modes above)",
    )
    parser.add_argument("--games", type=int, help="number of training or evaluation games")
    parser.add_argument("--depth", type=int, help="AI search depth; higher is stronger but slower")
    parser.add_argument("--seed", type=int, help="random seed for repeatable runs")
    parser.add_argument("--out", help="folder where training checkpoints and metrics are saved")
    parser.add_argument("--model", help="saved model checkpoint (.npz) for viewing or testing")
    parser.print_help()


def main() -> None:
    argv = sys.argv[1:]

    if argv in (["--help"], ["-h"]):
        show_help()
        return

    if not argv or argv[0] in AI_COMMANDS or "--mode" not in argv:
        from selfplay_ai import main as ai_main

        if not argv:
            sys.argv = ["main.py", "train"]
        elif argv[0] not in AI_COMMANDS:
            sys.argv = ["main.py", "train", *argv]
        else:
            sys.argv = ["main.py", *argv]
        ai_main()
        return

    parser = argparse.ArgumentParser(
        description="Graphical modes for the self-learning 2048 AI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        type=int,
        choices=range(1, 5),
        required=True,
        help="1: fast training view; 2: slower training view; 3: watch AI; 4: step through AI moves",
    )
    parser.add_argument("--games", type=int, default=100000, help="training games for modes 1 and 2")
    parser.add_argument("--depth", type=int, default=1, help="AI search depth; higher is stronger but slower")
    parser.add_argument("--seed", type=int, default=20261028, help="random seed for repeatable runs")
    parser.add_argument("--out", default="models/selfplay", help="folder for saved training checkpoints")
    parser.add_argument("--model", default=None, help="checkpoint .npz to load in modes 3 and 4")
    args = parser.parse_args()

    from selfplay_ui import main as ui_main

    sys.argv = [
        "main.py", "--mode", str(args.mode), "--games", str(args.games),
        "--depth", str(args.depth), "--seed", str(args.seed), "--out", args.out,
    ]
    if args.model:
        sys.argv.extend(["--model", args.model])
    ui_main()


if __name__ == "__main__":
    main()
