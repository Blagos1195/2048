"""Main entry point for the self-learning 2048 AI.

Training, evaluation and benchmarking all run through this file:
  python main.py                         # train with defaults
  python main.py train --games 10000
  python main.py evaluate --model models/selfplay/best.npz --games 100
  python main.py benchmark --model models/selfplay/best.npz --depths 1 2
  python main.py --mode 1 --games 10000  # live four-board training view
  python main.py --mode 3                 # watch the learned AI play
  python main.py --mode 4                 # LARP mode: keypresses trigger AI moves
"""
from __future__ import annotations

import argparse
import sys

AI_COMMANDS = {"train", "evaluate", "benchmark"}


def main() -> None:
    argv = sys.argv[1:]

    # Keep one public command entry point. The implementation stays separated
    # into modules, but users never need to invoke those modules directly.
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

    parser = argparse.ArgumentParser(description="Self-learning 2048 AI")
    parser.add_argument("--mode", type=int, choices=range(1, 5), required=True,
                        help="1=fast training view, 2=slow training view, "
                             "3=AI showcase, 4=LARP mode")
    parser.add_argument("--games", type=int, default=100000,
                        help="training episodes for modes 1 and 2")
    parser.add_argument("--depth", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20261028)
    parser.add_argument("--out", default="models/selfplay")
    parser.add_argument("--model", default=None)
    args = parser.parse_args()

    from selfplay_ui import main as ui_main

    sys.argv = ["main.py", "--mode", str(args.mode),
                "--games", str(args.games), "--depth", str(args.depth),
                "--seed", str(args.seed), "--out", args.out]
    if args.model:
        sys.argv.extend(["--model", args.model])
    ui_main()


if __name__ == "__main__":
    main()
