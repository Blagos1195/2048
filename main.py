"""Single entry point for training, evaluating, benchmarking, and viewing the 2048 AI."""
from __future__ import annotations

import argparse
import sys

AI_COMMANDS = {"train", "evaluate", "benchmark"}


def show_help() -> None:
    print("""2048 SELF-LEARNING AI
=====================

Choose what you want to do.

1. TEACH THE AI TO PLAY
   Start training with the usual settings:
     python main.py

   Choose how many games to train for:
     python main.py train --games 100000

   Training makes the AI learn from games it plays. It saves its progress
   in the models/selfplay folder.

2. TEST A MODEL YOU HAVE ALREADY TRAINED
     python main.py evaluate --model models/selfplay/best.npz

   This plays test games and reports the scores and largest tiles reached.
   Use this to check how well your saved AI performs.

3. COMPARE PLAYING STRENGTH AND SPEED
     python main.py benchmark --model models/selfplay/best.npz --depths 1 2

   This tests the same model at different search depths.
   Depth 1 is much faster. Depth 2 can make better decisions, but takes
   considerably longer. Try depth 1 first if you are unsure.

4. WATCH THE AI IN A WINDOW
   These modes open a visual version of the game:

     python main.py --mode 1
       Watch four games while the AI trains. Faster animation.

     python main.py --mode 2
       Watch four games while the AI trains. Slower animation.

     python main.py --mode 3 --model models/selfplay/best.npz
       Watch a saved model play by itself.

     python main.py --mode 4 --model models/selfplay/best.npz
       Step through a saved model one move at a time. Press R to restart,
       or Esc/Q to quit.

   Modes 1 and 2 train the AI. Modes 3 and 4 need a model file, usually
   best.npz, created by training.

USEFUL THINGS TO KNOW
---------------------
--games
   How many games to play. More training games take longer.

--depth
   How far ahead the AI looks before choosing a move. Higher is slower,
   and does not guarantee a better result every single game.

--model
   The saved AI file to test or watch. Use the path shown in the examples.

--out
   Where training saves its files. You usually do not need to change this.

--seed
   Makes runs more repeatable for comparisons. You can ignore this at first.

GET MORE HELP
-------------
See all training settings:
  python main.py train --help

See all testing settings:
  python main.py evaluate --help

See all comparison settings:
  python main.py benchmark --help

See graphical settings:
  python main.py --mode 1 --help

NEW TO THIS? Start with:
  python main.py

That starts training with the default settings. You can explore the other
commands once you have a saved model.
""")


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
