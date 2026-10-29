"""Main entry point for the self-learning 2048 AI.

Examples:
  python main.py                         # train headlessly
  python main.py --mode 1 --games 10000   # live four-board training view
  python main.py --mode 2 --games 10000   # slower training view
  python main.py --mode 3                 # watch the learned AI play
  python main.py --mode 4                 # LARP mode: keypresses trigger AI moves
"""
import argparse
import sys


def main():
    parser = argparse.ArgumentParser(description="Self-learning 2048 AI")
    parser.add_argument("--mode", type=int, choices=range(5), default=0,
                        help="0=headless training, 1=fast training view, 2=slow training view, "
                             "3=AI showcase, 4=LARP mode")
    parser.add_argument("--games", type=int, default=100000)
    parser.add_argument("--depth", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20261028)
    parser.add_argument("--out", default="models/selfplay")
    parser.add_argument("--model", default=None)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--workers", type=int, default=0, help="parallel self-play workers; 0 auto-selects")
    args = parser.parse_args()

    if args.mode == 0:
        from selfplay_ai import main as training_main
        sys.argv = ["selfplay_ai.py", "train", "--games", str(args.games),
                    "--depth", str(args.depth), "--seed", str(args.seed),
                    "--out", args.out, "--workers", str(args.workers)]
        if args.resume:
            sys.argv.extend(["--resume", args.resume])
        training_main()
        return

    from selfplay_ui import main as ui_main
    sys.argv = ["selfplay_ui.py", "--mode", str(args.mode),
                "--games", str(args.games), "--depth", str(args.depth),
                "--seed", str(args.seed), "--out", args.out]
    if args.model:
        sys.argv.extend(["--model", args.model])
    ui_main()


if __name__ == "__main__":
    main()
