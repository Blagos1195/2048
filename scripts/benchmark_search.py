from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the main.py benchmark on fixed seeds and write a CSV report."
    )
    parser.add_argument("--model", default="models/selfplay/best.npz")
    parser.add_argument("--games", type=int, default=1)
    parser.add_argument("--depths", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--seed", type=int, default=4_200_000)
    parser.add_argument("--output", default="artifacts/benchmark.csv")
    args = parser.parse_args()

    model = Path(args.model).resolve()
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(REPOSITORY_ROOT / "main.py"),
        "benchmark",
        "--model",
        str(model),
        "--games",
        str(args.games),
        "--depths",
        *(str(depth) for depth in args.depths),
        "--seed",
        str(args.seed),
        "--output",
        str(output),
    ]
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
