from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models" / "selfplay"
BEST_MODEL = MODEL_DIR / "best.npz"
LATEST_MODEL = MODEL_DIR / "latest.npz"


def ask(prompt: str, default: str) -> str:
    value = input(f"{prompt} [{default}]: ").strip()
    return value or default


def ask_int(prompt: str, default: int, minimum: int = 1) -> int:
    while True:
        value = ask(prompt, str(default))
        try:
            number = int(value)
            if number >= minimum:
                return number
        except ValueError:
            pass
        print(f"Enter a whole number of at least {minimum}.")


def choose_model() -> Path | None:
    if BEST_MODEL.is_file():
        return BEST_MODEL
    if LATEST_MODEL.is_file():
        return LATEST_MODEL
    print("\nNo trained model was found in models/selfplay.")
    print("Train the AI first, then come back here.")
    return None


def run_command(*args: str) -> None:
    print()
    try:
        subprocess.run([sys.executable, *args], cwd=ROOT, check=False)
    except KeyboardInterrupt:
        print("\nStopped.")
    input("\nPress Enter to return to the menu...")


def train(resume: bool) -> None:
    games = ask_int("Number of training games", 100000)
    depth = ask_int("Training search depth (1 is fastest)", 1)
    command = ["main.py", "train", "--games", str(games), "--depth", str(depth)]

    if resume:
        checkpoint = LATEST_MODEL if LATEST_MODEL.is_file() else BEST_MODEL
        if not checkpoint.is_file():
            print("\nNo saved checkpoint was found. Choose Train from scratch first.")
            input("\nPress Enter to return to the menu...")
            return
        command.extend(["--resume", str(checkpoint.relative_to(ROOT)), "--resume-metrics"])
    elif MODEL_DIR.exists() and any(MODEL_DIR.glob("*.npz")):
        print("\nA saved model already exists. Training from scratch will replace the")
        print("checkpoints and training metrics in models/selfplay.")
        confirm = input("Continue? Type YES to confirm: ").strip()
        if confirm != "YES":
            return

    run_command(*command)


def watch_ai(larp: bool) -> None:
    model = choose_model()
    if model is None:
        input("\nPress Enter to return to the menu...")
        return
    depth = ask_int("AI search depth", 2)
    mode = "4" if larp else "3"
    run_command("main.py", "--mode", mode, "--model", str(model.relative_to(ROOT)), "--depth", str(depth))


def live_training() -> None:
    mode = ask("Animation speed: 1 = fast, 2 = slow", "1")
    if mode not in {"1", "2"}:
        print("Choose 1 or 2.")
        input("\nPress Enter to return to the menu...")
        return
    games = ask_int("Number of training games", 10000)
    run_command("main.py", "--mode", mode, "--games", str(games))


def evaluate() -> None:
    model = choose_model()
    if model is None:
        input("\nPress Enter to return to the menu...")
        return
    games = ask_int("Number of evaluation games", 100)
    depth = ask_int("Search depth", 2)
    run_command("main.py", "evaluate", "--model", str(model.relative_to(ROOT)),
                "--games", str(games), "--depth", str(depth))


def benchmark() -> None:
    model = choose_model()
    if model is None:
        input("\nPress Enter to return to the menu...")
        return
    games = ask_int("Games per depth", 10)
    depths = ask("Search depths to compare (separate with spaces)", "1 2")
    parts = depths.split()
    try:
        parsed = [int(value) for value in parts]
        if not parsed or min(parsed) < 1:
            raise ValueError
    except ValueError:
        print("Enter one or more positive whole-number depths, such as: 1 2 3")
        input("\nPress Enter to return to the menu...")
        return
    run_command("main.py", "benchmark", "--model", str(model.relative_to(ROOT)),
                "--games", str(games), "--depths", *map(str, parsed))


def main() -> None:
    if not (ROOT / "main.py").is_file():
        raise SystemExit("Could not find main.py. Run this launcher from the project folder.")

    while True:
        print("\n" + "=" * 42)
        print("          SELF-LEARNING 2048")
        print("=" * 42)
        print("1. Train a new AI")
        print("2. Continue training")
        print("3. Watch the AI play")
        print("4. LARP mode")
        print("5. Watch live training")
        print("6. Evaluate the current model")
        print("7. Compare search depths")
        print("8. Run tests")
        print("0. Exit")
        choice = input("\nChoose an option: ").strip()

        if choice == "1":
            train(False)
        elif choice == "2":
            train(True)
        elif choice == "3":
            watch_ai(False)
        elif choice == "4":
            watch_ai(True)
        elif choice == "5":
            live_training()
        elif choice == "6":
            evaluate()
        elif choice == "7":
            benchmark()
        elif choice == "8":
            run_command("-m", "pytest", "-q")
        elif choice == "0":
            print("Goodbye.")
            break
        else:
            print("Choose a number from 0 to 8.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nGoodbye.")
