from __future__ import annotations

import os
import signal
import subprocess
import sys
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models" / "selfplay"
BEST_MODEL = MODEL_DIR / "best.npz"
LATEST_MODEL = MODEL_DIR / "latest.npz"


class Launcher:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.process = None
        self.hidden_for_game = False
        self.root.title("Self-learning 2048")
        self.root.geometry("560x590")
        self.root.minsize(480, 520)

        style = ttk.Style()
        if "clam" in style.theme_names():
            style.theme_use("clam")
        style.configure("Title.TLabel", font=("Segoe UI", 21, "bold"))
        style.configure("Subtitle.TLabel", font=("Segoe UI", 10))
        style.configure("Action.TButton", font=("Segoe UI", 11), padding=10)

        frame = ttk.Frame(root, padding=24)
        frame.pack(fill="both", expand=True)
        ttk.Label(frame, text="SELF-LEARNING 2048", style="Title.TLabel").pack(anchor="w")
        ttk.Label(
            frame,
            text="Choose an activity. You can change its settings before it starts.",
            style="Subtitle.TLabel",
        ).pack(anchor="w", pady=(4, 20))

        self.buttons = []
        actions = [
            ("Train a new AI", lambda: self.training(False)),
            ("Continue training", lambda: self.training(True)),
            ("Watch the AI play", lambda: self.watch(False)),
            ("LARP mode", lambda: self.watch(True)),
            ("Watch live training", self.live_training),
            ("Evaluate a model", self.evaluate),
            ("Compare search depths", self.benchmark),
            ("Run tests", self.tests),
        ]
        for label, command in actions:
            button = ttk.Button(frame, text=label, command=command, style="Action.TButton")
            button.pack(fill="x", pady=3)
            self.buttons.append(button)

        self.status = tk.StringVar(value="Ready")
        ttk.Separator(frame).pack(fill="x", pady=(18, 10))
        ttk.Label(frame, textvariable=self.status).pack(anchor="w")
        self.stop_button = ttk.Button(frame, text="Stop training safely", command=self.stop_training, state="disabled")
        self.stop_button.pack(anchor="w", pady=(8, 0))
        ttk.Label(
            frame,
            text="Training and test output appears in the terminal. Game windows open separately.",
            wraplength=490,
            style="Subtitle.TLabel",
        ).pack(anchor="w", pady=(8, 0))

        self.root.protocol("WM_DELETE_WINDOW", self.close)

    def model_choices(self):
        choices = []
        if BEST_MODEL.is_file():
            choices.append(("Best checkpoint", BEST_MODEL))
        if LATEST_MODEL.is_file():
            choices.append(("Latest checkpoint", LATEST_MODEL))
        return choices

    def model_dialog(self, title="Choose model"):
        choices = self.model_choices()
        if not choices:
            messagebox.showwarning("No saved model", "No checkpoint was found in models/selfplay. Train an AI first.", parent=self.root)
            return None

        window = self.dialog(title)
        selected = tk.StringVar(value=choices[0][0])
        ttk.Label(window, text="Which saved model should be used?").pack(anchor="w", pady=(0, 8))
        ttk.Combobox(window, textvariable=selected, values=[name for name, _ in choices], state="readonly").pack(fill="x")
        result = {"path": None}

        def start():
            result["path"] = next(path for name, path in choices if name == selected.get())
            window.destroy()

        self.dialog_buttons(window, start)
        self.root.wait_window(window)
        return result["path"]

    def dialog(self, title):
        window = tk.Toplevel(self.root)
        window.title(title)
        window.transient(self.root)
        window.resizable(False, False)
        frame = ttk.Frame(window, padding=18)
        frame.pack(fill="both", expand=True)
        window.frame = frame
        return window

    def field(self, window, label, variable, values=None):
        row = ttk.Frame(window.frame)
        row.pack(fill="x", pady=5)
        ttk.Label(row, text=label, width=25).pack(side="left")
        if values:
            widget = ttk.Combobox(row, textvariable=variable, values=values, state="readonly", width=20)
        else:
            widget = ttk.Entry(row, textvariable=variable, width=23)
        widget.pack(side="right")
        return widget

    def dialog_buttons(self, window, callback):
        row = ttk.Frame(window.frame)
        row.pack(fill="x", pady=(16, 0))
        ttk.Button(row, text="Cancel", command=window.destroy).pack(side="right")
        ttk.Button(row, text="Continue", command=callback).pack(side="right", padx=8)

    def confirm(self, title, lines, command, game=False):
        summary = "\n".join(lines)
        if not messagebox.askyesno(title, f"{summary}\n\nStart this task?", parent=self.root):
            return
        self.launch(command, game=game)

    def training(self, resume):
        if resume and not self.model_choices():
            messagebox.showwarning("No checkpoint", "There is no saved checkpoint to continue from. Train a new AI first.", parent=self.root)
            return

        window = self.dialog("Continue training" if resume else "Train a new AI")
        games = tk.StringVar(value="100000")
        depth = tk.StringVar(value="2")
        workers = tk.StringVar(value="Automatic (up to 4)")
        eval_every = tk.StringVar(value="10000")
        eval_games = tk.StringVar(value="20")
        eval_depth = tk.StringVar(value="2")
        checkpoint = tk.StringVar(value="Latest checkpoint")
        out = tk.StringVar(value="models/selfplay")

        ttk.Label(window.frame, text="Training settings", font=("Segoe UI", 13, "bold")).pack(anchor="w", pady=(0, 8))
        self.field(window, "Training games", games, ["1000", "10000", "50000", "100000", "250000", "500000"])
        self.field(window, "Workers", workers, ["Automatic (up to 4)", "1", "2", "3", "4"])
        self.field(window, "Evaluate every N games", eval_every, ["0", "1000", "5000", "10000", "25000"])
        self.field(window, "Games per evaluation", eval_games, ["5", "10", "20", "30", "50"])
        self.field(window, "Evaluation depth", eval_depth, ["1", "2", "3"])
        if resume:
            self.field(window, "Resume from", checkpoint, ["Latest checkpoint", "Best checkpoint"])
        self.field(window, "Save folder", out)

        def start():
            try:
                values = [int(games.get()), int(depth.get()), int(eval_every.get()), int(eval_games.get()), int(eval_depth.get())]
                if values[0] < 1 or min(values[1:]) < 0 or values[1] < 1 or values[4] < 1:
                    raise ValueError
                if not out.get().strip():
                    raise ValueError
            except ValueError:
                messagebox.showerror("Invalid settings", "Enter valid whole numbers and a save folder.", parent=window)
                return
            command = ["main.py", "train", "--games", str(values[0]), "--depth", str(values[1]),
                       "--workers", "0" if workers.get().startswith("Automatic") else workers.get(),
                       "--eval-every", str(values[2]), "--eval-games", str(values[3]),
                       "--eval-depth", str(values[4]), "--out", out.get().strip()]
            if resume:
                path = LATEST_MODEL if checkpoint.get() == "Latest checkpoint" else BEST_MODEL
                if not path.is_file():
                    messagebox.showerror("Missing checkpoint", f"{path.name} was not found.", parent=window)
                    return
                command.extend(["--resume", str(path.relative_to(ROOT)), "--resume-metrics"])
            elif any(MODEL_DIR.glob("*.npz")) if MODEL_DIR.exists() else False:
                if not messagebox.askyesno("Start from scratch", "Existing checkpoints may be overwritten. Continue?", parent=window):
                    return
            window.destroy()
            self.confirm(
                "Confirm training",
                [f"Mode: {'Continue' if resume else 'Start from scratch'}",
                 f"Games: {values[0]:,}", f"Training depth: {values[1]}",
                 f"Workers: {workers.get()}", f"Evaluation: every {values[2]:,} games, {values[3]} games at depth {values[4]}",
                 f"Save folder: {out.get()}"],
                command,
            )

        self.dialog_buttons(window, start)

    def watch(self, larp):
        model = self.model_dialog("Choose AI model")
        if model is None:
            return
        window = self.dialog("LARP mode" if larp else "Watch the AI")
        depth = tk.StringVar(value="2")
        self.field(window, "Search depth", depth, ["1", "2", "3", "4", "5"])
        def start():
            if depth.get() not in {"1", "2", "3", "4", "5"}:
                messagebox.showerror("Invalid depth", "Choose a search depth from 1 to 5.", parent=window)
                return
            window.destroy()
            mode = "4" if larp else "3"
            self.confirm("Launch game", [f"Mode: {'LARP' if larp else 'AI plays automatically'}",
                                         f"Model: {model.name}", f"Search depth: {depth.get()}"],
                         ["main.py", "--mode", mode, "--model", str(model.relative_to(ROOT)), "--depth", depth.get()],
                         game=True)
        self.dialog_buttons(window, start)

    def live_training(self):
        window = self.dialog("Watch live training")
        games = tk.StringVar(value="10000")
        speed = tk.StringVar(value="Fast")
        depth = tk.StringVar(value="2")
        self.field(window, "Training games", games, ["100", "1000", "5000", "10000", "50000"])
        self.field(window, "Animation speed", speed, ["Fast", "Slow"])
        def start():
            try:
                count = int(games.get())
                if count < 1:
                    raise ValueError
            except ValueError:
                messagebox.showerror("Invalid games", "Enter a positive number of games.", parent=window)
                return
            window.destroy()
            mode = "1" if speed.get() == "Fast" else "2"
            self.confirm("Confirm live training", [f"Games: {count:,}", f"Speed: {speed.get()}", f"Depth: {depth.get()}"],
                         ["main.py", "--mode", mode, "--games", str(count), "--depth", depth.get()], game=True)
        self.dialog_buttons(window, start)

    def evaluate(self):
        model = self.model_dialog("Choose model to evaluate")
        if model is None:
            return
        window = self.dialog("Evaluate model")
        games = tk.StringVar(value="100")
        depth = tk.StringVar(value="2")
        self.field(window, "Evaluation games", games, ["10", "20", "30", "50", "100", "250"])
        self.field(window, "Search depth", depth, ["1", "2", "3"])
        def start():
            try:
                count = int(games.get())
                if count < 1 or depth.get() not in {"1", "2", "3"}:
                    raise ValueError
            except ValueError:
                messagebox.showerror("Invalid settings", "Enter valid evaluation settings.", parent=window)
                return
            window.destroy()
            self.confirm("Confirm evaluation", [f"Model: {model.name}", f"Games: {count}", f"Search depth: {depth.get()}"],
                         ["main.py", "evaluate", "--model", str(model.relative_to(ROOT)), "--games", str(count), "--depth", depth.get()])
        self.dialog_buttons(window, start)

    def benchmark(self):
        model = self.model_dialog("Choose model to compare")
        if model is None:
            return
        window = self.dialog("Compare search depths")
        games = tk.StringVar(value="10")
        depths = tk.StringVar(value="1 2")
        self.field(window, "Games per depth", games, ["5", "10", "20", "30", "50"])
        self.field(window, "Depths (space-separated)", depths)
        def start():
            try:
                count = int(games.get())
                parsed = [int(part) for part in depths.get().split()]
                if count < 1 or not parsed or min(parsed) < 1 or len(set(parsed)) != len(parsed):
                    raise ValueError
            except ValueError:
                messagebox.showerror("Invalid settings", "Enter positive games and distinct positive search depths.", parent=window)
                return
            window.destroy()
            self.confirm("Confirm comparison", [f"Model: {model.name}", f"Games per depth: {count}", f"Depths: {', '.join(map(str, parsed))}"],
                         ["main.py", "benchmark", "--model", str(model.relative_to(ROOT)),
                          "--games", str(count), "--depths", *map(str, parsed)])
        self.dialog_buttons(window, start)

    def tests(self):
        self.confirm("Run tests", ["Run the complete pytest suite."], ["-m", "pytest", "-q"])

    def launch(self, command, game=False):
        if self.process and self.process.poll() is None:
            messagebox.showwarning("Task running", "Wait for the current task to finish first.", parent=self.root)
            return
        print("\n$ " + " ".join([sys.executable, *command]), flush=True)
        self.set_enabled(False)
        self.status.set("Running... check the terminal for output.")
        if game:
            self.hidden_for_game = True
            self.root.withdraw()
        try:
            process_options = {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP} if os.name == "nt" else {"start_new_session": True}
            self.process = subprocess.Popen([sys.executable, *command], cwd=ROOT, **process_options)
            self.training_process = len(command) > 1 and command[1] == "train"
            self.stop_button.configure(state="normal" if self.training_process else "disabled")
        except OSError as exc:
            self.process = None
            self.restore_after_task()
            messagebox.showerror("Could not start", str(exc), parent=self.root)
            return
        self.root.after(500, self.check_process)

    def check_process(self):
        if not self.process:
            return
        code = self.process.poll()
        if code is None:
            self.root.after(500, self.check_process)
            return
        self.process = None
        self.training_process = False
        self.stop_button.configure(state="disabled")
        self.restore_after_task()
        if code == 0:
            self.status.set("Finished successfully.")
        else:
            self.status.set(f"Finished with exit code {code}. See terminal output.")
            if not self.hidden_for_game:
                messagebox.showwarning("Task finished", f"The task exited with code {code}. Check the terminal for details.", parent=self.root)

    def stop_training(self):
        if not self.process or self.process.poll() is not None or not getattr(self, "training_process", False):
            return
        self.stop_button.configure(state="disabled")
        self.status.set("Stopping training safely and saving the latest checkpoint...")
        try:
            if os.name == "nt":
                self.process.send_signal(signal.CTRL_BREAK_EVENT)
            else:
                os.killpg(self.process.pid, signal.SIGINT)
        except ProcessLookupError:
            pass

    def restore_after_task(self):
        if self.hidden_for_game:
            self.root.deiconify()
            self.hidden_for_game = False
        self.set_enabled(True)

    def set_enabled(self, enabled):
        for button in self.buttons:
            button.configure(state="normal" if enabled else "disabled")

    def close(self):
        if self.process and self.process.poll() is None:
            if not messagebox.askyesno("Task still running", "A task is still running. Close the launcher and leave it running?", parent=self.root):
                return
        self.root.destroy()


def main():
    if not (ROOT / "main.py").is_file():
        raise SystemExit("Could not find main.py. Run this launcher from the project folder.")
    root = tk.Tk()
    Launcher(root)
    root.mainloop()


if __name__ == "__main__":
    main()
