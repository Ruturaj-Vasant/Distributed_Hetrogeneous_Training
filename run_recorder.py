"""
run_recorder.py  —  Per-run metrics logger

Creates:
    runs/<model>__<dataset>__<optimizer>__topk<K>__<timestamp>/
        config.json    hyperparameters
        metrics.csv    step,loss  and  epoch,val_acc rows (interleaved)
        summary.json   final scalars
        loss.png       loss curve  (requires matplotlib)
"""

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Optional


class RunRecorder:
    def __init__(
        self,
        model_name:   str,
        dataset:      str,
        optimizer:    str,
        topk_k:       int,
        epochs:       int,
        lr:           float,
        weight_decay: float,
        runs_root:    str = "runs",
    ) -> None:
        topk_tag = f"topk{topk_k}" if topk_k > 0 else "topkfull"
        ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"{model_name}__{dataset}__{optimizer}__{topk_tag}__{ts}"

        self.run_dir = Path(runs_root) / run_name
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self._config = {
            "model":        model_name,
            "dataset":      dataset,
            "optimizer":    optimizer,
            "topk_k":       topk_k,
            "epochs":       epochs,
            "lr":           lr,
            "weight_decay": weight_decay,
            "run_name":     run_name,
            "started_at":   ts,
        }
        (self.run_dir / "config.json").write_text(
            json.dumps(self._config, indent=2)
        )

        self._csv_path = self.run_dir / "metrics.csv"
        self._csv_file = open(self._csv_path, "w", newline="")
        self._writer   = csv.writer(self._csv_file)
        self._writer.writerow(["step", "loss", "round_ms", "epoch", "val_acc"])
        self._csv_file.flush()

        self._steps:  list[int]            = []
        self._losses: list[float]          = []
        self._epoch_accs: list[tuple[int, float]] = []
        self._last_loss:    float          = 0.0
        self._last_val_acc: Optional[float] = None

    # ── Public API ────────────────────────────────────────────────────────────

    def log_step(self, step: int, loss: float, round_ms: float = 0.0) -> None:
        self._steps.append(step)
        self._losses.append(loss)
        self._last_loss = loss
        self._writer.writerow([step, f"{loss:.6f}", f"{round_ms:.1f}", "", ""])
        self._csv_file.flush()

    def log_epoch(self, epoch: int, val_acc: float) -> None:
        self._epoch_accs.append((epoch, val_acc))
        self._last_val_acc = val_acc
        self._writer.writerow(["", "", "", epoch, f"{val_acc:.4f}"])
        self._csv_file.flush()

    def close(self) -> Path:
        """Flush CSV, write summary.json, save loss.png. Returns run_dir."""
        self._csv_file.close()
        summary = {
            **self._config,
            "final_loss":    self._last_loss,
            "final_val_acc": self._last_val_acc,
            "total_steps":   len(self._steps),
            "epoch_accs":    self._epoch_accs,
        }
        (self.run_dir / "summary.json").write_text(
            json.dumps(summary, indent=2)
        )
        self._save_loss_plot()
        return self.run_dir

    # ── Private ───────────────────────────────────────────────────────────────

    def _save_loss_plot(self) -> None:
        if not self._steps:
            return
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            return

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(self._steps, self._losses, linewidth=1.0, color="#2563eb")
        ax.set_xlabel("Step")
        ax.set_ylabel("Loss")
        ax.set_title(self._config["run_name"])
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(self.run_dir / "loss.png", dpi=120)
        plt.close(fig)
