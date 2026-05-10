"""trainer/utils/recorder — Per-run metrics logger."""

from __future__ import annotations
import csv
import json
import time
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
        batch_size:   int = 32,
        world_size:   int = 1,
        dataset_samples:           int = 0,
        raw_gradient_numel:        int = 0,
        compressed_gradient_numel: int = 0,
        runs_root:    str = "runs",
    ) -> None:
        topk_tag = f"topk{topk_k}" if topk_k > 0 else "topkfull"
        ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"{model_name}__{dataset}__{optimizer}__{topk_tag}__{ts}"

        self.run_dir = Path(runs_root) / run_name
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self._batch_size    = batch_size
        self._world_size    = world_size
        self._dataset_samples          = dataset_samples
        self._raw_gradient_numel       = raw_gradient_numel
        self._compressed_gradient_numel = compressed_gradient_numel or raw_gradient_numel
        self._epochs        = epochs
        self._start_time    = time.monotonic()

        self._config = {
            "model":        model_name,
            "dataset":      dataset,
            "optimizer":    optimizer,
            "topk_k":       topk_k,
            "epochs":       epochs,
            "lr":           lr,
            "weight_decay": weight_decay,
            "batch_size":   batch_size,
            "world_size":   world_size,
            "parallelism":  "solo" if world_size == 1 else "parameter_server",
            "dataset_samples": dataset_samples,
            "raw_gradient_numel":        raw_gradient_numel,
            "compressed_gradient_numel": self._compressed_gradient_numel,
            "run_name":   run_name,
            "started_at": ts,
        }
        (self.run_dir / "config.json").write_text(json.dumps(self._config, indent=2))

        self._csv_path = self.run_dir / "metrics.csv"
        self._csv_file = open(self._csv_path, "w", newline="")
        self._writer   = csv.writer(self._csv_file)
        self._writer.writerow(["step", "loss", "round_ms", "straggler_delay_s", "epoch", "val_acc"])
        self._csv_file.flush()

        self._steps:    list[int]   = []
        self._losses:   list[float] = []
        self._round_ms_list:    list[float] = []
        self._straggler_delays: list[float] = []
        self._epoch_accs: list[tuple[int, float]] = []
        self._last_loss:    float          = 0.0
        self._last_val_acc: Optional[float] = None

    def log_step(
        self,
        step:              int,
        loss:              float,
        round_ms:          float = 0.0,
        straggler_delay_s: float = 0.0,
    ) -> None:
        self._steps.append(step)
        self._losses.append(loss)
        self._last_loss = loss
        if round_ms > 0:
            self._round_ms_list.append(round_ms)
        if straggler_delay_s > 0:
            self._straggler_delays.append(straggler_delay_s)
        self._writer.writerow(
            [step, f"{loss:.6f}", f"{round_ms:.1f}", f"{straggler_delay_s:.4f}", "", ""]
        )
        self._csv_file.flush()

    def log_epoch(self, epoch: int, val_acc: float) -> None:
        self._epoch_accs.append((epoch, val_acc))
        self._last_val_acc = val_acc
        self._writer.writerow(["", "", "", "", epoch, f"{val_acc:.4f}"])
        self._csv_file.flush()

    def close(self) -> Path:
        self._csv_file.close()

        duration_s   = time.monotonic() - self._start_time
        total_steps  = len(self._steps)
        avg_round_ms = (
            sum(self._round_ms_list) / len(self._round_ms_list)
            if self._round_ms_list else 0.0
        )
        batches_per_epoch = (
            self._dataset_samples // self._batch_size
            if self._batch_size > 0 else 0
        )
        samples_per_second = (
            (self._dataset_samples * self._epochs) / duration_s
            if duration_s > 0 and self._dataset_samples > 0 else 0.0
        )
        compression_ratio = (
            self._compressed_gradient_numel / self._raw_gradient_numel
            if self._raw_gradient_numel > 0 else 1.0
        )
        straggler_delay_max   = max(self._straggler_delays) if self._straggler_delays else 0.0
        straggler_delay_total = sum(self._straggler_delays)

        summary = {
            **self._config,
            "final_loss":    self._last_loss,
            "final_val_acc": self._last_val_acc,
            "total_steps":   total_steps,
            "epoch_accs":    self._epoch_accs,
            "duration_seconds":   round(duration_s, 2),
            "batches_per_epoch":  batches_per_epoch,
            "seconds_per_batch":  round(avg_round_ms / 1000.0, 4),
            "samples_per_second": round(samples_per_second, 2),
            "compression_ratio":  round(compression_ratio, 6),
            "straggler_delay_seconds":       round(straggler_delay_max, 4),
            "straggler_delay_total_seconds": round(straggler_delay_total, 4),
        }
        (self.run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
        self._save_loss_plot()
        return self.run_dir

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
