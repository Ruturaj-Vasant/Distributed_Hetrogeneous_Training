"""
run_experiments.py  —  Experiment comparison tool

After completing each experiment manually, point this script at the run
directories to generate a combined comparison plot and summary table.

Usage:
    # Compare 3 runs and write comparison.png
    python3 run_experiments.py \\
        --runs runs/exp1_dir runs/exp2_dir runs/exp3_dir \\
        --labels "EXP1 Solo" "EXP2 PS Full" "EXP3 PS TopK" \\
        --output comparison.png

    # List all run dirs under runs/
    python3 run_experiments.py --list

Experiment setup reference
──────────────────────────
EXP1  Solo baseline
    Terminal 1 (leader machine):   python3 leader.py --model resnet18 --epochs 5
    Terminal 2 (leader machine):   python3 worker.py
    Measures: time/epoch, val-acc with no network overhead.

EXP2  Parameter server — full gradients
    Terminal 1 (leader machine):   python3 leader.py --model resnet18 --epochs 5 --topk 0
    Terminal 2 (leader machine):   python3 worker.py
    Terminal 3 (remote machine):   python3 worker.py --leader <leader-tailscale-ip>
    Measures: speedup vs EXP1, communication cost with full gradient payload.

EXP3  Parameter server — TopK sparsification
    Same topology as EXP2 but add --topk 50000 on the leader.
    Measures: gradient compression ratio, accuracy delta vs EXP2.
"""

import argparse
import json
import sys
from pathlib import Path


def _load_run(run_dir: Path) -> dict:
    """Load summary.json and metrics.csv from a run directory."""
    summary_path = run_dir / "summary.json"
    metrics_path = run_dir / "metrics.csv"

    if not summary_path.exists():
        print(f"  [warn] No summary.json in {run_dir} — skipping", file=sys.stderr)
        return {}

    with open(summary_path) as f:
        summary = json.load(f)

    steps, losses, epochs, val_accs = [], [], [], []
    if metrics_path.exists():
        import csv
        with open(metrics_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row["step"]:
                    steps.append(int(row["step"]))
                    losses.append(float(row["loss"]))
                if row["epoch"]:
                    epochs.append(int(row["epoch"]))
                    val_accs.append(float(row["val_acc"]))

    return {
        "summary":   summary,
        "steps":     steps,
        "losses":    losses,
        "epochs":    epochs,
        "val_accs":  val_accs,
    }


def _list_runs(runs_root: Path) -> None:
    runs = sorted(runs_root.iterdir()) if runs_root.exists() else []
    if not runs:
        print(f"No runs found under {runs_root}/")
        return
    print(f"\nRuns in {runs_root}/\n")
    hdr = f"  {'Name':<52} {'Workers':>7} {'Duration':>9} {'Samp/s':>8} {'Loss':>8} {'ValAcc':>8} {'Compress':>9}"
    print(hdr)
    print("  " + "─" * (len(hdr) - 2))
    for r in sorted(runs):
        if not r.is_dir():
            continue
        summary_path = r / "summary.json"
        if not summary_path.exists():
            print(f"  {r.name}  (no summary.json)")
            continue
        with open(summary_path) as f:
            s = json.load(f)
        acc      = s.get("final_val_acc")
        acc_s    = f"{acc:.4f}" if acc is not None else "n/a"
        loss     = s.get("final_loss", 0.0)
        workers  = s.get("world_size", "?")
        dur      = s.get("duration_seconds", 0.0)
        dur_s    = f"{dur:.0f}s" if dur else "n/a"
        sps      = s.get("samples_per_second", 0.0)
        sps_s    = f"{sps:.1f}" if sps else "n/a"
        cr       = s.get("compression_ratio", 1.0)
        cr_s     = f"{cr:.4f}" if cr != 1.0 else "1.0 (full)"
        print(
            f"  {r.name:<52} {str(workers):>7} {dur_s:>9} {sps_s:>8}"
            f" {loss:>8.4f} {acc_s:>8} {cr_s:>9}"
        )
    print()


def _compare(runs: list[Path], labels: list[str], output: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed — run: pip install matplotlib")
        sys.exit(1)

    COLORS = ["#2563eb", "#16a34a", "#dc2626", "#9333ea"]

    data = [_load_run(r) for r in runs]
    valid = [(d, lbl) for d, lbl in zip(data, labels) if d]

    if not valid:
        print("No valid run data found.")
        sys.exit(1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: loss curves
    ax_loss = axes[0]
    for i, (d, lbl) in enumerate(valid):
        if d["steps"]:
            ax_loss.plot(
                d["steps"], d["losses"],
                label=lbl, color=COLORS[i % len(COLORS)], linewidth=1.2,
            )
    ax_loss.set_xlabel("Step")
    ax_loss.set_ylabel("Training Loss")
    ax_loss.set_title("Training Loss")
    ax_loss.legend()
    ax_loss.grid(True, alpha=0.3)

    # Right: val accuracy per epoch
    ax_acc = axes[1]
    for i, (d, lbl) in enumerate(valid):
        if d["epochs"]:
            ax_acc.plot(
                d["epochs"], d["val_accs"],
                marker="o", label=lbl, color=COLORS[i % len(COLORS)], linewidth=1.5,
            )
    ax_acc.set_xlabel("Epoch")
    ax_acc.set_ylabel("Val Accuracy (top-1)")
    ax_acc.set_title("Validation Accuracy")
    ax_acc.legend()
    ax_acc.grid(True, alpha=0.3)

    fig.suptitle("Experiment Comparison", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    print(f"Saved {output}")

    # ── Summary table with speedup / efficiency ───────────────────────────────
    # Use the solo run (world_size == 1) as baseline; fall back to first run.
    baseline_dur = None
    for d, _ in valid:
        if d["summary"].get("world_size", 1) == 1:
            baseline_dur = d["summary"].get("duration_seconds")
            break
    if baseline_dur is None:
        baseline_dur = valid[0][0]["summary"].get("duration_seconds")

    print()
    print(
        f"{'Label':<22} {'Model':<10} {'Workers':>7} {'TopK':>8} "
        f"{'Steps':>7} {'Duration':>9} {'Samp/s':>8} "
        f"{'Loss':>8} {'ValAcc':>8} {'Compress':>9} {'Speedup':>8} {'Efficiency':>10}"
    )
    print("─" * 120)
    for d, lbl in valid:
        s        = d["summary"]
        model    = s.get("model", "?")
        workers  = s.get("world_size", 1)
        topk     = str(s.get("topk_k", "n/a"))
        steps    = s.get("total_steps", len(d["steps"]))
        dur      = s.get("duration_seconds", 0.0)
        dur_s    = f"{dur:.0f}s" if dur else "n/a"
        sps      = s.get("samples_per_second", 0.0)
        sps_s    = f"{sps:.1f}" if sps else "n/a"
        loss     = s.get("final_loss", 0.0)
        acc      = s.get("final_val_acc")
        acc_s    = f"{acc:.4f}" if acc is not None else "n/a"
        cr       = s.get("compression_ratio", 1.0)
        cr_s     = f"{cr:.4f}"
        # speedup = baseline_duration / this_duration
        if baseline_dur and dur and dur > 0:
            speedup    = baseline_dur / dur
            efficiency = speedup / max(1, workers)
            spd_s      = f"{speedup:.2f}x"
            eff_s      = f"{efficiency:.2f}"
        else:
            spd_s = eff_s = "n/a"
        print(
            f"{lbl:<22} {model:<10} {str(workers):>7} {topk:>8} "
            f"{steps:>7} {dur_s:>9} {sps_s:>8} "
            f"{loss:>8.4f} {acc_s:>8} {cr_s:>9} {spd_s:>8} {eff_s:>10}"
        )
    print()


def main() -> None:
    p = argparse.ArgumentParser(description="Distributed ResNet experiment comparison tool")
    p.add_argument("--runs",   nargs="+", metavar="DIR",
                   help="Run directories to compare")
    p.add_argument("--labels", nargs="+", metavar="LABEL",
                   help="Legend labels (one per --runs dir, default: dir name)")
    p.add_argument("--output", default="comparison.png",
                   help="Output plot path (default: comparison.png)")
    p.add_argument("--list",   action="store_true",
                   help="List all run directories under runs/")
    p.add_argument("--runs-root", default="runs", dest="runs_root",
                   help="Root directory for runs (default: runs/)")
    cfg = p.parse_args()

    runs_root = Path(cfg.runs_root)

    if cfg.list:
        _list_runs(runs_root)
        return

    if not cfg.runs:
        p.print_help()
        print("\nHint: use --list to see available runs.")
        sys.exit(0)

    runs = [Path(r) for r in cfg.runs]
    for r in runs:
        if not r.exists():
            print(f"Run directory not found: {r}")
            sys.exit(1)

    labels = cfg.labels or [r.name for r in runs]
    if len(labels) < len(runs):
        labels += [r.name for r in runs[len(labels):]]

    _compare(runs, labels, Path(cfg.output))


if __name__ == "__main__":
    main()
