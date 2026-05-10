"""cli/leader_cli — Argument parsing and entry point for the leader (parameter server)."""
import argparse
import asyncio

from trainer.leader.server import (
    main,
    _DEFAULT_HEARTBEAT_TIMEOUT,
    _DEFAULT_HEARTBEAT_CHECK_INTERVAL,
    _DEFAULT_GRAD_SYNC_TIMEOUT,
)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="dtrain-leader — Distributed ResNet Parameter Server")
    p.add_argument("--port",         type=int,   default=50051)
    p.add_argument("--min-workers",  type=int,   default=1,    dest="min_workers",
                   help="Minimum workers before auto-start triggers")
    p.add_argument("--auto-start",   action="store_true",      dest="auto_start",
                   help="Start training automatically when min-workers threshold met")
    p.add_argument("--dataset",      default="tinyimagenet",
                   choices=["tinyimagenet", "cifar10"])
    p.add_argument("--epochs",       type=int,   default=90)
    p.add_argument("--train-samples", type=int,  default=None, dest="train_samples",
                   help="Cap training samples per epoch (default: full dataset)")
    p.add_argument("--lr",           type=float, default=0.1)
    p.add_argument("--weight-decay", type=float, default=1e-4, dest="weight_decay")
    p.add_argument("--batch-size",   type=int,   default=32,   dest="batch_size",
                   help="Base batch size per worker (scales with score, default 32)")
    p.add_argument("--topk",         type=int,   default=50_000,
                   help="Top-K gradient elements per layer (0 = full gradients)")
    p.add_argument("--grad-accum",   type=int,   default=1,    dest="grad_accum",
                   help="Gradient accumulation steps — sync every N batches")
    p.add_argument("--model",        default="resnet101",      dest="model_name",
                   choices=["resnet18", "resnet50", "resnet101"])
    p.add_argument("--heartbeat-timeout", type=float,
                   default=_DEFAULT_HEARTBEAT_TIMEOUT,         dest="heartbeat_timeout",
                   help="Seconds of silence before a worker is marked dead")
    p.add_argument("--heartbeat-check-interval", type=float,
                   default=_DEFAULT_HEARTBEAT_CHECK_INTERVAL,  dest="heartbeat_check_interval",
                   help="How often to check for dead workers (seconds)")
    p.add_argument("--grad-sync-timeout", type=float,
                   default=_DEFAULT_GRAD_SYNC_TIMEOUT,         dest="grad_sync_timeout",
                   help="Max seconds to wait for all workers per gradient round")
    p.add_argument("--data-root",    default=None,             dest="data_root",
                   help="Path to dataset cache (default: ~/.cache/tiny-imagenet-200)")
    p.add_argument("--runs-root",    default="runs",           dest="runs_root",
                   help="Directory for run artifacts (default: runs/)")
    p.add_argument("--dashboard-port", type=int, default=8080, dest="dashboard_port",
                   help="Port for the web dashboard (default: 8080)")
    p.add_argument("--no-dashboard", action="store_true", dest="no_dashboard",
                   help="Disable the web dashboard (useful for headless/CI runs)")
    return p


def cli_main() -> None:
    cfg = _build_parser().parse_args()
    asyncio.run(main(cfg))
