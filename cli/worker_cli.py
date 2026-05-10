"""cli/worker_cli — Argument parsing and entry point for the worker."""
import argparse
import asyncio
import sys

from trainer.worker.runner import run


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="dtrain-worker — Distributed ResNet Worker")
    p.add_argument(
        "--leader",
        default="leader-macbook-pro.taila5426e.ts.net",
        help="Leader hostname or Tailscale magic DNS",
    )
    p.add_argument("--port",    type=int, default=50051)
    p.add_argument(
        "--dataset", default="tinyimagenet", choices=["tinyimagenet", "cifar10"],
        help="Dataset to pre-download before connecting",
    )
    p.add_argument(
        "--preload-datasets", default=None, dest="preload_datasets",
        help="Comma-separated datasets to pre-download (tinyimagenet,cifar10,all)",
    )
    p.add_argument(
        "--cache-dir", default=None, dest="cache_dir",
        help="Override dataset cache directory",
    )
    p.add_argument(
        "--dry-run", action="store_true", dest="dry_run",
        help="Use synthetic tensors — skip dataset download (protocol testing)",
    )
    p.add_argument(
        "--connect-timeout", type=float, default=20.0, dest="connect_timeout",
        help="Seconds to wait for gRPC channel readiness per attempt",
    )
    p.add_argument(
        "--rpc-timeout", type=float, default=30.0, dest="rpc_timeout",
        help="Seconds to wait for registration RPC",
    )
    p.add_argument(
        "--connect-retries", type=int, default=6, dest="connect_retries",
        help="Connection/registration attempts before failing",
    )
    p.add_argument(
        "--retry-backoff", type=float, default=2.0, dest="retry_backoff",
        help="Base seconds between retries (multiplied by attempt number)",
    )
    p.add_argument(
        "--max-retry-backoff", type=float, default=15.0, dest="max_retry_backoff",
        help="Maximum seconds to sleep between retries",
    )
    return p


def cli_main() -> None:
    cfg = _build_parser().parse_args()
    try:
        asyncio.run(run(cfg))
    except KeyboardInterrupt:
        sys.exit(0)
    except (RuntimeError, Exception) as exc:
        import logging
        logging.getLogger("worker").error(str(exc))
        sys.exit(1)
