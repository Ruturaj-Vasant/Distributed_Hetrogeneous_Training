# Lab 02: Two-Worker Parameter Server With Full Gradients

## Overview

This lab runs the two-machine parameter-server experiment with full gradients by setting `--topk 0`. The leader runs on the Mac, one worker can run on the leader Mac, and the second worker runs on another machine over Tailscale.

This measures the distributed parameter-server path without gradient compression. Use it to compare shard split, gradient round time, and speedup against Lab 01.

## Prerequisites

- Tailscale is installed and authenticated on both machines.
- Both machines can resolve and reach `leader-macbook-pro.taila5426e.ts.net`.
- The project is cloned and installed on both machines:

```bash
pip install -e ".[dev]"
```

- Proto stubs are generated after any `.proto` change:

```bash
bash scripts/generate_proto.sh
```

## Commands

Terminal 1 on the leader Mac:

```bash
cd /Users/ruturaj_vasant/Desktop/Academic/AI_New/distributed-resnet
dtrain-leader --model resnet18 --epochs 5 --lr 0.01 --batch-size 32 --topk 0
```

The dashboard is available at:

```text
http://localhost:8080
```

Terminal 2 on the leader Mac, optional local worker:

```bash
cd /Users/ruturaj_vasant/Desktop/Academic/AI_New/distributed-resnet
dtrain-worker --leader leader-macbook-pro.taila5426e.ts.net
```

Terminal 3 on the second machine:

```bash
cd ~/distributed-resnet
dtrain-worker --leader leader-macbook-pro.taila5426e.ts.net
```

Back in the leader terminal, begin training:

```text
start
```

Optional terminal, watch the cluster:

```bash
dtrain-watch --leader leader-macbook-pro.taila5426e.ts.net
```

## What to watch for

- The leader assigns shards proportionally to worker benchmark score:

```text
worker_fraction = worker_score / sum(all_worker_scores)
worker_samples = int(worker_fraction * train_samples)
```

- Faster machines should receive a larger shard and often a larger local batch size.
- `metrics.csv` records `round_ms`, which is the synchronized gradient round duration.
- With full gradients, network transfer can dominate `round_ms`, so speedup may be less than linear.
- Expected speedup should be measured against Lab 01 using throughput and duration, not only raw worker count.

## How to compare results

List saved runs:

```bash
dtrain-compare --list
```

Compare Lab 02 against the solo baseline:

```bash
dtrain-compare \
  --runs runs/<lab01_run> runs/<lab02_run> \
  --labels "EXP1 Solo" "EXP2 PS Full" \
  --output comparison.png
```

Check `runs/<lab02_run>/summary.json` for `samples_per_second`, `duration_seconds`, `world_size`, and `compression_ratio`. Full gradients should report a compression ratio of `1.0`.
