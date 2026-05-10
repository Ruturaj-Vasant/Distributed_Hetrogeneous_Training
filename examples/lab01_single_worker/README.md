# Lab 01: Single-Worker Baseline

## Overview

This lab runs the solo baseline experiment on one machine. The leader and worker run in two terminals on the same Mac, but the worker still connects through the Tailscale DNS name `leader-macbook-pro.taila5426e.ts.net` so the command path matches the rest of the labs.

Use this run as the baseline for loss curve, validation accuracy, and run artifact layout before adding distributed workers.

## Prerequisites

- Tailscale is installed and authenticated on the leader Mac.
- The project is cloned on the leader Mac.
- The package is installed from the repo root:

```bash
pip install -e ".[dev]"
```

- Tiny ImageNet-200 can be downloaded to the default cache, or pass `--dataset cifar10` for a smaller smoke test.

## Commands

Terminal 1, start the leader:

```bash
cd /Users/ruturaj_vasant/Desktop/Academic/AI_New/distributed-resnet
dtrain-leader --model resnet18 --epochs 5 --lr 0.01 --batch-size 32 --topk 0
```

The dashboard is available at:

```text
http://localhost:8080
```

Terminal 2, start one worker on the same machine:

```bash
cd /Users/ruturaj_vasant/Desktop/Academic/AI_New/distributed-resnet
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

- The worker registers, moves from pending/admitted state, and receives a shard.
- The loss curve should start updating in the dashboard and in `runs/<run_name>/loss.png`.
- Validation accuracy is logged after epochs.
- Run artifacts are written under `runs/`:

```text
runs/<model>__<dataset>__<optimizer>__topkfull__<timestamp>/
  config.json
  metrics.csv
  summary.json
  loss.png
```

## How to compare results

List saved runs:

```bash
dtrain-compare --list
```

Use the run directory from this lab as `EXP1 Solo` when comparing against the two-worker and TopK labs:

```bash
dtrain-compare \
  --runs runs/<lab01_run> runs/<lab02_run> runs/<lab03_run> \
  --labels "EXP1 Solo" "EXP2 PS Full" "EXP3 PS TopK" \
  --output comparison.png
```
