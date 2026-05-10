# Lab 03: Two-Worker Parameter Server With TopK Compression

## Overview

This lab uses the same two-machine topology as Lab 02, but enables TopK gradient compression with `--topk 5000`. Each worker keeps only the largest 5000 gradient elements per layer by absolute value before sending gradients to the leader.

Use this run to study the trade-off between smaller gradient payloads, lower communication time, and possible validation-accuracy changes.

## Prerequisites

- Lab 02 has already run successfully on the same machines.
- Tailscale is installed and authenticated on both machines.
- Both workers can reach `leader-macbook-pro.taila5426e.ts.net`.
- The project is installed on both machines:

```bash
pip install -e ".[dev]"
```

## Commands

Terminal 1 on the leader Mac:

```bash
cd /Users/ruturaj_vasant/Desktop/Academic/AI_New/distributed-resnet
dtrain-leader --model resnet18 --epochs 5 --lr 0.01 --batch-size 32 --topk 5000
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

- `summary.json` should report a `compression_ratio` below `1.0`.
- `round_ms` should usually improve compared with Lab 02 because each gradient payload is smaller.
- Validation accuracy may differ from full gradients because TopK discards smaller gradient elements.
- If accuracy drops too much, try a larger value such as `--topk 50000`; if communication is still too slow, try a smaller value.

## How to compare results

List saved runs:

```bash
dtrain-compare --list
```

Compare all three labs:

```bash
dtrain-compare \
  --runs runs/<lab01_run> runs/<lab02_run> runs/<lab03_run> \
  --labels "EXP1 Solo" "EXP2 PS Full" "EXP3 PS TopK" \
  --output comparison.png
```

Use the comparison plot with `summary.json` from each run. The key fields are `final_val_acc`, `duration_seconds`, `samples_per_second`, `seconds_per_batch`, and `compression_ratio`.
