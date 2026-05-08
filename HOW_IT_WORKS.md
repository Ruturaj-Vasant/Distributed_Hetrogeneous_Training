# How Distributed Heterogeneous Training Works
### A complete guide from zero to understanding the system

---

## Chapter 1 — What is Training a Neural Network?

### The knobs analogy

A neural network is a mathematical function that takes an image as input and produces a guess as output. For example, you feed in a picture of a cat and it outputs "cat" (hopefully).

Inside the network there are millions of **weights** — think of them as numbered knobs on a mixing board. ResNet18, the model we use, has **11,279,112 knobs**. When you first create the network, every knob is set to a random value. The network knows nothing. It will guess wrong almost every time.

**Training** is the process of slowly adjusting all 11 million knobs until the network guesses correctly most of the time.

---

### What is a dataset?

We use **Tiny ImageNet-200** — a collection of 100,000 small images (64×64 pixels) belonging to 200 different categories: goldfish, mushroom, sports car, etc. There are 500 images per category.

There is also a separate **validation set** of 10,000 images the network never trains on. We use this to test how well the network generalises to images it has never seen before.

The dataset is downloaded automatically to `~/.cache/tiny-imagenet-200/` the first time a worker runs. It is about 236MB as a zip file and expands to ~500MB on disk. Each worker downloads its own copy locally — the leader never sends images over the network.

---

### What is a batch?

You can't feed all 100,000 images into the network at once — it would require enormous amounts of memory. Instead, you feed images in small groups called **batches**.

We use `--batch-size 32`, meaning 32 images at a time.

```
100,000 training images ÷ 32 images per batch = 3,125 batches
```

---

### What is a step?

One **step** = processing one batch completely:

```
Step breakdown:
  1. Take 32 images from the dataset
  2. Pass them through the network → get 32 predictions
  3. Compare each prediction to the correct label → compute loss
  4. Work backwards through the network (backpropagation)
  5. Adjust all 11 million knobs slightly in the right direction
```

3,125 steps = 1 epoch (the network has seen every training image exactly once).

---

### What is loss?

Loss is a number that measures how wrong the network is right now. Lower is better.

For a 200-class problem, if the network knows absolutely nothing (random guessing):

```
Random loss = log(200) = 5.298
```

After training, loss should fall toward 1.0 or lower as the network gets better. In our observed runs we started at ~5.3 and reached ~4.5 after 1 epoch — barely better than random, because Tiny ImageNet-200 needs 30–90 epochs to properly converge.

---

### What is a gradient?

After you compute how wrong the network is (the loss), you need to know which direction to turn each knob.

**Backpropagation** uses calculus (specifically the chain rule) to compute, for every single knob: "if I increase this knob by a tiny amount, does the loss go up or down, and by how much?"

That value — direction + magnitude — is the **gradient** for that knob.

```
Gradient examples:
  Knob #4,521,003  →  gradient = +0.34   (turning it up makes things worse)
  Knob #7,890,122  →  gradient = -0.82   (turning it up makes things better)
  Knob #1,000,001  →  gradient =  0.001  (nearly irrelevant)
```

The **optimizer** (we use SGD — Stochastic Gradient Descent) then updates every knob:

```
new_knob_value = old_knob_value - (learning_rate × gradient)
```

With `--lr 0.01` (learning rate = 0.01):

```
Knob #4,521,003:  new = old - (0.01 ×  0.34) = old - 0.0034   ← nudged down
Knob #7,890,122:  new = old - (0.01 × -0.82) = old + 0.0082   ← nudged up
```

Do this for all 11,279,112 knobs, 3,125 times per epoch, for 90 epochs — and the network learns.

---

### What is an epoch?

One **epoch** = one complete pass through the entire training dataset.

```
1 epoch  =  3,125 steps  =  100,000 images seen
2 epochs =  6,250 steps  =  200,000 images seen (each image seen twice)
```

Neural networks need to see each image many times from different angles (random augmentations apply random crops and flips each time) before they learn robust features. Tiny ImageNet-200 typically needs 90 epochs for full convergence.

---

## Chapter 2 — The Parameter Server: How Distributed Training Works

### Why distribute training at all?

Training ResNet18 on 100,000 images for 90 epochs takes a very long time on one machine. If you have two machines, you could potentially do it faster by splitting the work.

But it is not as simple as "two machines → 2× faster." The machines need to stay in sync — they both need to be learning from the same model, not diverging in different directions.

This is the core challenge of distributed training.

---

### What is a Parameter Server?

A **parameter server** architecture has two roles:

- **Leader** (your MacBook Pro): Owns the single authoritative copy of the model. Receives gradients from all workers. Updates the model. Sends weight updates back.
- **Worker** (any machine): Has a local copy of the model. Trains on its assigned portion of the data. Computes gradients. Sends them to the leader.

```
                    ┌─────────────────────┐
                    │       LEADER        │
                    │   (MacBook Pro)     │
                    │                     │
                    │  ┌───────────────┐  │
                    │  │  THE MODEL    │  │
                    │  │  11M knobs    │  │
                    │  └───────────────┘  │
                    │  ┌───────────────┐  │
                    │  │  OPTIMIZER    │  │
                    │  │  (SGD)        │  │
                    │  └───────────────┘  │
                    └──────────┬──────────┘
                               │
              ┌────────────────┴────────────────┐
              │  gRPC over Tailscale / localhost │
              │                                  │
    ┌─────────┴────────┐              ┌──────────┴───────┐
    │    WORKER 1      │              │    WORKER 2      │
    │  (your Mac)      │              │  (Nikhil's Mac)  │
    │  25,000 images   │              │  75,000 images   │
    └──────────────────┘              └──────────────────┘
```

---

### What is score-proportional sharding?

When training starts, the leader probes each worker's hardware:

- CPU cores, RAM
- GPU/MPS presence and capability
- A benchmark score (measured by running a test forward pass and timing it)

Based on this score, the leader assigns each worker a **shard** — a slice of the 100,000 training images proportional to its capability.

```
Example:
  Worker 1 (weaker Mac):   score = 1000  →  25% shard = 25,000 images
  Worker 2 (stronger Mac): score = 3000  →  75% shard = 75,000 images

  Total: 100,000 images divided, nothing wasted
```

The batch size also scales with the score, so the stronger machine processes larger batches and pushes gradients at a higher rate.

---

### The flow of one training step (distributed)

Here is exactly what happens for a single batch in the distributed setup:

```
WORKER SIDE                              LEADER SIDE
──────────────────────────────────────   ──────────────────────────────────
1. Load 32 images from local shard
2. Run forward pass through model
   → get 32 predictions
3. Compute loss (how wrong)
4. Run backpropagation
   → compute gradient for all 11M knobs
5. Compress gradients (TopK)
6. Send gradients over gRPC ──────────►  7. Receive gradients from all workers
                                          8. Wait until ALL workers have sent
                                          9. Aggregate (weighted average)
                                         10. Run optimizer step
                                             → update all 11M knobs
                                         11. Compute weight delta
                                             (new knobs − old knobs)
                                         12. Send delta back ◄──────────────
13. Receive weight delta
14. Apply delta to local model
    (local model now matches leader)
15. → repeat for next batch
```

This entire cycle happens **3,125 times per epoch** (or **781 times** with `--grad-accum 4`).

---

### What is gradient accumulation?

In the standard setup, the worker syncs with the leader after every single batch. This means 3,125 gRPC round trips per epoch — very expensive.

**Gradient accumulation** (flag: `--grad-accum N`) makes the worker train N batches locally before sending a single combined gradient to the leader.

```
Without accumulation (accum=1):
  batch → sync → batch → sync → batch → sync   (3,125 syncs/epoch)

With accumulation (accum=4):
  batch → batch → batch → batch → sync          (781 syncs/epoch)
```

PyTorch accumulates gradients naturally — `loss.backward()` adds to existing gradients rather than replacing them. We divide the loss by N before each backward pass so the final accumulated gradient is an average, not a sum.

```
(loss / 4).backward()   ← do this 4 times, THEN sync
```

The effective batch size becomes 4 × 32 = 128. Larger effective batches often improve training stability too.

---

## Chapter 3 — The Technology: gRPC, Protobuf, and Data Transfer

### What is gRPC?

**gRPC** (Google Remote Procedure Call) is a framework for calling functions on another machine over a network, as if they were local functions.

Without gRPC you would have to manually open a TCP socket, define a message format, handle serialisation, errors, timeouts, and retries yourself. gRPC handles all of that.

We define our messages and function signatures in a `.proto` file:

```protobuf
service TrainerService {
  rpc Register(RegisterRequest)        returns (RegisterResponse);
  rpc GetAssignment(GetAssignmentRequest) returns (GetAssignmentResponse);
  rpc ExchangeGradients(GradientPush)  returns (WeightUpdate);
  rpc Heartbeat(stream HeartbeatRequest) returns (stream HeartbeatResponse);
  rpc GetClusterStatus(ClusterStatusRequest) returns (ClusterStatusResponse);
  rpc AdmitWorkers(AdmitWorkersRequest) returns (AdmitWorkersResponse);
}
```

gRPC auto-generates Python code from this file. The worker calls `stub.ExchangeGradients(...)` and it sends the gradients to the leader over the network. The leader's `ExchangeGradients()` function runs and returns the result. To the code it looks like a local function call.

---

### What is Protobuf?

**Protocol Buffers (protobuf)** is the serialisation format gRPC uses. Serialisation means converting Python objects (tensors, lists, numbers) into raw bytes that can be sent over a network, and back again on the other side.

Protobuf is compact and fast compared to JSON.

```
11,279,112 gradient values as JSON text:   ~200MB per sync
11,279,112 gradient values as protobuf:    ~88MB per sync (still large)
Top 5,000 values per layer with TopK:      ~2.4MB per sync
Top 500  values per layer with TopK=500:   ~0.24MB per sync
```

---

### What exactly is sent in each ExchangeGradients call?

With `--grad-accum 4`, the worker sends one message every 4 batches:

```
GradientPush:
  worker_id:         "a3f9"             (which worker)
  global_step:       391                (which sync round)
  local_batch_count: 128                (32 images × 4 accumulated batches)
  loss:              4.4821             (average loss over the 4 batches)

  gradients: [                          (one entry per layer, 62 total)
    SparseGradient {
      layer_name: "layer4.1.conv2.weight"
      shape:      [512, 512, 3, 3]
      indices:    [0, 4421, 19832, ...]  (which of the 2.36M knobs to update)
      values:     [0.034, -0.12, ...]   (the accumulated gradient values)
    },
    ...
  ]
```

The leader responds with a `WeightUpdate`:

```
WeightUpdate:
  global_step: 391
  payload:     <44MB binary>    (torch.save of new_weights − old_weights)
```

---

### The 44MB problem — and how we changed gRPC to handle it

By default, gRPC limits message sizes to **4MB**. gRPC was designed for small API calls — usernames, IDs, short text — not for moving 44MB of neural network weights.

Our gradient payloads and weight deltas are much larger:

```
Without TopK (full gradients):
  Worker → Leader:  11,279,112 values × 8 bytes = ~88MB per sync
  Leader → Worker:  11,279,112 weights × 4 bytes = ~44MB per sync

With TopK=5000:
  Worker → Leader:  ~310,000 values × 8 bytes = ~2.4MB per sync
  Leader → Worker:  still 44MB (weight delta is always the full model)
```

We override the default limits in `leader.py` and `worker.py`:

```python
# leader.py
server = aio.server(options=[
    ("grpc.max_receive_message_length", 256 * 1024 * 1024),  # 256MB
    ("grpc.max_send_message_length",    256 * 1024 * 1024),  # 256MB
])

# worker.py
options = [
    ("grpc.max_send_message_length",    256 * 1024 * 1024),
    ("grpc.max_receive_message_length", 256 * 1024 * 1024),
]
```

Without this, every `ExchangeGradients` call fails with "message too large."

---

### What is Tailscale and why do we use it?

Your MacBook Pro and Nikhil's Mac are on different home networks. Direct connections between them are blocked by routers (NAT/firewall). You cannot just open a port and connect.

**Tailscale** creates a private encrypted network between any devices you add to it. Every device gets a stable DNS address (e.g. `leader-macbook-pro.taila5426e.ts.net`). Traffic is encrypted using **WireGuard** — a modern VPN protocol.

```
Nikhil's Mac  ──►  WireGuard encrypted tunnel  ──►  Your MacBook Pro
              (works through any NAT, any firewall)
```

The cost: WireGuard encryption caps throughput at ~200–400 Mbps, versus 10+ Gbps on localhost loopback.

```
44MB weight delta over Tailscale:  44 ÷ 25 MB/s = 1.76 seconds per sync
44MB weight delta on localhost:    44 ÷ 1000 MB/s = 0.044 seconds per sync

That is a 40× difference in transfer speed.
```

This is why EXP0 (localhost) vs EXP1 (Tailscale, same machine) directly measures the WireGuard tax.

---

### The heartbeat stream

Beyond gradient exchange, every worker maintains a long-lived **bidirectional streaming RPC** with the leader called Heartbeat. Every 5 seconds:

```
Worker → Leader:  HeartbeatRequest  { worker_id, status, current_loss, steps_done }
Leader → Worker:  HeartbeatResponse { command: CONTINUE | PAUSE | STOP | RESHARD }
```

This serves three purposes:
1. **Liveness detection** — if a worker stops sending heartbeats for 60 seconds, the leader marks it dead and removes it from the gradient sync group so other workers are not blocked forever waiting for it.
2. **Command delivery** — the leader can tell a worker to pause or stop by queuing a command that gets delivered in the next heartbeat response.
3. **Shard rebalancing** — when a new worker is admitted mid-training, the leader sends a `RESHARD` command to existing workers via the heartbeat, carrying their new (smaller) shard assignment.

---

### The pending worker pool and dynamic admission (Phase 1)

When a worker connects mid-training (after `start` has been typed), it is held in a **pending pool** rather than immediately joining the active training. The operator sees it in `watch_cluster.py` and explicitly admits it.

```
Worker registers mid-run
        ↓
Leader: "You're pending. Wait."
        ↓
watch.py shows: PENDING WORKERS table
        ↓
Operator types: admit all
        ↓
Leader calls AdmitWorkers RPC
        ↓
Phase 2 shard rebalancing kicks in (see below)
        ↓
Worker gets its shard assignment + model weights → joins gradient sync rounds
Existing workers get RESHARD command → rebuild loaders with new (smaller) shards
```

This gives the operator control over when a new worker enters — useful if a machine joins with low battery, unstable network, or you simply want to finish the current epoch cleanly before adding load.

---

### True shard rebalancing when a worker is admitted (Phase 2)

Phase 1 only held workers in a queue. Phase 2 is what actually happens when `admit` is called — the entire dataset is redistributed proportionally:

**Before admission** (1 worker, score=1000):
```
laptop-first:   100% of dataset  →  100,000 samples
laptop-late:    waiting in pending pool
```

**After `admit laptop-late`** (2 workers, scores 1000 + 500):
```
laptop-first:   66.7% of dataset  →  66,666 samples  ← REDUCED
laptop-late:    33.3% of dataset  →  33,334 samples  ← NEW
```

The full sequence on the leader when `admit` is called:

```
1. Recompute score-proportional shards across ALL active + new workers
2. For each existing worker:
     a. Set assigned=False (exclude from gradient rounds immediately)
     b. Store new ShardAssignment in pending_reshard field
     c. Queue RESHARD command in cmd_queue
3. For the new worker:
     a. Store its new shard in shard_indices
     b. Unblock its GetAssignment call → it starts training
```

The existing workers receive the RESHARD on their next heartbeat (within ≤5 seconds):

```
Worker receives RESHARD heartbeat response
        ↓
Sets shared.stop = True, stores new_indices in shared.reshard_indices
        ↓
Finishes current batch (including ExchangeGradients sync)
        ↓
Exits training_loop at top of next batch
        ↓
Rebuilds DataLoader with new (smaller) shard indices
        ↓
Re-enters training_loop from the correct epoch (not from epoch 0)
        ↓
First ExchangeGradients call → leader re-includes worker in gradient rounds
```

**Key correctness properties:**
- Workers complete their in-flight gradient sync before stopping — no deadlock
- The leader excludes resharding workers from gradient round counts (`assigned=False`) so remaining workers are not blocked
- When a resharding worker's first new `ExchangeGradients` call arrives, the leader automatically re-includes it
- Epoch continuation: workers track `epoch_completed` so they restart from the right epoch, not from zero
- All 100,000 dataset samples remain covered — no images are skipped or double-counted after rebalancing

---

## Chapter 4 — The Full Journey: One Sync Round with Gradient Accumulation

With `--grad-accum 4` and `--topk 5000`, here is the full sequence for one sync round (covering 4 batches = 128 images):

---

### Batches 1–3: local accumulation (Worker side, ~4s total)

```
Batch 1:
  Load 32 images: ~370ms
  Forward pass:   ~300ms
  Backprop:       ~600ms
  (loss / 4).backward() → gradients ADDED to existing grad buffers
  No sync yet.

Batch 2: same → gradients ADDED again
Batch 3: same → gradients ADDED again
```

PyTorch grad buffers now hold the sum of 3 batches of gradients divided by 4.

---

### Batch 4: final accumulation + sync

```
Batch 4 compute:   ~1,270ms  (load + forward + backprop)
TopK compress:     ~50ms     (keep top 5,000 per layer)
gRPC send (2.4MB): ~2ms      (localhost) / ~96ms (Tailscale)
                             ── leader receives, waits for all workers ──
Leader aggregate:  ~200ms    (weighted average of all worker gradients)
Optimizer step:    ~50ms     (update 11M knobs)
Weight delta:      ~44ms     (torch.save + send back on localhost)
Apply delta:       ~100ms    (worker adds delta to its model)
Zero gradients:    ~20ms     (clear grad buffers, ready for next round)
───────────────────────────
Total per 4 batches: ~5.7s
Effective per batch: ~1.4s   (vs ~5s without any optimisation)
```

---

### Impact over a full epoch

```
Syncs per epoch: 3,125 ÷ 4 = 781
Total time:      781 × 5.7s = 4,454s ≈ 74 minutes per epoch (localhost)

Compare to original (no TopK, no accum, via Tailscale):
  3,125 × 5s = 15,625s ≈ 4.3 hours per epoch

Speedup: 4.3 hours → 74 minutes = 3.5× faster, same machine, same result
```

---

## Chapter 5 — Speed Optimisations: Why Each One Helps

### Optimisation 1: TopK gradient sparsification

**Problem:** Sending 11M gradient values per sync costs 88MB and ~1.76s over Tailscale.

**Fix:** Only send the K largest gradients (by absolute value) per layer. The other 11M − K gradients are small and contribute little to learning. They will appear in a future sync when they grow larger.

```
topk=5000:  310,000 values sent  → 2.4MB   (36× smaller)
topk=500:    31,000 values sent  → 0.24MB  (367× smaller)
```

**Cost:** Slightly slower convergence because small gradients are skipped each step. In practice, with a parameter server and many steps, the accuracy difference is small.

---

### Optimisation 2: Gradient accumulation

**Problem:** 3,125 gRPC round trips per epoch. Each round trip has fixed overhead (connection, serialisation, lock acquisition on leader) regardless of payload size.

**Fix:** Train N batches locally, accumulate gradients, send once. Reduces round trips by N×.

**Why accumulated gradients are still correct:** Averaging gradients over N batches is mathematically equivalent to computing the gradient over a single batch of N×32 images. The learning signal is the same; you just paid the communication cost once instead of N times.

**Cost:** Higher memory usage (grad buffers stay alive for N batches). Effective batch size increases (can require adjusting learning rate for long training runs, not a concern for our 2-epoch experiments).

---

### Optimisation 3: localhost vs Tailscale

**Problem:** Same-machine Tailscale routes traffic through WireGuard, capping throughput at ~25 MB/s even when leader and worker are on the same computer.

**Fix:** `python3 worker.py --leader localhost` — uses loopback at 1000+ MB/s.

**Rule:** Use `localhost` when leader and worker are on the same machine. Use `leader-macbook-pro.taila5426e.ts.net` only when connecting from a different machine.

---

## Chapter 6 — The Four Experiments

The experiments are designed to isolate and measure three independent costs:
1. The Tailscale/WireGuard overhead (EXP0 vs EXP1)
2. The benefit of a second worker (EXP1 vs EXP2)
3. The effect of aggressive gradient compression (EXP2 vs EXP3)

---

### EXP0 — Localhost solo baseline

One leader, one worker, same machine, worker connects via `localhost`. This is the **pure compute baseline** — no WireGuard, minimal network overhead.

```bash
# Terminal 1
python3 leader.py --model resnet18 --epochs 2 --topk 5000 --grad-accum 4

# Terminal 2 (same machine)
python3 worker.py --leader localhost
```

**Records:** duration, loss curve, val_acc, samples/sec.
**Purpose:** Reference point. Everything else is measured relative to this.

---

### EXP1 — Tailscale solo (same machine, WireGuard overhead)

Same setup as EXP0 but the worker connects via the Tailscale DNS name. Traffic flows through WireGuard even though both processes are on the same machine.

```bash
# Terminal 1
python3 leader.py --model resnet18 --epochs 2 --topk 5000 --grad-accum 4

# Terminal 2 (same machine, but via Tailscale)
python3 worker.py --leader leader-macbook-pro.taila5426e.ts.net
```

**Records:** same as EXP0.
**Measures:** EXP1 duration − EXP0 duration = pure WireGuard tax.

Expected result: EXP1 is slower than EXP0 by the transfer overhead of the 44MB weight delta through WireGuard (~1.76s per sync × 781 syncs = ~23 extra minutes per epoch).

---

### EXP2 — Two machines, distributed parameter server

Two workers on two different machines, connected over Tailscale. Your Mac runs one worker (via localhost to avoid double-Tailscale overhead); Nikhil's Mac runs the other via Tailscale.

```bash
# Terminal 1 — Leader (your Mac)
python3 leader.py --model resnet18 --epochs 2 --topk 5000 --grad-accum 4

# Terminal 2 — Worker 1 (your Mac, same machine)
python3 worker.py --leader localhost

# Nikhil's Mac — Worker 2
python3 worker.py --leader leader-macbook-pro.taila5426e.ts.net

# Terminal 3 — Live monitor (any machine)
python3 watch_cluster.py
```

**Measures:**
- **Speedup** = EXP0 duration ÷ EXP2 duration
- **Efficiency** = Speedup ÷ 2 workers
- **Straggler effect** — the slower worker delays every sync round. Visible in per-worker `compute_ms` and `wait_ms` logs.

Expected: ~1.4–1.8× speedup (not 2× due to straggler + Tailscale overhead on Nikhil's side).

---

### EXP3 — Two machines, aggressive TopK compression

Same topology as EXP2, but TopK is dropped from 5,000 to 500. This sends 10× fewer gradient values per sync.

```bash
# Terminal 1 — Leader
python3 leader.py --model resnet18 --epochs 2 --topk 500 --grad-accum 4

# Workers same as EXP2
```

**Measures:**
- **Compression ratio** = 31,000 / 11,279,112 = 0.003 (99.7% of gradients dropped)
- **Accuracy delta** — does val_acc drop vs EXP2 due to missing gradients?
- **Throughput delta** — does the smaller payload reduce per-sync time?

Expected: slightly faster syncs (less data over Tailscale), small accuracy drop (most skipped gradients were small anyway).

---

### Reading the results

After all four runs:

```bash
# List all runs with key metrics
python3 run_experiments.py --list

# Generate comparison plots + full table
python3 run_experiments.py \
    --runs runs/exp0_dir runs/exp1_dir runs/exp2_dir runs/exp3_dir \
    --labels "EXP0 Localhost" "EXP1 Tailscale Solo" "EXP2 Distributed" "EXP3 TopK=500" \
    --output comparison.png
```

Output table columns: Workers | Duration | Samples/s | Loss | Val Acc | Compression | Speedup | Efficiency

```
Expected results:
  EXP0  1 worker  localhost   fastest solo     baseline accuracy   speedup=1.0×
  EXP1  1 worker  Tailscale   ~30min slower    same accuracy       shows WireGuard cost
  EXP2  2 workers Tailscale   1.4–1.8× faster  same accuracy       shows distributed gain
  EXP3  2 workers Tailscale   slightly faster  small drop          shows compression tradeoff
```

---

## Chapter 7 — System Features

### Self-bootstrapping workers

When you run `python3 worker.py` on a fresh machine, the script:

1. Detects it is not inside a virtual environment
2. Creates `.venv/` using the system Python
3. Installs all dependencies from `requirements.txt` (torch, torchvision, grpc, etc.)
4. Generates gRPC Python stubs from `proto/trainer.proto`
5. Re-executes itself inside the virtual environment

All of this happens before any third-party imports are attempted, using only Python's standard library. The re-execution on Mac/Linux uses `os.execv` (replaces the process in-place). On Windows it uses `subprocess.run` + `sys.exit`.

On subsequent runs, a hash of `requirements.txt` is stored in `.venv/.deps_hash` and pip is only re-run if the hash changes — so startup is instant when nothing has changed.

---

### Auto-reconnect on leader restart

If the leader process is stopped and restarted (e.g. to change flags), workers do not exit. They detect the connection loss via a `leader_disconnected` flag set in the heartbeat loop, wait 10 seconds, then reconnect and re-register automatically.

This means you can restart the leader with different `--topk` or `--epochs` settings without having to restart every worker.

---

### Straggler detection

Every sync round, the leader records when each worker's gradient arrives. After aggregation it computes:

```
compute_ms = time from previous round end to this worker's arrival
             (= how long the worker spent training its batch)

wait_ms    = time from this worker's arrival to round completion
             (= how long this worker waited for the slowest peer)
```

The worker with the highest `wait_ms` is the straggler — everyone waited for it. If `wait_ms > 200ms` and there are multiple workers, the leader logs:

```
laptop-weak: compute=380ms wait=215ms ← straggler
laptop-strong: compute=390ms wait=0ms
```

The `round_ms` and per-step `straggler_delay_s` are saved to `metrics.csv` for post-run analysis.

---

### Run recording

Every training run is saved to `runs/<model>__<dataset>__<optimizer>__topk<K>__<timestamp>/`:

```
config.json    — all hyperparameters + system metadata
metrics.csv    — per-step: loss, round_ms, straggler_delay_s
               — per-epoch: val_acc
summary.json   — everything in config + derived results:
                   duration_seconds, batches_per_epoch,
                   seconds_per_batch, samples_per_second,
                   compression_ratio, straggler_delay_seconds,
                   straggler_delay_total_seconds,
                   world_size, parallelism
loss.png       — training loss curve (matplotlib)
```

---

## Glossary

| Term | Meaning |
|---|---|
| Weight / Parameter | A single adjustable number inside the neural network (a "knob") |
| Gradient | The direction and magnitude to adjust one knob to reduce loss |
| Loss | A number measuring how wrong the network's predictions are |
| Batch | A small group of training images processed together (we use 32) |
| Step / Sync round | One gradient exchange with the leader (covers 1 batch without accum, N batches with) |
| Epoch | One full pass through the entire training dataset (3,125 steps at batch=32) |
| Gradient accumulation | Training N batches locally before one sync, reducing round trips by N× |
| Backpropagation | Algorithm that computes gradients by working backwards through the network |
| Learning rate | How much to adjust each knob per step (we use 0.01) |
| SGD | Stochastic Gradient Descent — the optimizer that adjusts knobs using gradients |
| Parameter Server | Architecture where one machine (leader) owns the model and coordinates workers |
| Shard | Each worker's assigned slice of the training dataset |
| Score-proportional sharding | Giving stronger workers a larger slice of the dataset |
| TopK sparsification | Only sending the K largest-magnitude gradients, dropping the rest |
| Gradient compression ratio | Fraction of gradients actually sent (sent / total) |
| gRPC | Framework for calling functions on a remote machine over a network |
| Protobuf | Binary serialisation format used by gRPC |
| Tailscale | VPN that connects machines across different networks using WireGuard |
| WireGuard | Encrypted tunnel protocol used by Tailscale (~25 MB/s throughput) |
| Localhost loopback | Direct same-machine communication (~1000 MB/s, no encryption) |
| Speedup | EXP0 duration ÷ this run's duration |
| Efficiency | Speedup ÷ number of workers (100% = perfect linear scaling) |
| Straggler | The slowest worker in a round — all others wait for it |
| Straggler delay | How long faster workers waited idle for the straggler (wait_ms) |
| val_acc | Validation accuracy — fraction of unseen images correctly classified |
| Pending pool | Workers that registered mid-training and await manual admission |
| AdmitWorkers | RPC that moves pending workers into the active training run |
| Phase 1 admission | Holding late workers in a pending pool until the operator explicitly admits them |
| Phase 2 rebalancing | When a worker is admitted, redistributing ALL shards proportionally (not just assigning a leftover slice) |
| RESHARD command | Heartbeat command telling an existing worker to stop, reload a new (smaller) shard, and continue training |
| assigned=False | Temporary state during reshard: worker excluded from gradient rounds until its new loader is ready |
| Heartbeat | Bidirectional stream that keeps leader informed of worker liveness |
| Weight delta | new_weights − old_weights, sent from leader to workers each sync |
| Self-bootstrap | worker.py creates its own venv and installs deps on first run |
