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

---

### What is a batch?

You can't feed all 100,000 images into the network at once — it would require enormous amounts of memory (RAM/GPU). Instead, you feed images in small groups called **batches**.

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

After training, loss should fall toward 1.0 or lower as the network gets better. In our run we started at ~5.3 and reached ~4.5 after 1 epoch — barely better than random, because Tiny ImageNet-200 needs 30–90 epochs to properly converge.

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
Knob #4,521,003:  new = old - (0.01 × 0.34)  = old - 0.0034   ← nudged down
Knob #7,890,122:  new = old - (0.01 × -0.82) = old + 0.0082   ← nudged up
```

Do this for all 11,279,112 knobs, 3,125 times per epoch, for 90 epochs — and the network learns.

---

### What is an epoch?

One **epoch** = one complete pass through the entire training dataset.

```
1 epoch  =  3,125 steps  =  100,000 images seen
5 epochs =  15,625 steps =  500,000 images seen (each image seen 5 times)
```

Neural networks need to see each image many times from different angles (due to random augmentations) before they learn robust features. Tiny ImageNet-200 typically needs 90 epochs for full convergence.

---

## Chapter 2 — The Parameter Server: How Distributed Training Works

### Why distribute training at all?

Training ResNet18 on 100,000 images for 90 epochs takes a very long time on one machine. If you have two machines, you could potentially do it faster by splitting the work.

But it's not as simple as "two machines → 2× faster." The machines need to stay in sync — they both need to be learning from the same model, not diverging in different directions.

This is the core challenge of distributed training.

---

### What is a Parameter Server?

A **parameter server** architecture has two roles:

- **Leader** (your MacBook Pro): Owns the single authoritative copy of the model. Receives gradients from all workers. Updates the model. Sends updates back.
- **Worker** (any machine): Has a copy of the model. Trains on its assigned portion of the data. Computes gradients. Sends them to the leader.

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
              │  gRPC over Tailscale network     │
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

- CPU cores
- RAM
- GPU/MPS capability
- A benchmark score (how fast it can run a test forward pass)

Based on this score, the leader assigns each worker a **shard** — a slice of the 100,000 training images proportional to the worker's capability.

```
Example:
  Worker 1 (weaker Mac):  score = 1000  →  25% shard = 25,000 images
  Worker 2 (stronger Mac): score = 3000  →  75% shard = 75,000 images

  Total: 100,000 images split, nothing wasted
```

The stronger machine does more work because it can handle it. This is the "heterogeneous" part of our system — it works with machines of different capabilities rather than requiring identical hardware.

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
5. Compress gradients (TopK if enabled)
6. Send gradients over gRPC ──────────►  7. Receive gradients from all workers
                                          8. Wait until ALL workers have sent
                                          9. Aggregate (weighted average)
                                         10. Run optimizer step
                                             → update all 11M knobs
                                         11. Compute weight delta
                                             (new knobs - old knobs)
                                         12. Send delta back ◄──────────────
13. Receive weight delta
14. Apply delta to local model
    (local model now matches leader)
15. → repeat for next batch
```

This entire cycle happens **3,125 times per epoch**.

---

## Chapter 3 — The Technology: gRPC, Protobuf, and Data Transfer

### What is gRPC?

**gRPC** (Google Remote Procedure Call) is a framework for calling functions on another machine over a network, as if they were local functions.

Without gRPC you'd have to manually open a TCP socket, define a message format, handle errors, timeouts, and retries yourself. gRPC handles all of that.

We define our messages and function signatures in a `.proto` file:

```protobuf
// trainer.proto (simplified)

service TrainerService {
  rpc Register(RegisterRequest)   returns (RegisterResponse);
  rpc ExchangeGradients(GradientPush) returns (WeightUpdate);
  rpc Heartbeat(stream HeartbeatRequest) returns (stream HeartbeatResponse);
}
```

gRPC auto-generates Python code from this file. The worker calls `stub.ExchangeGradients(...)` and it magically sends the gradients to the leader over the network.

---

### What is Protobuf?

**Protocol Buffers (protobuf)** is the serialisation format gRPC uses. Serialisation means converting Python objects (tensors, lists, numbers) into raw bytes that can be sent over a network, and back.

Protobuf is compact and fast compared to JSON (which would be enormous for 11 million gradient values).

```
11,279,112 gradient values as JSON text:  ~200MB per step
11,279,112 gradient values as protobuf:   ~88MB per step (still large)
11,279,112 values with TopK (top 5,000):  ~0.08MB per step
```

---

### What exactly is sent in each ExchangeGradients request?

Every batch step, the worker sends a `GradientPush` message containing:

```
GradientPush:
  worker_id:         "a3f9"          (which worker is this)
  global_step:       1547            (which training step)
  local_batch_count: 32              (how many images in this batch)
  loss:              4.5820          (training loss for this batch)
  
  gradients: [                       (one entry per layer in the network)
    SparseGradient {
      layer_name: "layer1.0.conv1.weight"
      shape:      [64, 3, 3, 3]      (shape of this layer's weight tensor)
      indices:    [0, 5, 12, ...]    (which knobs have non-zero gradients)
      values:     [0.34, -0.82, ...] (the gradient values at those positions)
    },
    SparseGradient { ... },          (one per layer, 62 layers total)
    ...
  ]
```

The leader responds with a `WeightUpdate`:

```
WeightUpdate:
  global_step: 1547
  payload:     <44MB of binary data>   (weight delta for all 11M knobs)
```

---

### The 44MB problem — and how we changed gRPC to handle it

By default, gRPC limits message sizes to **4MB**. This is intentional — gRPC was designed for small, fast API calls (usernames, IDs, short text), not for moving 44MB of neural network weights.

Our gradients + weight delta are far larger:

```
Without TopK (full gradients):
  Worker → Leader:  11,279,112 values × 8 bytes = ~88MB per step
  Leader → Worker:  11,279,112 weights × 4 bytes = ~44MB per step

With TopK (top 5,000):
  Worker → Leader:  5,000 values × 8 bytes = ~0.04MB per step
  Leader → Worker:  still 44MB (we send the full weight delta back)
```

We override the default gRPC limits in `leader.py`:

```python
server = aio.server(options=[
    ("grpc.max_receive_message_length", 256 * 1024 * 1024),  # 256MB
    ("grpc.max_send_message_length",    256 * 1024 * 1024),  # 256MB
])
```

And on the worker side too, when opening the channel:

```python
options = [
    ("grpc.max_send_message_length",    256 * 1024 * 1024),
    ("grpc.max_receive_message_length", 256 * 1024 * 1024),
]
channel = grpc.insecure_channel(f"{leader}:{port}", options=options)
```

Without this change, every `ExchangeGradients` call would fail with "message too large."

---

### What is Tailscale and why do we use it?

Your MacBook Pro and Nikhil's Mac are on different home networks. Direct connections between home machines are blocked by routers (NAT/firewall). You can't just open a port and connect.

**Tailscale** creates a private encrypted network between any devices you add to it. Every device gets a stable address (e.g. `leader-macbook-pro.taila5426e.ts.net`). Traffic between devices is encrypted using **WireGuard** (a modern VPN protocol).

```
Nikhil's Mac  ──►  Tailscale WireGuard tunnel  ──►  Your MacBook Pro
              (encrypted, works through NAT)
```

The downside: WireGuard encryption has CPU overhead and caps throughput at ~200–400 Mbps, vs 10+ Gbps on localhost loopback.

```
44MB weight delta over Tailscale:   44 ÷ 25 MB/s = 1.76 seconds per step
44MB weight delta on localhost:     44 ÷ 1000 MB/s = 0.044 seconds per step

40× slower over Tailscale!
```

This is why we use `--leader localhost` when running both worker and leader on the same machine, and only use the Tailscale DNS when connecting from a different machine.

---

## Chapter 4 — The Full Journey: One Batch from Disk to Weight Update

Let's trace exactly what happens to one batch of 32 images, with real numbers from our system.

---

### Step 1: Data loading (Worker side, ~500ms)

The worker has been assigned 25,000 images (its shard). These are JPEG files on disk at `~/.cache/tiny-imagenet-200/`.

The DataLoader picks 32 random images from this shard:

```
Load image from disk:  ~5ms × 32 images = ~160ms
JPEG decoding:         ~3ms × 32 images = ~96ms
Transforms:
  RandomResizedCrop    ~2ms × 32 = ~64ms
  RandomHorizontalFlip ~0.5ms × 32 = ~16ms
  Normalize            ~1ms × 32 = ~32ms

Total data loading: ~370ms
```

The result is a tensor of shape `[32, 3, 64, 64]` — 32 images, 3 colour channels (RGB), 64×64 pixels.

---

### Step 2: Forward pass (Worker side, ~300ms)

The 32-image tensor is fed through ResNet18's layers:

```
Input:          [32, 3, 64, 64]      (32 images, RGB, 64px)
Conv layer 1:   [32, 64, 32, 32]     (64 feature maps, 32px)
Conv layer 2:   [32, 64, 32, 32]
ResBlock 1-4:   [32, 128, 16, 16]
ResBlock 5-8:   [32, 256, 8, 8]
ResBlock 9-12:  [32, 512, 4, 4]
GlobalAvgPool:  [32, 512]
Linear:         [32, 200]            (200 class scores)
```

Output: 32 rows of 200 numbers each. The highest number in each row is the network's prediction.

```
Image 1: [0.02, -0.5, 3.1, 0.7, ...]  →  predicted class 3 ("bullfrog")
Image 2: [-1.2, 4.3, 0.1, -0.3, ...]  →  predicted class 2 ("goldfish")
...
```

---

### Step 3: Loss computation (Worker side, ~5ms)

Cross-entropy loss compares predictions to correct labels:

```
Image 1: predicted class 3 (bullfrog), actual class 47 (basketball) → wrong
Image 2: predicted class 2 (goldfish),  actual class 2 (goldfish)   → correct

Loss = average of -log(probability assigned to correct class) across 32 images

If network was random: loss ≈ log(200) = 5.298
After some training:   loss ≈ 4.5      (slowly improving)
```

---

### Step 4: Backpropagation (Worker side, ~600ms)

PyTorch automatically works backwards through every operation in the forward pass using the chain rule of calculus, computing the gradient (the "blame") for every one of the 11,279,112 knobs.

```
Result: a gradient tensor for each of the 62 parameter layers in ResNet18

layer4.1.conv2.weight: gradient tensor of shape [512, 512, 3, 3] = 2,359,296 values
layer4.1.conv1.weight: gradient tensor of shape [512, 512, 3, 3] = 2,359,296 values
...
conv1.weight:          gradient tensor of shape [64, 3, 3, 3]    = 1,728 values

Total: 11,279,112 gradient values across 62 layers
```

---

### Step 5: TopK compression (Worker side, ~50ms)

With `--topk 5000`, only the 5,000 largest-magnitude gradients per layer are kept. The rest are discarded (treated as zero this step).

```
layer4.1.conv2.weight has 2,359,296 gradients.
We keep only the top 5,000 by absolute value.
Compression ratio for this layer: 5,000 / 2,359,296 = 0.21%

Small layers (e.g. conv1.weight with 1,728 values) send all gradients
since 1,728 < 5,000.

Without TopK: 11,279,112 values × 8 bytes = 88MB
With TopK:    ~310,000 values × 8 bytes   = 2.4MB  (36× smaller)
```

Each kept gradient is sent as a pair: `(index, value)` so the leader knows which knob it belongs to.

---

### Step 6: gRPC send — Worker to Leader (~100ms on localhost, ~1.5s over Tailscale)

The worker calls `stub.ExchangeGradients(push)` — a blocking call. The worker waits here until the leader responds.

```
Payload sent:
  Header + metadata:  ~1KB
  62 SparseGradient messages: ~2.4MB (with TopK)
  
Wire format: protobuf binary encoding
Transport: TCP (via Tailscale WireGuard or localhost loopback)
```

If multiple workers are running, the worker blocks here until ALL workers have sent their gradients. It does not receive the response until the last worker also sends.

---

### Step 7: Aggregation on Leader (~200ms)

The leader receives gradients from all workers and combines them using a **weighted average**. Workers with more images in their batch get higher weight.

```
Worker 1 processed 16 images (batch_count=16)
Worker 2 processed 32 images (batch_count=32)
Total: 48 images

Weight for Worker 1: 16/48 = 0.333
Weight for Worker 2: 32/48 = 0.667

For each gradient position:
  combined_gradient = 0.333 × worker1_gradient + 0.667 × worker2_gradient
```

This is mathematically equivalent to computing the gradient over all 48 images together.

---

### Step 8: Optimizer step on Leader (~50ms)

The leader applies SGD to update every knob on its copy of the model:

```
For every knob:
  knob = knob - (lr × combined_gradient)

With lr = 0.01:
  knob #4,521,003:  new = old - (0.01 × 0.34) = old - 0.0034
```

All 11,279,112 knobs are updated.

---

### Step 9: Compute and send weight delta (~1.5s)

The leader computes the delta (what changed) and sends it back:

```
delta = new_weights - old_weights

Serialised with torch.save → binary format
Size: ~44MB for all 11,279,112 parameters (always full, not sparse)

This is currently the biggest bottleneck:
  - torch.save must copy all tensors from GPU to CPU
  - Serialise them (pickle-based format)
  - Send 44MB over the network
  - Worker deserialises and copies back to GPU
```

---

### Step 10: Worker applies delta (~100ms)

```
worker_model_weights += delta
```

The worker's model now exactly matches the leader's model. Both are in sync.

One step is complete. On to batch 2 of 3,125.

---

## Chapter 5 — Why Is It Slow and What We Can Do

### Current per-step timing (solo, same machine via Tailscale)

```
Data loading:           ~370ms
Forward pass:           ~300ms
Backpropagation:        ~600ms
gRPC send (88MB):       ~1,760ms  ← via Tailscale at 25MB/s
Leader aggregation:     ~200ms
torch.save + send back: ~1,760ms  ← via Tailscale at 25MB/s
Apply delta:            ~100ms
─────────────────────────────────
Total per step:         ~5,090ms  ≈ 5 seconds

3,125 steps × 5 seconds = 15,625 seconds = 4.3 hours per epoch
```

This matches exactly what we observed.

---

### With TopK (topk=5000) and localhost

```
Data loading:           ~370ms
Forward pass:           ~300ms
Backpropagation:        ~600ms
Compress (TopK):        ~50ms
gRPC send (2.4MB):      ~2ms     ← localhost at 1000MB/s
Leader aggregation:     ~200ms
torch.save + send back: ~44ms    ← still 44MB weight delta
Apply delta:            ~100ms
─────────────────────────────────
Total per step:         ~1,666ms ≈ 1.7 seconds

3,125 steps × 1.7 seconds = 5,312 seconds = 1.5 hours per epoch
```

The bottleneck shifts: gradient send becomes negligible, but the 44MB weight delta sent back is still a cost (44ms on localhost — acceptable).

---

### With gradient accumulation (accum=4) + TopK + localhost

Instead of syncing every batch, sync every 4 batches:

```
4 batches of compute:   4 × 1,370ms = 5,480ms
1 gRPC round trip:      ~250ms
─────────────────────────────────
Per 4 batches:          5,730ms = 1,432ms per batch effective

3,125 ÷ 4 = 781 syncs × 5,730ms = 4,474 seconds = 1.2 hours per epoch
```

Smaller win than TopK because compute dominates.

---

## Chapter 6 — The Experiments

### EXP1: Solo baseline

One leader, one worker, both on the same machine. No network overhead. This is the reference point for measuring how much distributed training helps (or hurts).

```
Command:
  Terminal 1: python3 leader.py --model resnet18 --epochs 2 --topk 5000
  Terminal 2: python3 worker.py --leader localhost
```

Records: training time, loss curve, val_acc per epoch.

---

### EXP2: Distributed, full parameter server

Two workers on different machines, both connected over Tailscale. Gradients flow fully (or with TopK) over the network.

```
Command:
  Terminal 1 (your Mac):    python3 leader.py --model resnet18 --epochs 2 --topk 5000
  Terminal 2 (your Mac):    python3 worker.py --leader localhost
  Terminal 3 (Nikhil's Mac): python3 worker.py --leader leader-macbook-pro.taila5426e.ts.net
```

Measures: does adding a second worker reduce training time? By how much?

**Speedup** = EXP1 duration ÷ EXP2 duration
**Efficiency** = Speedup ÷ number of workers (how well did we use the extra machine?)

---

### EXP3: Distributed with aggressive TopK

Same as EXP2 but with higher compression. Fewer gradient values sent per step.

```
Command (only leader changes):
  python3 leader.py --model resnet18 --epochs 2 --topk 500
```

Measures:
- Does heavy compression hurt accuracy? (compare val_acc to EXP2)
- Does it speed things up? (compare duration to EXP2)
- What is the compression ratio? (compressed_gradient_numel ÷ raw_gradient_numel)

```
topk=5000:  send 5,000 × 62 layers ≈ 310,000 values  (compression ratio: 0.027)
topk=500:   send  500 × 62 layers ≈  31,000 values   (compression ratio: 0.003)
```

---

### Reading the results

After all three runs, use the comparison tool:

```bash
python3 run_experiments.py --list
python3 run_experiments.py \
    --runs runs/exp1_dir runs/exp2_dir runs/exp3_dir \
    --labels "EXP1 Solo" "EXP2 Distributed" "EXP3 TopK" \
    --output comparison.png
```

This generates a table showing:

| Metric | EXP1 Solo | EXP2 Distributed | EXP3 TopK |
|---|---|---|---|
| Workers | 1 | 2 | 2 |
| Duration | baseline | ? | ? |
| Samples/sec | baseline | ? | ? |
| Val accuracy | baseline | should match | may drop slightly |
| Compression ratio | 1.0 | 1.0 | 0.003 |
| Speedup | 1.0× | target: ~1.5–1.8× | similar |
| Efficiency | 100% | target: ~75–90% | similar |

Perfect linear speedup (2 workers = 2× faster) is impossible in practice because of communication overhead and the straggler effect (the leader waits for the slowest worker every step). Achieving 75–90% efficiency is considered good in distributed training research.

---

## Glossary

| Term | Meaning |
|---|---|
| Weight / Parameter | A single adjustable number inside the neural network (a "knob") |
| Gradient | The direction and magnitude to adjust one knob to reduce loss |
| Loss | A number measuring how wrong the network's predictions are |
| Batch | A small group of training images processed together |
| Step | Processing one batch: forward → loss → backprop → update |
| Epoch | One full pass through the entire training dataset |
| Backpropagation | Algorithm that computes gradients by working backwards through the network |
| Learning rate | How much to adjust each knob per step (we use 0.01) |
| SGD | Stochastic Gradient Descent — the optimizer that adjusts knobs using gradients |
| Parameter Server | Architecture where one machine (leader) owns the model and coordinates workers |
| Shard | Each worker's assigned slice of the training dataset |
| TopK sparsification | Only sending the K largest gradients, discarding the small ones |
| gRPC | Framework for calling functions on a remote machine over a network |
| Protobuf | Binary serialisation format used by gRPC |
| Tailscale | VPN that connects machines across different networks using WireGuard |
| WireGuard | Encrypted tunnel protocol used by Tailscale |
| Speedup | How much faster distributed training is vs solo (EXP1 time / EXP2 time) |
| Efficiency | Speedup per worker (speedup ÷ num_workers, ideally 100%) |
| Straggler | The slowest worker in a round — everyone waits for it |
| val_acc | Validation accuracy — how often the model is correct on unseen images |
| Compression ratio | Fraction of gradients actually sent (compressed / raw) |
