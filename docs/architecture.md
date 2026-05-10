# Architecture

## System Design

This project uses a parameter-server architecture for heterogeneous ResNet training across laptops connected by Tailscale. The leader owns the authoritative model and optimizer. Workers own local data shards, compute gradients, send compressed gradients to the leader, receive weight deltas, and apply those deltas to their local model copies.

The system intentionally does not use PyTorch DistributedDataParallel. DDP works best when the cluster is relatively homogeneous and can use a shared backend such as NCCL or Gloo. This project targets mixed Mac MPS and Windows CUDA machines, where those assumptions break down. A parameter server can assign unequal work by hardware score, tolerate mixed devices, and manage late workers explicitly.

## gRPC API

The protobuf service defines six RPCs:

- `Register`: a worker sends hardware information and benchmark results; the leader returns a worker id.
- `Heartbeat`: bidirectional stream used by workers to report status, loss, and step count; the leader replies with commands such as continue, pause, stop, or reshard.
- `GetAssignment`: a worker blocks until training starts, then receives its shard indices, training config, primary device hint, and initial model weights.
- `ExchangeGradients`: a worker sends sparse or full gradients for one synchronized training step; the leader aggregates, steps the optimizer, and returns a weight delta.
- `GetClusterStatus`: clients such as `dtrain-watch` and the dashboard read current workers, pending workers, global step, loss, epoch, and training phase.
- `AdmitWorkers`: the operator admits pending workers before training or triggers Phase-2 resharding for workers that join mid-run.

## Worker Channels

Each worker opens two gRPC channels to the leader:

- `aio.insecure_channel`: used for `Heartbeat` and `GetAssignment`.
- `grpc.insecure_channel`: used for `ExchangeGradients` from the synchronous training thread.

The split exists because the training loop runs inside `run_in_executor()`. Keeping gradient exchange synchronous inside that thread avoids mixing async calls into the inner training loop while the heartbeat stream stays responsive.

## Training Step Data Flow

Each synchronized step follows this path:

1. Worker runs forward pass.
2. Worker runs `loss.backward()`.
3. Worker applies TopK compression per layer.
4. Worker calls `ExchangeGradients`.
5. Leader waits for all alive assigned workers.
6. Last arriving worker triggers aggregation.
7. Leader computes a weighted average by local batch count.
8. Leader runs `optimizer.step()`.
9. Leader serializes `new_weights - old_weights` as a float16 weight delta.
10. Worker runs `apply_delta()` to update local parameters in place.

With `--topk 0`, workers send full gradients. With `--topk 5000`, each layer sends only the 5000 largest gradient elements by absolute value plus their indices and shape.

## Sharding

Shards are score-proportional. Every worker reports a hardware benchmark score during registration. The leader computes:

```text
worker_fraction = worker_score / sum(all_worker_scores)
worker_samples = int(worker_fraction * train_samples)
```

The base batch size also scales by score fraction with minimum and maximum bounds. Faster workers receive more samples and can receive larger local batches, which helps avoid forcing the cluster to run at the speed of the slowest machine.

## Late Workers And Phase-2 Reshard

Workers that register during an active run are placed into a pending pool. They do not immediately join gradient rounds because that would make existing workers wait for a worker that has not received weights or a shard yet.

When the operator admits a late worker through the dashboard, `dtrain-watch`, or `AdmitWorkers`, the leader performs Phase-2 resharding:

1. Recompute score-proportional shards for all active workers plus the admitted worker.
2. Store pending shard assignments on each worker state.
3. Send a `RESHARD` command on the next heartbeat response.
4. Existing workers rebuild loaders and rejoin gradient rounds.
5. The late worker receives current weights and starts with its assigned shard.

The leader excludes unassigned workers from `_count_alive()` so existing gradient rounds do not block while a new worker is still joining.

## Runtime Surfaces

The same leader state supports three operator surfaces:

- `dtrain-leader`: interactive leader terminal and gRPC server.
- `dtrain-watch`: terminal cluster monitor and admission tool.
- Web dashboard: live status and controls at `http://localhost:8080`.

All three surfaces interact with the same gRPC service and leader state rather than duplicating training logic.
