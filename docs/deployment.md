# Deployment

## 1. Install And Authenticate Tailscale

Install Tailscale on both the leader Mac and the worker machine, then sign in to the same tailnet.

On each machine, verify the leader DNS name:

```bash
tailscale ping leader-macbook-pro.taila5426e.ts.net
```

The worker commands in this project always use:

```text
leader-macbook-pro.taila5426e.ts.net
```

## 2. Install On The Leader Mac

Clone the repository and install the package:

```bash
git clone https://github.com/Ruturaj-Vasant/Distributed_Hetrogeneous_Training.git
cd Distributed_Hetrogeneous_Training
pip install -e ".[dev]"
```

If you already have the local working copy:

```bash
cd /Users/ruturaj_vasant/Desktop/Academic/AI_New/distributed-resnet
pip install -e ".[dev]"
```

## 3. Install On The Windows Worker

Clone the repository from PowerShell:

```powershell
git clone https://github.com/Ruturaj-Vasant/Distributed_Hetrogeneous_Training.git $HOME\distributed-resnet
cd $HOME\distributed-resnet
pip install -e ".[dev]"
```

For CUDA PyTorch installs, use the same PyTorch index URL pattern as `scripts/worker_windows.sh`. For CUDA 12.1:

```powershell
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

For CPU-only testing:

```powershell
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

Then reinstall the local package if needed:

```powershell
pip install -e ".[dev]"
```

## 4. Generate Proto Stubs

After editing `proto/trainer.proto`, regenerate Python stubs from the repo root:

```bash
bash scripts/generate_proto.sh
```

Run this on any clone where the generated stubs are missing or stale.

## 5. Run The Leader

On the leader Mac:

```bash
cd /Users/ruturaj_vasant/Desktop/Academic/AI_New/distributed-resnet
dtrain-leader --model resnet18 --epochs 5 --lr 0.01 --batch-size 32 --topk 0
```

The gRPC server listens on port `50051`. The dashboard starts at:

```text
http://localhost:8080
```

Use `--dashboard-port 9090` to choose another dashboard port, or `--no-dashboard` for headless runs.

## 6. Run Workers

On the leader Mac or another Mac:

```bash
cd ~/distributed-resnet
dtrain-worker --leader leader-macbook-pro.taila5426e.ts.net
```

On Windows PowerShell:

```powershell
cd $HOME\distributed-resnet
dtrain-worker --leader leader-macbook-pro.taila5426e.ts.net
```

Back in the leader terminal, type:

```text
start
```

Optional monitor:

```bash
dtrain-watch --leader leader-macbook-pro.taila5426e.ts.net
```

## Troubleshooting

### Port Not Reachable

Check Tailscale connectivity:

```bash
tailscale ping leader-macbook-pro.taila5426e.ts.net
nc -zv leader-macbook-pro.taila5426e.ts.net 50051
```

Make sure `dtrain-leader` is running and that no firewall is blocking port `50051`.

### MPS Not Available

On macOS, verify PyTorch can see MPS:

```bash
python -c "import torch; print(torch.backends.mps.is_available())"
```

If this prints `False`, update macOS/PyTorch or run with CPU for protocol testing using:

```bash
dtrain-worker --leader leader-macbook-pro.taila5426e.ts.net --dry-run
```

### Proto Stubs Missing

If imports such as `proto.trainer_pb2` fail, regenerate stubs:

```bash
bash scripts/generate_proto.sh
```

Then reinstall the package:

```bash
pip install -e ".[dev]"
```
