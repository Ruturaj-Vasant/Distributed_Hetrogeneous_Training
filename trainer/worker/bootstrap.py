"""
trainer/worker/bootstrap.py — stdlib-only venv + proto setup.

Called before any third-party imports so it can create the venv and
install dependencies if this is a fresh clone. Uses ONLY Python stdlib.
"""


def bootstrap() -> None:
    import os, subprocess, sys, hashlib
    from pathlib import Path

    proj = Path(__file__).resolve().parent.parent.parent  # project root
    venv = proj / ".venv"
    is_win = sys.platform == "win32"
    venv_py = venv / ("Scripts/python.exe" if is_win else "bin/python")

    # Already inside a venv — nothing to do.
    if sys.prefix != sys.base_prefix:
        return

    if not venv_py.exists():
        print("[bootstrap] Creating .venv …", flush=True)
        subprocess.run([sys.executable, "-m", "venv", str(venv)], check=True)

    req_file   = proj / "requirements.txt"
    stamp_file = venv / ".deps_hash"

    req_hash = hashlib.md5(req_file.read_bytes()).hexdigest() if req_file.exists() else ""
    cached   = stamp_file.read_text().strip() if stamp_file.exists() else ""

    probe_ok = subprocess.run(
        [str(venv_py), "-c", "import torch, torchvision, grpc, psutil, numpy"],
        capture_output=True,
    ).returncode == 0

    if not probe_ok or cached != req_hash:
        print("[bootstrap] Installing dependencies …", flush=True)
        subprocess.run(
            [str(venv_py), "-m", "pip", "install", "-r", str(req_file), "-q"],
            check=True,
        )
        stamp_file.write_text(req_hash)

    proto_dir = proj / "proto"
    if not (proto_dir / "trainer_pb2.py").exists():
        print("[bootstrap] Generating gRPC proto stubs …", flush=True)
        subprocess.run(
            [
                str(venv_py), "-m", "grpc_tools.protoc",
                f"--proto_path={proto_dir}",
                f"--python_out={proto_dir}",
                f"--grpc_python_out={proto_dir}",
                str(proto_dir / "trainer.proto"),
            ],
            check=True,
        )
        grpc_f = proto_dir / "trainer_pb2_grpc.py"
        if grpc_f.exists():
            grpc_f.write_text(
                grpc_f.read_text().replace(
                    "import trainer_pb2", "from . import trainer_pb2", 1
                )
            )
        init = proto_dir / "__init__.py"
        if not init.exists():
            init.write_text("")

    print("[bootstrap] Launching inside .venv …", flush=True)
    if is_win:
        result = subprocess.run([str(venv_py)] + sys.argv)
        sys.exit(result.returncode)
    else:
        os.execv(str(venv_py), [str(venv_py)] + sys.argv)
