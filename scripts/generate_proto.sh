#!/usr/bin/env bash
# Generate Python gRPC stubs from proto/trainer.proto.
# Run once after installing requirements, or after editing the .proto file.
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PROTO_DIR="${PROJECT_DIR}/proto"
PYTHON="${PYTHON:-python3}"

"${PYTHON}" -m grpc_tools.protoc \
  --proto_path="${PROTO_DIR}" \
  --python_out="${PROTO_DIR}" \
  --grpc_python_out="${PROTO_DIR}" \
  "${PROTO_DIR}/trainer.proto"

# grpc_tools emits an absolute import; patch to package-relative.
GRPC_FILE="${PROTO_DIR}/trainer_pb2_grpc.py"
if grep -q "^import trainer_pb2" "${GRPC_FILE}"; then
  # Use Python itself to do the replacement — avoids sed -i portability issues
  "${PYTHON}" - "${GRPC_FILE}" <<'PYEOF'
import pathlib, re, sys
p = pathlib.Path(sys.argv[1])
p.write_text(re.sub(r'^import trainer_pb2', 'from . import trainer_pb2', p.read_text(), flags=re.MULTILINE))
PYEOF
fi

touch "${PROTO_DIR}/__init__.py"
echo "Proto stubs generated in proto/"
