#!/usr/bin/env bash
# After 'bazel build //rtp_llm/cpp/model_rpc/proto:model_rpc_service_py //rtp_llm/cpp/model_rpc/proto:flexlb_schedule_service_py',
# create symlinks in this directory to the generated *_pb2.py files.
# Run from repo root.
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
cd "$REPO_ROOT"
BAZEL_BIN="$(bazelisk info bazel-bin 2>/dev/null || echo "bazel-out/k8-opt/bin")"
TARGET_DIR="$BAZEL_BIN/rtp_llm/cpp/model_rpc/proto"
for f in model_rpc_service_pb2.py model_rpc_service_pb2_grpc.py flexlb_schedule_service_pb2.py flexlb_schedule_service_pb2_grpc.py; do
  if [ -f "$TARGET_DIR/$f" ]; then
    ln -sf "$(readlink -f "$TARGET_DIR/$f")" "$SCRIPT_DIR/$f"
    echo "linked $SCRIPT_DIR/$f -> $TARGET_DIR/$f"
  else
    echo "skip $f (run: bazel build //rtp_llm/cpp/model_rpc/proto:model_rpc_service_py //rtp_llm/cpp/model_rpc/proto:flexlb_schedule_service_py)" >&2
  fi
done
