#!/bin/bash

set_env_para(){
    SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
    WORKSPACE="${SCRIPT_DIR}/../../../"
    export RTP_PATH="${WORKSPACE}/rtp-llm"
    git config --global --add safe.directory ${RTP_PATH}
    export LOG_DIR="${WORKSPACE}/logs"
    [ ! -d ${LOG_DIR} ] && mkdir -p ${LOG_DIR}
    # unset PIP_EXTRA_INDEX_URL
    unset PIP_INDEX_URL
    sed -i 's|^\s*index-url\s*=.*|#&|'  ~/.config/pip/pip.conf
    sed -i 's/^/#/' $RTP_PATH/bazel/bazel_downloader.cfg
    echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
    # tmp 
    #yum --disablerepo="*" --enablerepo=alinux3-updates install -y openblas openblas-devel
    #/opt/conda310/bin/python3 -m pip uninstall -y pytorch_triton_rocm
    if [ "${RTP_CI_QWEN35_397B_ONLY:-0}" = "1" ]; then
        /opt/conda310/bin/python - <<'PY'
import triton
print(f"Initial ROCm Triton: {triton.__version__}")
PY
    else
        /opt/conda310/bin/python -m pip uninstall -y triton
        /opt/conda310/bin/python -m pip install -y triton==3.5.0 --index-url https://pypi.org/simple
    fi
}

record_env() {
    output_file=${LOG_DIR}/env.md
    hostname=$(hostname)
    gcc_version=$(gcc --version | head -n1)
    hipblaslt_version=$(ls /opt/rocm/lib | grep hipblaslt.so | grep -oP '\d+(\.\d+)+')
    cpu_gpu_version=$(rocminfo | grep "Marketing Name:" | cut -d ':' -f2- | sort -u | sed 's/^ *//')
    commit_sha=$(git -C ${RTP_PATH} rev-parse HEAD)
    branch=$(git -C ${RTP_PATH} branch --show-current)
    echo "Hostname: $hostname" > $output_file
    echo "CPU-GPU Version: $cpu_gpu_version" >> $output_file
    echo "Image Name: $IMAGE_NAME" >> $output_file
    echo "GCC Version: $gcc_version" >> $output_file
    echo "HIPBLASLT Version: $hipblaslt_version" >> $output_file
    echo "Git Branch: $branch" >> $output_file
    echo "Git Commit SHA: $commit_sha" >> $output_file
    /opt/conda310/bin/python3 -m pip freeze >> $output_file
}

compile_rtp(){
    cd ${RTP_PATH}
    EXIT_CODE=0
    set -o pipefail
    # yum --disablerepo="*" --enablerepo=alinux3-os install -y patch
    if [ "${RTP_CI_QWEN35_397B_ONLY:-0}" = "1" ]; then
        # Qwen3.5-397B ROCm cases need newer transformers APIs than the image
        # provides, while full requirements_rocm currently hits resolver
        # conflicts. Keep this scoped to this regression job.
        /opt/conda310/bin/python3 -m pip install --no-cache-dir --index-url https://pypi.org/simple \
            "https://sinian-metrics-platform.oss-cn-hangzhou.aliyuncs.com/kis/AMD/triton/triton-3.7.0%2Bamd.rocm7.2.0.gitd0d77a509-cp310-cp310-linux_x86_64.whl" \
            "https://sinian-metrics-platform.oss-cn-hangzhou.aliyuncs.com/kis/AMD/triton/triton_kernels-1.0.0%2Bamd.rocm7.2.0.gitd0d77a509-py3-none-any.whl" \
            watchdog==6.0.0 \
            jsonschema==4.26.0 \
            "https://sinian-metrics-platform.oss-cn-hangzhou.aliyuncs.com/kis/AMD/aiter/aiter-0.1.17.dev79%2Bg2570b35f9.d20260623-cp310-cp310-linux_x86_64.whl" || EXIT_CODE=1
        /opt/conda310/bin/python3 -m pip install --no-cache-dir --no-deps --index-url https://pypi.org/simple \
            transformers==5.2.0 \
            huggingface-hub==1.3.0 \
            hf-xet==1.5.0 \
            tokenizers==0.22.2 \
            safetensors==0.4.4 || EXIT_CODE=1
    else
        /opt/conda310/bin/python3 -m pip install --no-cache-dir -r ./deps/requirements_rocm.txt --index-url https://pypi.org/simple
    fi
    if [ "${RTP_CI_QWEN35_397B_ONLY:-0}" = "1" ]; then
        /opt/conda310/bin/python3 - <<'PY' || EXIT_CODE=1
import importlib.util
missing = [name for name in ("aiter", "watchdog", "jsonschema") if importlib.util.find_spec(name) is None]
if missing:
    raise SystemExit(f"Missing CI runtime dependencies: {', '.join(missing)}")
import triton
triton_version = triton.__version__.split("+", 1)[0]
triton_tuple = tuple(int(part) for part in triton_version.split(".")[:3])
if triton_tuple < (3, 6, 0):
    raise SystemExit(f"ROCm Triton >= 3.6.0 required by aiter, found {triton.__version__}")
import transformers
if transformers.__version__ != "5.2.0":
    raise SystemExit(f"transformers==5.2.0 required by qwen3.5_moe, found {transformers.__version__}")
import transformers.modeling_layers  # noqa: F401
print(f"CI runtime dependencies ready: aiter, watchdog, jsonschema, triton {triton.__version__}, transformers {transformers.__version__}")
PY
    fi
    # /opt/conda310/bin/python3 -m pip install /mnt/raid0/yuzho/BACKUPS/flash_attn-2.8.3-cp310-cp310-linux_x86_64.whl # flash attn whl
    /opt/conda310/bin/python -m pip install ninja -i https://pypi.org/simple/
    if [ "${RTP_CI_QWEN35_397B_ONLY:-0}" = "1" ]; then
        # The image's flash-attn wheel is linked against ROCm 6 (libamdhip64.so.6),
        # but this job runs on ROCm 7.2. Removing it keeps transformers from loading
        # the incompatible extension when qwen3.5_moe imports VIT modules.
        /opt/conda310/bin/python3 -m pip uninstall -y flash-attn flash_attn || true
        /opt/conda310/bin/python3 - <<'PY' || EXIT_CODE=1
from transformers import CLIPVisionModel
print("Transformers CLIP import ready without incompatible flash_attn")
PY
    fi

    # try to build
    bazelisk build //rtp_llm:rtp_llm //rtp_llm/dash_sc/proto:predict_v2_py //rtp_llm/cpp/model_rpc/proto:model_rpc_service_py //rtp_llm/cpp/model_rpc/proto:flexlb_schedule_service_py --jobs 150 --verbose_failures --config=rocm 2>&1 | tee "${LOG_DIR}/bazelbuild.log"
    BUILD_RESULT=$?

    # if build failed and is because timeout, set PIP_TIMEOUT and try again
    if [ $BUILD_RESULT -ne 0 ]; then
        if grep -q -i "timeout\|timed out" "${LOG_DIR}/bazelbuild.log"; then
            echo "bazel build failed due to time out，set PIP_TIMEOUT=300 and try again..."
            export PIP_TIMEOUT=300
            bazelisk build //rtp_llm:rtp_llm //rtp_llm/dash_sc/proto:predict_v2_py //rtp_llm/cpp/model_rpc/proto:model_rpc_service_py //rtp_llm/cpp/model_rpc/proto:flexlb_schedule_service_py --jobs 150 --verbose_failures --config=rocm 2>&1 | tee -a "${LOG_DIR}/bazelbuild.log" || EXIT_CODE=1
        else
            EXIT_CODE=1
        fi
    fi
    bash ${RTP_PATH}/rtp_llm/dash_sc/proto/link_py_proto.sh || true
    bash ${RTP_PATH}/rtp_llm/cpp/model_rpc/proto/link_py_proto.sh || true
    record_env
}

set_env_para
compile_rtp
if [[ "${BASH_SOURCE[0]}" != "${0}" ]]; then
    # 被source调用
    return $EXIT_CODE
else
    # 直接执行
    exit $EXIT_CODE
fi
