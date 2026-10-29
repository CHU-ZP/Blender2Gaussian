#!/usr/bin/env bash
# Python packages are managed by uv; only the CUDA compiler is external.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
command -v uv >/dev/null || { echo "uv is required: https://docs.astral.sh/uv/" >&2; exit 1; }

if [[ -n "${CUDA_HOME:-}" ]]; then
    cuda_candidates=("$CUDA_HOME/bin/nvcc")
else
    shopt -s nullglob
    cuda_candidates=(
        /usr/local/cuda-12.8/bin/nvcc
        /usr/local/cuda/bin/nvcc
    )
    if command -v nvcc >/dev/null; then
        cuda_candidates+=("$(command -v nvcc)")
    fi
    cuda_candidates+=(
        "$HOME"/miniconda3/envs/*/bin/nvcc
        "$HOME"/anaconda3/envs/*/bin/nvcc
    )
fi

selected_nvcc=""
for candidate in "${cuda_candidates[@]}"; do
    if [[ -x "$candidate" ]] && "$candidate" --version | grep -q 'release 12\.8,'; then
        selected_nvcc="$candidate"
        break
    fi
done
if [[ -z "$selected_nvcc" ]]; then
    echo "CUDA Toolkit 12.8 is required to build _C for torch 2.11.0+cu128." >&2
    echo "Set CUDA_HOME to its root directory, then run this script again." >&2
    echo "For the Python reference only: uv sync --locked --no-install-project" >&2
    exit 1
fi

export CUDA_HOME
CUDA_HOME="$(cd "$(dirname "$selected_nvcc")/.." && pwd)"
export PATH="$CUDA_HOME/bin:$PATH"
# Keep compilation within the memory budget of a laptop.
export MAX_JOBS="${MAX_JOBS:-1}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.9}"

echo "CUDA compiler: $selected_nvcc"
echo "CUDA architectures: $TORCH_CUDA_ARCH_LIST (RTX 4060 Laptop: 8.9)"
uv sync --locked "$@"
uv run --no-sync python scripts/check_environment.py
