#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec uv run --project "$project_dir" --locked --no-dev --group render python "$project_dir/scripts/run_batch.py" "$@"
