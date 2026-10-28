#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

# shellcheck source=lib.sh
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

COMPILER="${COMPILER:-compile.py}"

run_category_tests "4 (fused)" "$COMPILER" \
    f01_rmsnorm \
    f02_row_softmax \
    f03_l2norm
