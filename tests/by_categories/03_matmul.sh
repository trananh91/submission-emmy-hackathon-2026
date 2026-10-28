#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

# shellcheck source=lib.sh
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

COMPILER="${COMPILER:-compile.py}"

run_category_tests "3 (matmul)" "$COMPILER" \
    m01_matmul_512 \
    m02_matmul_rect
