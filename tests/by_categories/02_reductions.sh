#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

# shellcheck source=lib.sh
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

COMPILER="${COMPILER:-compile.py}"

run_category_tests "2 (reductions)" "$COMPILER" \
    r01_array_sum \
    r02_big_sum \
    r03_mean \
    r04_row_sum
