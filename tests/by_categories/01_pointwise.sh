#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

# shellcheck source=lib.sh
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

COMPILER="${COMPILER:-compile.py}"

run_category_tests "1 (pointwise)" "$COMPILER" \
    p00_scalar_add \
    p01_vector_add \
    p02_scale_shift \
    p03_relu_mul
