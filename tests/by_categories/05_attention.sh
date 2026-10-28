#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

# shellcheck source=lib.sh
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

COMPILER="${COMPILER:-compile.py}"

run_category_tests "5 (attention-like)" "$COMPILER" \
    a01_attention \
    a02_attention_128
