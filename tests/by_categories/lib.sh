# Shared helpers for category test scripts. Source from 01_pointwise.sh, etc.

run_category_tests() {
    local category_label="$1"
    local compiler="${2:-compile.py}"
    shift 2
    local problems=("$@")

    local passed=0
    local failed=0
    local -a failed_problems=()

    for problem in "${problems[@]}"; do
        echo "---- $problem ----"
        if python submit.py "$problem" --compiler "$compiler"; then
            passed=$((passed + 1))
        else
            failed=$((failed + 1))
            failed_problems+=("$problem")
        fi
    done

    local total=$((passed + failed))
    echo ""
    echo "=== category $category_label: $passed/$total passed ==="
    if ((failed > 0)); then
        echo "Failed: ${failed_problems[*]}"
        return 1
    fi
    return 0
}
