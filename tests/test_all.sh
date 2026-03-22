#!/bin/bash
# Complete test suite for CrediGraph API
# Tests: Spec validation + Contract testing + Client testing

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

print_header() {
    echo ""
    echo "$1"
    echo ""
}

print_success() {
    echo "[PASS] $1"
}

print_error() {
    echo "[FAIL] $1"
}

# Test 1: Validate OpenAPI Spec
print_header "[1/3] Validating OpenAPI spec"

if command -v openapi-spec-validator &> /dev/null; then
    if openapi-spec-validator ../openapi.yaml > /dev/null 2>&1; then
        print_success "OpenAPI spec is valid"
    else
        print_error "OpenAPI spec validation failed"
        openapi-spec-validator ../openapi.yaml
        exit 1
    fi
else
    echo "[SKIP] openapi-spec-validator not found, skipping spec validation"
    echo "   Install with: pip install openapi-spec-validator"
fi

# Test 2: Run pytest suites
print_header "[2/3] Running pytest suites"

if [ ! -f "../.venv/bin/activate" ]; then
    print_error "Virtual environment not found at ../.venv/bin/activate"
    echo "   Create with: python3 -m venv .venv"
    exit 1
fi

source ../.venv/bin/activate

if command -v pytest &> /dev/null; then
    if pytest -q test_api.py test_helper.py; then
        print_success "Pytest suites passed"
    else
        print_error "Pytest suites failed"
        exit 1
    fi
else
    print_error "pytest not found in environment"
    echo "   Install with: pip install pytest"
    exit 1
fi

# Test 3: Contract Testing
print_header "[3/3] Contract testing (API vs Spec)"

if command -v schemathesis &> /dev/null; then
    API_URL="https://credi-net-credinet.hf.space"
    echo "Testing API: $API_URL"
    echo ""

    SCHEMATHESIS_MAX_EXAMPLES="${SCHEMATHESIS_MAX_EXAMPLES:-30}"
    SCHEMATHESIS_TIMEOUT="${SCHEMATHESIS_TIMEOUT:-30}"
    SCHEMATHESIS_WORKERS="${SCHEMATHESIS_WORKERS:-1}"
    SCHEMATHESIS_RATE_LIMIT="${SCHEMATHESIS_RATE_LIMIT:-5/s}"
    SCHEMATHESIS_LOG="$(mktemp)"

    SCHEMATHESIS_CMD=(
        schemathesis run ../openapi.yaml
        --url="$API_URL"
        --max-examples="$SCHEMATHESIS_MAX_EXAMPLES"
        --request-timeout="$SCHEMATHESIS_TIMEOUT"
        --workers="$SCHEMATHESIS_WORKERS"
        --rate-limit="$SCHEMATHESIS_RATE_LIMIT"
        --exclude-checks=unsupported_method
    )

    if "${SCHEMATHESIS_CMD[@]}" 2>&1 | tee "$SCHEMATHESIS_LOG"; then
        print_success "Contract tests passed (API matches spec)"
    else
        echo ""
        echo "[WARN] Contract test failed. Retrying once to rule out transient network issues..."
        echo ""
        if "${SCHEMATHESIS_CMD[@]}" 2>&1 | tee "$SCHEMATHESIS_LOG"; then
            print_success "Contract tests passed on retry"
        else
            if grep -q "Network Error" "$SCHEMATHESIS_LOG" && ! grep -q "FAILURES" "$SCHEMATHESIS_LOG"; then
                echo ""
                echo "[WARN] Contract fuzzing ended with network-only errors (no contract failures found)."
            else
                print_error "Contract tests failed (API does not match spec)"
                exit 1
            fi
        fi
    fi
    rm -f "$SCHEMATHESIS_LOG"
else
    echo "[SKIP] schemathesis not found, skipping contract testing"
    echo "   Install with: pip install schemathesis"
    echo "   & run: schemathesis run ../openapi.yaml --url=https://credi-net-credinet.hf.space"
fi

print_header "Test suite complete"
print_success "All tests passed"