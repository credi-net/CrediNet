#!/bin/bash
# test suite for CrediGraph API
# tests: Spec validation + Contract testing + Client testing

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

# 1: Validate OpenAPI Spec
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

# 2: Test Python Client
print_header "[2/3] Testing Python client"

if [ ! -f "../.venv/bin/activate" ]; then
    print_error "Virtual environment not found at ../.venv/bin/activate"
    echo "   Create with: python -m venv .venv"
    exit 1
fi

source ../.venv/bin/activate

if python test_api.py; then
    print_success "Python client tests passed"
else
    print_error "Python client tests failed"
    exit 1
fi

# 3: Contract Testing
print_header "[3/3] Contract testing (API vs Spec)"

if command -v schemathesis &> /dev/null; then
    API_URL="https://credi-net-credinet.hf.space"
    echo "Testing API: $API_URL"
    echo ""
    
    if schemathesis run ../openapi.yaml --url="$API_URL" --max-examples=50 --exclude-checks=unsupported_method; then
        print_success "Contract tests passed (API matches spec)"
    else
        print_error "Contract tests failed (API does not match spec)"
        exit 1
    fi
else
    echo "[SKIP] schemathesis not found, skipping contract testing"
    echo "   Install with: pip install schemathesis"
    echo "   Then run: schemathesis run ../openapi.yaml --url=https://credi-net-credinet.hf.space"
fi

# Final summary
print_header "Test suite complete"
print_success "All tests passed"
echo ""
echo "Next steps:"
echo "  - Before release: tests/test_all.sh"
echo "  - When API changes: schemathesis run ../openapi.yaml --url=https://credi-net-credinet.hf.space"
echo "  - When client changes: python tests/test_api.py"
echo ""
