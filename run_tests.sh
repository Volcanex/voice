#!/bin/bash
# Script to run all tests for the voice assistant

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
BOLD='\033[1m'
NC='\033[0m' # No Color

echo -e "${YELLOW}${BOLD}===== Running Voice Assistant Tests =====${NC}"

# Ensure working directory is the project root
cd "$(dirname "$0")"

# Ensure test directories exist
mkdir -p ./test_models/asr
mkdir -p ./test_models/llm
mkdir -p ./test_models/csm

# Function to run tests and display summary
run_test() {
    local module=$1
    local test_file=$2
    
    echo -e "\n${YELLOW}${BOLD}Running $module module tests...${NC}"
    
    # Run the test and capture output and exit code
    TEST_OUTPUT=$(python3 -m pytest $test_file -v)
    TEST_EXIT_CODE=$?
    
    # Extract test summary
    TEST_SUMMARY=$(echo "$TEST_OUTPUT" | grep -E "=+ .* =+$" | tail -n 1)
    PASSED=$(echo "$TEST_SUMMARY" | grep -oP '\d+(?= passed)')
    FAILED=$(echo "$TEST_SUMMARY" | grep -oP '\d+(?= failed)')
    SKIPPED=$(echo "$TEST_SUMMARY" | grep -oP '\d+(?= skipped)')
    
    # Format summary output
    if [ -n "$PASSED" ] && [ "$PASSED" -gt 0 ]; then
        echo -e "  ${GREEN}✓ $PASSED tests passed${NC}"
    fi
    
    if [ -n "$FAILED" ] && [ "$FAILED" -gt 0 ]; then
        echo -e "  ${RED}✗ $FAILED tests failed${NC}"
    fi
    
    if [ -n "$SKIPPED" ] && [ "$SKIPPED" -gt 0 ]; then
        echo -e "  ${BLUE}⚠ $SKIPPED tests skipped${NC}"
    fi
    
    # Display full output
    echo -e "\n$TEST_OUTPUT"
    
    # Return test exit code
    return $TEST_EXIT_CODE
}

# Track overall status
OVERALL_STATUS=0

# Run each test module
run_test "ASR" "tests/test_asr.py" || OVERALL_STATUS=1
run_test "LLM" "tests/test_llm.py" || OVERALL_STATUS=1
run_test "CSM" "tests/test_csm.py" || OVERALL_STATUS=1
run_test "End-to-End" "tests/test_e2e.py" || OVERALL_STATUS=1
run_test "ConnectionManager" "tests/test_connection_manager.py" || OVERALL_STATUS=1
run_test "ServerDialog" "tests/test_server_dialog.py" || OVERALL_STATUS=1
run_test "ClientIntegration" "tests/test_client_integration.py" || OVERALL_STATUS=1

# Display overall test summary
if [ $OVERALL_STATUS -eq 0 ]; then
    echo -e "\n${GREEN}${BOLD}===== All tests completed successfully! =====${NC}"
else
    echo -e "\n${RED}${BOLD}===== Some tests failed! =====${NC}"
    exit 1
fi