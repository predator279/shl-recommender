import subprocess
import time
import sys

def main():
    print("=== SHL Recommender: Safe Test Runner (Rate-Limit Avoidance) ===")
    print("Collecting tests...")
    
    # Run pytest collect-only to get all test nodes
    cmd = ["python", "-m", "pytest", "--collect-only", "-q"]
    result = subprocess.run(cmd, capture_output=True, text=True, shell=True)
    
    if result.returncode != 0:
        print("Failed to collect tests:")
        print(result.stderr or result.stdout)
        sys.exit(1)
        
    # Parse the test cases from the output
    test_cases = []
    for line in result.stdout.splitlines():
        line = line.strip()
        if not line or "collected" in line or line.startswith("="):
            continue
        # Typical format: tests/test_schema_compliance.py::test_health_endpoint
        if "::" in line:
            test_cases.append(line)
            
    print(f"Found {len(test_cases)} tests to run.")
    
    failed = []
    passed = 0
    
    for i, test in enumerate(test_cases, start=1):
        print(f"\n[{i}/{len(test_cases)}] Running: {test}")
        
        # Run the single test
        test_cmd = ["python", "-m", "pytest", test, "-v"]
        test_res = subprocess.run(test_cmd, shell=True)
        
        if test_res.returncode == 0:
            print("=> PASSED")
            passed += 1
        else:
            print("=> FAILED")
            failed.append(test)
            
        # If not the last test, sleep 4.5 seconds to avoid Gemini's 15 RPM limit
        if i < len(test_cases):
            print("Sleeping 4.5 seconds to respect rate limits...")
            time.sleep(4.5)
            
    print("\n" + "="*50)
    print("TEST SUMMARY")
    print(f"Total: {len(test_cases)}")
    print(f"Passed: {passed}")
    print(f"Failed: {len(failed)}")
    if failed:
        print("\nFailed Tests:")
        for f in failed:
            print(f"  - {f}")
        sys.exit(1)
    else:
        print("\nAll tests passed successfully!")
        sys.exit(0)

if __name__ == "__main__":
    main()
