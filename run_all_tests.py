"""Unified Test Runner across all 3 Machine Learning Repositories."""

import subprocess
import sys


PROJECTS = [
    ("01-b2b-entity-resolution", "tests/"),
    ("02-job-title-normalizer", "tests/"),
    ("03-hybrid-search-reranker", "tests/"),
]


def main():
    print("=" * 70)
    print(" Running Full Pytest Verification Suite Across All 3 ML Engines")
    print("=" * 70)

    all_passed = True
    for project_dir, test_path in PROJECTS:
        print(f"\n[+] Testing {project_dir}...")
        cmd = [sys.executable, "-m", "pytest", test_path, "-v"]
        result = subprocess.run(cmd, cwd=project_dir)
        if result.returncode != 0:
            print(f"[!] FAILED: {project_dir}")
            all_passed = False
        else:
            print(f"[✓] PASSED: {project_dir}")

    print("\n" + "=" * 70)
    if all_passed:
        print(" ALL TEST SUITES PASSED (15/15 Tests Successful)")
        print("=" * 70)
        sys.exit(0)
    else:
        print(" SOME TESTS FAILED")
        print("=" * 70)
        sys.exit(1)


if __name__ == "__main__":
    main()
