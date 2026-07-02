"""
Run every module's own unit tests (python -m src.<module>) in one pass.

Each module under src/ is independently runnable and self-testing (see the
test_* functions + `if __name__ == "__main__"` block at the bottom of each
file). This script just calls all of them and aggregates the result, so
there's one command to check the whole codebase:

    python scripts/run_all_unit_tests.py
"""
import subprocess
import sys
import os

MODULES = ["schema", "transforms", "episode", "entities", "tokenizer", "dataset", "model"]


def main():
    repo_root = os.path.join(os.path.dirname(__file__), "..")
    results = {}
    for mod in MODULES:
        print(f"\n{'=' * 60}\nsrc.{mod}\n{'=' * 60}")
        proc = subprocess.run(
            [sys.executable, "-m", f"src.{mod}"],
            cwd=repo_root, capture_output=True, text=True,
        )
        print(proc.stdout)
        if proc.returncode != 0:
            print(proc.stderr)
        results[mod] = proc.returncode == 0

    print(f"\n{'=' * 60}\nSUMMARY\n{'=' * 60}")
    for mod, ok in results.items():
        print(f"  {'PASS' if ok else 'FAIL'}  src.{mod}")
    if all(results.values()):
        print(f"\nall {len(results)} modules passed.")
    else:
        failed = [m for m, ok in results.items() if not ok]
        print(f"\n{len(failed)} module(s) FAILED: {failed}")
        sys.exit(1)


if __name__ == "__main__":
    main()
