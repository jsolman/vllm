"""Run the full REBASE-3 validation suite; print a PASS/FAIL table."""
import glob
import os
import subprocess
import sys

ROOT = os.environ.get("REBASE_ROOT", os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
env = dict(os.environ)
env["TRITON_INTERPRET"] = "1"

fails = 0
for t in sorted(glob.glob(os.path.join(ROOT, "tests_local/rebase3_validation/test_*.py"))):
    r = subprocess.run(
        [sys.executable, t], capture_output=True, text=True, env=env, cwd=ROOT
    )
    out = r.stdout.strip().splitlines()
    last = out[-1] if out else "<no output>"
    status = "PASS" if r.returncode == 0 else f"FAIL rc={r.returncode}"
    print(f"{status:10s} {os.path.basename(t):42s} | {last[:70]}")
    if r.returncode != 0:
        fails += 1
        print("  stdout tail:", r.stdout[-400:])
        print("  stderr tail:", r.stderr[-400:])
print("TOTAL FILES:", len(glob.glob(os.path.join(ROOT, 'tests_local/rebase3_validation/test_*.py'))), "FAILS:", fails)
