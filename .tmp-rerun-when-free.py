#!/usr/bin/env python3
# Fire the PR #1558 failed-job rerun the moment maxibookair24 goes idle,
# then watch the checks to a terminal verdict. (Temp tooling, not committed.)
import json
import subprocess
import time

CWD = "/Users/mboch/Dev/maxi-tools/maxi-ml-worktrees/minicpm5-1558"
PENDING = {"QUEUED", "IN_PROGRESS", "PENDING"}
BAD = {"FAILURE", "ERROR", "CANCELLED"}


def gh(*args, timeout=60):
    return subprocess.run(["gh", *args], capture_output=True, text=True, cwd=CWD, timeout=timeout)


def air24_idle():
    r = gh("api", "/orgs/maxi-tools/actions/runners",
           "--jq", '.runners[] | select(.name=="maxibookair24") | "\\(.status) \\(.busy)"')
    return r.returncode == 0 and r.stdout.strip() == "online false"


def checks_pending():
    r = gh("pr", "checks", "1558", "--json", "name,state")
    if r.returncode not in (0, 8) or not r.stdout.strip():
        return None
    return json.loads(r.stdout)


# Phase 1: wait for the runner to be idle, then rerun failed jobs.
while True:
    if air24_idle():
        rr = gh("run", "rerun", "29137514112", "--failed")
        print(f"RERUN FIRED (rc={rr.returncode}) {rr.stderr.strip()[:120]}", flush=True)
        break
    time.sleep(60)

# Phase 2: watch to terminal verdict.
time.sleep(120)
while True:
    checks = checks_pending()
    if checks and not any(c["state"] in PENDING for c in checks):
        for c in checks:
            if c["state"] not in ("SUCCESS", "SKIPPED"):
                print(f'{c["name"]}: {c["state"]}', flush=True)
        bad = [c for c in checks if c["state"] in BAD]
        print("VERDICT_FAILURES" if bad else "VERDICT_GREEN", flush=True)
        break
    time.sleep(120)
