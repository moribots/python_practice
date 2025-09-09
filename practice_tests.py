#!/usr/bin/env python3
"""
Unified test runner for the python_practice repo.

Features
--------
- Suite selection: --suites einops python pandas pytorch numpy rl robotics (default: all)
- RNG seeding for reproducibility: --seed 0
- Result aggregation by parsing per-suite stdout (✅/❌ lines you already print)
- Timing per case and per suite
- JSON/CSV output: --json-out results.json --csv-out results.csv
- Pattern / range filters applied to parsed cases (problems still run inside suite)
- Non-zero exit code on failures by default (disable with --no-exit-nonzero)

It does NOT change problem mechanics; it wraps existing test entry points.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import io
import json
import os
import re
import sys
import time
from dataclasses import asdict, dataclass
from typing import Callable, Dict, List, Optional, Tuple

# --- Import your existing test entry points (these already exist in your repo) ---
# Repo structure (abbrev): practice_tests.py; einops_practice_dir/test_einops.py; python/test_python.py; etc.
# We rely on their public 'test_*' functions as suite entrypoints.
# :contentReference[oaicite:1]{index=1}
from einops_practice_dir.test_einops import test_einops
# :contentReference[oaicite:2]{index=2}
from python.test_python import test_python
# :contentReference[oaicite:3]{index=3}
from pandas_practice_dir.test_pandas import test_pandas
# :contentReference[oaicite:4]{index=4}
from pytorch.test_pytorch import test_pytorch
# :contentReference[oaicite:5]{index=5}
from numpy_practice_dir.test_numpy import test_numpy
# :contentReference[oaicite:6]{index=6}
from reinforcement_learning.test_reinforcement_learning import test_rl
# :contentReference[oaicite:7]{index=7}
from robotics_fundamentals.test_robotics import test_robotics
from common.test_utils import set_debug_mode

# ------------------------
# Data model for a test case
# ------------------------


@dataclass
class CaseResult:
    suite: str
    # Parsed from "Problem N"; None if not detected
    problem_id: Optional[int]
    name: str                        # "Problem N (Suite) ..."
    status: str                      # "pass" | "fail"
    duration_s: float
    error: Optional[str] = None      # message if fail
    tags: List[str] = None           # coarse topical tags

    def to_row(self) -> List[str]:
        return [
            self.suite,
            str(self.problem_id) if self.problem_id is not None else "",
            self.name,
            self.status,
            f"{self.duration_s:.6f}",
            self.error or "",
            ",".join(self.tags or []),
        ]


# ------------------------
# Suite registry
# ------------------------
SUITE_FUNCS: Dict[str, Callable[[], None]] = {
    "einops": test_einops,
    "python": test_python,
    "pandas": test_pandas,
    "pytorch": test_pytorch,
    "numpy": test_numpy,
    "rl": test_rl,
    "robotics": test_robotics,
}

DEFAULT_ORDER = list(SUITE_FUNCS.keys())

# ------------------------
# Utilities
# ------------------------
PASS_RE = re.compile(r"^\s*✅\s+Problem\s+(\d+)\s+\(([^)]+)\)\s+Passed\s*$")
FAIL_RE = re.compile(
    r"^\s*❌\s+Problem\s+(\d+)\s+\(([^)]+)\)\s+Failed:\s*(.*)$")


def infer_tags(suite: str, problem_id: Optional[int]) -> List[str]:
    # Coarse, interview-relevant tags
    base = {
        "einops": ["einops", "einsum"],
        "python": ["dsa", "algorithms", "python"],
        "pandas": ["pandas", "data"],
        "pytorch": ["pytorch", "ml"],
        "numpy": ["numpy", "linear-algebra"],
        "rl": ["rl", "robot-learning"],
        "robotics": ["robotics", "planning", "control"],
    }.get(suite.lower(), [])
    # minor specialization: einops problems 11+ focus einsum
    if suite.lower() == "einops" and problem_id and problem_id >= 11:
        if "einsum" not in base:
            base.append("einsum")
    return base


def set_seed(seed: Optional[int]):
    if seed is None:
        return
    try:
        import random
        random.seed(seed)
    except Exception:
        pass
    try:
        import numpy as np
        np.random.seed(seed)
    except Exception:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def parse_suite_output(suite: str, stdout: str, elapsed: float) -> List[CaseResult]:
    """
    Parse lines printed by the suite test function.
    We look for:
      "✅ Problem {n} (Suite) Passed"
      "❌ Problem {n} (Suite) Failed: {message}"
    Unknown lines are ignored. Per-case timing isn't emitted by the suites,
    so we attribute a per-case time by dividing total elapsed among the parsed cases.
    If you later add explicit ⏱ prints per problem, we can refine this parser.
    """
    lines = stdout.splitlines()
    # (status, n, suite, error)
    raw: List[Tuple[str, int, str, Optional[str]]] = []
    for ln in lines:
        m = PASS_RE.match(ln)
        if m:
            n = int(m.group(1))
            raw.append(("pass", n, suite, None))
            continue
        m = FAIL_RE.match(ln)
        if m:
            n = int(m.group(1))
            err = m.group(3).strip()
            raw.append(("fail", n, suite, err))
            continue

    # Distribute elapsed across cases (rough but useful)
    per_case = elapsed / len(raw) if raw else 0.0
    results: List[CaseResult] = []
    for status, n, suite_name, err in raw:
        results.append(
            CaseResult(
                suite=suite_name.lower(),
                problem_id=n,
                name=f"Problem {n} ({suite_name})",
                status=status,
                duration_s=per_case,
                error=err,
                tags=infer_tags(suite_name, n),
            )
        )
    return results


def run_one_suite(suite: str) -> Tuple[List[CaseResult], str]:
    """
    Execute a single suite by capturing stdout/stderr of its test function.
    Returns (results, raw_captured_text).
    """
    from common.test_utils import DEBUG_MODE

    fn = SUITE_FUNCS[suite]
    buf = io.StringIO()
    errbuf = io.StringIO()
    t0 = time.perf_counter()

    # In debug mode, don't redirect stderr so debug prints are visible
    if DEBUG_MODE:
        with contextlib.redirect_stdout(buf):
            try:
                fn()
            except SystemExit:
                # In case a suite tries to exit; we still want to parse its output.
                pass
    else:
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(errbuf):
            try:
                fn()
            except SystemExit:
                # In case a suite tries to exit; we still want to parse its output.
                pass

    elapsed = time.perf_counter() - t0
    out = buf.getvalue()
    err = errbuf.getvalue()
    merged = (out + ("\n" + err if err else "")).strip()
    results = parse_suite_output(suite, out, elapsed)
    return results, merged


def filter_results(results: List[CaseResult],
                   problem_min: Optional[int],
                   problem_max: Optional[int],
                   pattern: Optional[str]) -> List[CaseResult]:
    out = []
    rx = re.compile(pattern) if pattern else None
    for r in results:
        if problem_min is not None and (r.problem_id is None or r.problem_id < problem_min):
            continue
        if problem_max is not None and (r.problem_id is None or r.problem_id > problem_max):
            continue
        if rx and not rx.search(r.name):
            continue
        out.append(r)
    return out


def summarize(all_results: List[CaseResult]) -> Dict[str, Dict[str, int]]:
    summary: Dict[str, Dict[str, int]] = {}
    for r in all_results:
        suite = r.suite
        if suite not in summary:
            summary[suite] = {"pass": 0, "fail": 0, "total": 0}
        summary[suite]["total"] += 1
        summary[suite][r.status] += 1
    return summary


def write_json(path: str, all_results: List[CaseResult], summary: Dict[str, Dict[str, int]]):
    payload = {
        "results": [asdict(r) for r in all_results],
        "summary": summary,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def write_csv(path: str, all_results: List[CaseResult]):
    headers = ["suite", "problem_id", "name", "status",
               "duration_s", "error", "tags"]
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(headers)
        for r in all_results:
            w.writerow(r.to_row())


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Unified test runner for python_practice")
    parser.add_argument(
        "--suites",
        nargs="+",
        choices=DEFAULT_ORDER,
        default=DEFAULT_ORDER,
        help="Which suites to run (default: all)",
    )
    parser.add_argument("--seed", type=int, default=None, help="Set RNG seed")
    parser.add_argument("--problem-min", type=int, default=None,
                        help="Filter: minimum problem number (inclusive)")
    parser.add_argument("--problem-max", type=int, default=None,
                        help="Filter: maximum problem number (inclusive)")
    parser.add_argument("--pattern", type=str, default=None,
                        help="Regex to filter case names (after running suite)")
    parser.add_argument("--json-out", type=str, default=None,
                        help="Write full JSON report to this path")
    parser.add_argument("--csv-out", type=str, default=None,
                        help="Write CSV report to this path")
    parser.add_argument(
        "--no-exit-nonzero",
        action="store_true",
        help="If set, always exit 0 even if there are failures",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug output (prints to stderr, visible during test runs)",
    )
    args = parser.parse_args(argv)

    # Enable debug mode if requested
    if args.debug:
        set_debug_mode(True)
        print("Debug mode enabled - debug output will appear on stderr",
              file=sys.stderr)

    set_seed(args.seed)

    all_results: List[CaseResult] = []
    suite_texts: Dict[str, str] = {}

    for suite in args.suites:
        print(f"\n=== Running suite: {suite} ===")
        results, captured = run_one_suite(suite)
        suite_texts[suite] = captured
        results = filter_results(
            results, args.problem_min, args.problem_max, args.pattern)
        # print a compact summary line
        passed = sum(1 for r in results if r.status == "pass")
        failed = sum(1 for r in results if r.status == "fail")
        print(f"[{suite}] parsed cases: {len(results)}  ✅ {passed}  ❌ {failed}")
        all_results.extend(results)

    # Global summary
    print("\n=== Summary ===")
    summary = summarize(all_results)
    total_pass = total_fail = 0
    for suite in args.suites:
        s = summary.get(suite, {"pass": 0, "fail": 0, "total": 0})
        print(
            f"{suite:>11}: total={s['total']:3d}  ✅ {s['pass']:3d}  ❌ {s['fail']:3d}")
        total_pass += s["pass"]
        total_fail += s["fail"]
    print(f"{'ALL':>11}: total={len(all_results):3d}  ✅ {total_pass:3d}  ❌ {total_fail:3d}")

    # Artifacts
    if args.json_out:
        write_json(args.json_out, all_results, summary)
        print(f"\n📝 Wrote JSON to: {args.json_out}")
    if args.csv_out:
        write_csv(args.csv_out, all_results)
        print(f"🧾 Wrote CSV to:  {args.csv_out}")

    if args.no_exit_nonzero:
        return 0
    return 0 if total_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
