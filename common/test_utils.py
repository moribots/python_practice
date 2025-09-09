# common/test_utils.py
from __future__ import annotations
import sys
import os

# Global debug flag - can be set by test runner
DEBUG_MODE = False


def _debug_print(*args, **kwargs) -> None:
    """Print debug information to stderr when debug mode is enabled."""
    if DEBUG_MODE or os.environ.get('PYTHON_PRACTICE_DEBUG', '').lower() in ('1', 'true', 'yes'):
        print("[DEBUG]", *args, file=sys.stderr, **kwargs)


def _pass(problem_num: int, suite: str) -> None:
    """Standard success line; keep exact format for the unified runner."""
    message = f"✅ Problem {problem_num} ({suite}) Passed"
    print(message)
    if DEBUG_MODE:
        print(f"[DEBUG] {message}", file=sys.stderr)


def _fail(problem_num: int, suite: str, e: Exception) -> None:
    """Standard failure line; keep exact format for the unified runner."""
    message = f"❌ Problem {problem_num} ({suite}) Failed: {e}"
    print(message)
    if DEBUG_MODE:
        print(f"[DEBUG] {message}", file=sys.stderr)


def set_debug_mode(enabled: bool) -> None:
    """Set global debug mode for test utilities."""
    global DEBUG_MODE
    DEBUG_MODE = enabled


# Convenient alias for debug printing that can be used in test files
debug_print = _debug_print
