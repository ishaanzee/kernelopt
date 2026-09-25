"""Benchmark hygiene: wait until competing GPU jobs (by process-name pattern) are gone, and flag measurements
that overlapped one so they can be retried."""
from __future__ import annotations

import subprocess
import threading
import time


def _running(patterns) -> bool:
    for p in patterns:
        if subprocess.run(["pgrep", "-f", p], capture_output=True).returncode == 0:
            return True
    return False


def wait_quiet(patterns, settle_s=5.0, poll_s=1.0, log=print):
    if not patterns:
        return
    announced = False
    quiet_since = None
    while True:
        if _running(patterns):
            quiet_since = None
            if not announced:
                log(f"  (waiting for competing jobs matching {patterns} to finish)")
                announced = True
        else:
            quiet_since = quiet_since or time.monotonic()
            if time.monotonic() - quiet_since >= settle_s:
                return
        time.sleep(poll_s)


class Guard:
    """with Guard(patterns) as g: ...measure...;  g.contaminated tells whether a competing job ran meanwhile."""

    def __init__(self, patterns, poll_s=0.5):
        self.patterns, self.poll_s = patterns, poll_s
        self.contaminated = False
        self._stop = threading.Event()

    def _watch(self):
        while not self._stop.is_set():
            if _running(self.patterns):
                self.contaminated = True
            self._stop.wait(self.poll_s)

    def __enter__(self):
        if self.patterns:
            self._t = threading.Thread(target=self._watch, daemon=True)
            self._t.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        if self.patterns:
            self._t.join()
            self.contaminated |= _running(self.patterns)
        return False


def measured(fn, patterns, retries=5, log=print):
    """Runs fn() on a quiet machine; retries if a competing job appeared during the run."""
    for attempt in range(retries):
        wait_quiet(patterns, log=log)
        with Guard(patterns) as g:
            result = fn()
        if not g.contaminated:
            return result
        log("  (measurement overlapped a competing job; retrying)")
    return result
