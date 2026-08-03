# Copyright (C) 2026, Tactics2D Authors. Released under the GNU GPLv3.
# SPDX-License-Identifier: GPL-3.0-or-later

"""System resource monitoring implementation."""

from __future__ import annotations

from contextlib import contextmanager


def _gpu_mem_process_mb() -> float:
    """Per-process GPU memory (MB), via nvidia-smi compute-apps or torch.

    nvidia-smi ``memory.used`` reports machine-wide usage; ``--query-compute-apps``
    reports per-process allocation, which is filtered to the current PID.  Falls
    back to PyTorch's ``memory_allocated`` when the per-process query is
    unavailable.  Returns 0.0 when no GPU context exists for this process.
    """
    import os

    pid = os.getpid()
    try:
        import subprocess

        r = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,used_gpu_memory",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=2,
        )
        total = 0.0
        for line in r.stdout.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) == 2 and parts[0].isdigit() and int(parts[0]) == pid:
                total += float(parts[1])
        if total > 0.0:
            return total
    except Exception:
        pass
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / 1024**2
    except Exception:
        pass
    return 0.0


def get_resource_stats() -> dict[str, float]:
    """Return current CPU% (0-100), memory MB, and GPU memory MB."""
    cpu = mem = gpu = 0.0
    try:
        import psutil

        p = psutil.Process()
        cpu = p.cpu_percent(interval=0)
        mem = p.memory_info().rss / 1024**2
    except Exception:
        pass
    gpu = _gpu_mem_process_mb()
    return {"cpu_percent": cpu, "mem_mb": mem, "gpu_mem_mb": gpu}


@contextmanager
def monitor_resources(interval: float = 0.5):
    """Context manager tracking peak CPU%, memory MB, GPU memory MB.

    Spawns a daemon thread that samples every *interval* seconds during
    the wrapped block and returns a dict with peak values on exit.

    Usage::

        with monitor_resources() as peaks:
            generate_map(...)
        print(peaks["cpu_peak"], peaks["mem_peak_mb"], peaks["gpu_peak_mb"])
    """
    import threading
    import time as _time

    peaks = {"cpu_peak": 0.0, "mem_peak_mb": 0.0, "gpu_peak_mb": 0.0}
    lock = threading.Lock()
    running = True

    def _sample():
        proc = None
        try:
            import psutil

            proc = psutil.Process()
            proc.cpu_percent(interval=0)  # warm-up
        except ImportError:
            pass

        while running:
            _time.sleep(interval)
            cpu = mem = 0.0
            if proc is not None:
                try:
                    cpu = proc.cpu_percent(interval=0)
                    mem = proc.memory_info().rss / 1024**2
                except Exception:
                    pass
            gpu = _gpu_mem_process_mb()
            with lock:
                peaks["cpu_peak"] = max(peaks["cpu_peak"], cpu)
                peaks["mem_peak_mb"] = max(peaks["mem_peak_mb"], mem)
                peaks["gpu_peak_mb"] = max(peaks["gpu_peak_mb"], gpu)

    t = threading.Thread(target=_sample, daemon=True)
    t.start()
    try:
        yield peaks
    finally:
        running = False
        t.join(timeout=3)
