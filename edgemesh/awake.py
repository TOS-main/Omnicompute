"""Keep the computer from going to sleep while it contributes (the screen may still turn off
and lock). Off by default; the "awake" usage-limit switch turns it on: on this device with
`edgemesh limits --awake on`, or from the main device's Devices tab."""

from __future__ import annotations

import os
import platform
import shutil
import subprocess

_proc: subprocess.Popen | None = None
_on = False


def set_awake(on: bool) -> None:
    global _proc, _on
    if on == _on:
        return
    _on = on
    system = platform.system()
    if system == "Windows":
        import ctypes

        ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
        # applies to the calling thread; the worker's event loop thread lives as long as EdgeMesh
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | (ES_SYSTEM_REQUIRED if on else 0))
        return
    if _proc is not None:
        _proc.terminate()
        _proc = None
    if not on:
        return
    if system == "Darwin":
        cmd = ["caffeinate", "-i", "-w", str(os.getpid())]
    elif shutil.which("systemd-inhibit"):
        cmd = ["systemd-inhibit", "--what=sleep:idle", "--who=EdgeMesh", "--why=contributing to an AI mesh",
               "--mode=block", "sleep", "infinity"]
    else:
        return
    try:
        _proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except OSError:
        _proc = None


def is_on() -> bool:
    return _on
