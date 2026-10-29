"""Per-device usage limits with time-of-day schedules.

A device's limits look like:

    {"cpu": 60, "schedule": [{"from": "22:00", "to": "07:00", "cpu": 100},
                             {"from": "09:00", "to": "17:00", "cpu": 0}]}

Optional power rules (phones and laptops):

    "battery":  {"cpu": 30, "cores": 2}   on battery: use at most this much
    "charging": {"cpu": 100, "cores": 6}  plugged in: use this instead of the default

"gpu_limit" is a separate usage limit for the GPU (null = the same as "cpu"); time windows and
battery rules still cap it.

"awake": true keeps the computer from sleeping while EdgeMesh runs (see awake.py).

"cores" caps how many CPU cores a browser worker uses at once (null = its default);
"gpu": false keeps a device on its CPU even when it has a usable GPU.

"always": true keeps the device in the split even when the planner thinks the mesh
would be faster without it (by default a device is used only when it helps).

"cpu" is the share of time the device may spend computing, in percent (0 = paused,
the device is left out of the layer split). The first schedule window that covers
the device's local time wins; windows may wrap past midnight. A "charging" rule replaces
the default while plugged in, and a "battery" rule caps everything (schedules too) while
on battery. Workers enforce the
limit by resting after each step in proportion to how long it took (duty cycling),
so the average load stays at the limit. The same rules live in web/worker.js.
"""

from __future__ import annotations

import glob
import json
import platform
import re
import subprocess
from datetime import datetime
from pathlib import Path

DEFAULT = {"cpu": 80, "schedule": []}  # leave headroom so devices stay usable
_TIME = re.compile(r"^([01]?\d|2[0-3]):([0-5]\d)$")


def _minutes(t: str) -> int:
    m = _TIME.match(t.strip())
    if not m:
        raise ValueError(f"bad time {t!r}, use HH:MM")
    return int(m.group(1)) * 60 + int(m.group(2))


def _pct(v) -> int:
    v = int(round(float(v)))
    if not 0 <= v <= 100:
        raise ValueError("limits are percentages from 0 to 100")
    return v


def _cores(v):
    if v in (None, "", 0):
        return None
    v = int(v)
    if not 1 <= v <= 256:
        raise ValueError("cores must be between 1 and 256")
    return v


def validate(limits: dict) -> dict:
    out = {"cpu": _pct(limits.get("cpu", DEFAULT["cpu"])), "schedule": [], "always": bool(limits.get("always", False)),
           "gpu": bool(limits.get("gpu", True)), "cores": _cores(limits.get("cores")),
           "awake": bool(limits.get("awake", False)),
           "gpu_limit": None if limits.get("gpu_limit") in (None, "") else _pct(limits["gpu_limit"])}
    for w in limits.get("schedule") or []:
        _minutes(w["from"]), _minutes(w["to"])
        out["schedule"].append({"from": w["from"].strip(), "to": w["to"].strip(), "cpu": _pct(w["cpu"])})
    for state in ("battery", "charging"):
        rule = limits.get(state)
        out[state] = {"cpu": _pct(rule.get("cpu", 100)), "cores": _cores(rule.get("cores"))} if rule else None
    return out


def parse_schedule(text: str) -> list[dict]:
    """CLI form: "22:00-07:00=100,09:00-17:00=0"."""
    windows = []
    for part in filter(None, (p.strip() for p in text.split(","))):
        span, _, pct = part.partition("=")
        start, _, end = span.partition("-")
        windows.append({"from": start, "to": end, "cpu": pct})
    return validate({"schedule": windows})["schedule"]


def effective(limits: dict, now: datetime | None = None, power: str | None = None) -> tuple[int, str]:
    """(cpu percent in force now, human-readable reason). power: "charging", "battery" or None."""
    now = now or datetime.now()
    minute = now.hour * 60 + now.minute
    cpu, why = limits.get("cpu", DEFAULT["cpu"]), "default"
    for w in limits.get("schedule", []):
        a, b = _minutes(w["from"]), _minutes(w["to"])
        inside = a <= minute < b if a < b else (minute >= a or minute < b)
        if inside:
            cpu, why = w["cpu"], f"schedule {w['from']}-{w['to']}"
            break
    else:
        if power == "charging" and limits.get("charging"):
            cpu, why = limits["charging"]["cpu"], "charging"
    if power == "battery" and limits.get("battery") and limits["battery"]["cpu"] < cpu:
        cpu, why = limits["battery"]["cpu"], "on battery"
    return cpu, why


def effective_for(limits: dict, gpu: bool, now: datetime | None = None, power: str | None = None) -> tuple[int, str]:
    """Like effective(), for the engine the device computes with now (GPU or CPU)."""
    cpu, why = effective(limits, now, power)
    g = limits.get("gpu_limit")
    if not gpu or g is None:
        return cpu, why
    return (g, "GPU limit") if why == "default" else (min(g, cpu), why)


def effective_cores(limits: dict, power: str | None = None) -> int | None:
    """Most CPU cores to use at once now (None = the device's default)."""
    cores = limits.get("cores")
    if power == "charging" and (limits.get("charging") or {}).get("cores"):
        cores = limits["charging"]["cores"]
    bat = (limits.get("battery") or {}).get("cores") if power == "battery" else None
    if bat:
        cores = min(cores or bat, bat)
    return cores


def power_state() -> str | None:
    """"charging" (plugged in), "battery", or None when the machine has no battery (desktops)."""
    try:
        system = platform.system()
        if system == "Windows":
            import ctypes

            class SPS(ctypes.Structure):
                _fields_ = [("ACLineStatus", ctypes.c_ubyte), ("BatteryFlag", ctypes.c_ubyte),
                            ("BatteryLifePercent", ctypes.c_byte), ("SystemStatusFlag", ctypes.c_byte),
                            ("BatteryLifeTime", ctypes.c_ulong), ("BatteryFullLifeTime", ctypes.c_ulong)]
            s = SPS()
            if not ctypes.windll.kernel32.GetSystemPowerStatus(ctypes.byref(s)) or s.BatteryFlag & 128:
                return None  # 128 = no system battery
            return "charging" if s.ACLineStatus == 1 else "battery"
        if system == "Darwin":
            out = subprocess.run(["pmset", "-g", "batt"], capture_output=True, text=True, timeout=5).stdout
            if "InternalBattery" not in out:
                return None
            return "charging" if "AC Power" in out else "battery"
        batteries = [b for b in glob.glob("/sys/class/power_supply/*") if Path(b, "type").read_text(encoding="utf-8").strip() == "Battery"]
        if not batteries:
            return None
        for ac in glob.glob("/sys/class/power_supply/*"):
            if Path(ac, "type").read_text(encoding="utf-8").strip() == "Mains" and Path(ac, "online").read_text(encoding="utf-8").strip() == "1":
                return "charging"
        return "battery"
    except Exception:
        return None


def rest_after(step_seconds: float, cpu: int) -> float:
    """Seconds to rest after a step so busy time stays at `cpu` percent."""
    if cpu >= 100 or cpu <= 0:
        return 0.0
    return step_seconds * (100 / cpu - 1)


class LimitStore:
    """Limits persisted on the device itself, so they survive restarts."""

    def __init__(self, path: Path, initial: dict | None = None):
        self.path = path
        self.limits = dict(DEFAULT)
        self._mtime = None
        self.reload_if_changed()
        if initial:  # command-line flags override what was saved
            self.set({**self.limits, **initial})

    def set(self, limits: dict) -> dict:
        self.limits = validate(limits)
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(self.limits))
            self._mtime = self.path.stat().st_mtime_ns
        except OSError:
            pass
        return self.limits

    def reload_if_changed(self) -> bool:
        """Pick up a change made on this device (`edgemesh limits ...`) while it runs."""
        try:
            m = self.path.stat().st_mtime_ns
            if m == self._mtime:
                return False
            self.limits, self._mtime = validate(json.loads(self.path.read_text(encoding="utf-8"))), m
            return True
        except (OSError, ValueError, KeyError):
            return False
