"""Speed / memory / disk estimates, and picking which devices to use.

A device only helps if the compute time it takes off the others is larger than the
network delay it adds (each extra hop costs a round of latency per token). So the
planner tries the fastest device alone, then the fastest two, and so on, and keeps
the combination with the best estimated speed that fits in memory. Devices marked
"always use" are always included.
"""

from __future__ import annotations

from .model import plan_layers, shard_bytes

DIRECT_HOP_MS = 8.0  # typical one-way latency between two devices on a direct UDP path
DEFAULT_RTT_MS = 40.0


def usable_workers(workers: list[dict], cfg: dict | None = None) -> list[dict]:
    """Workers that may get layers, with their speed scaled by their usage limit. Models the
    browser worker can't run (cfg "native_only") use apps and computers only."""
    native_only = bool(cfg and cfg.get("native_only"))
    return [dict(w, score=w.get("score", 1.0) * w.get("limit", 100) / 100) for w in workers
            if w.get("role", "worker") == "worker" and w.get("limit", 100) > 0
            and not (native_only and w.get("kind") == "browser")]


def score_plan(cfg: dict, plan: list, by_id: dict) -> dict:
    compute_s, devices = 0.0, []
    for node, s, e in plan:
        w = by_id[node]
        mem = shard_bytes(cfg, s, e, 4)
        t = 2 * (mem / 4) / (max(w["score"], 1e-3) * 1e9)  # 2 FLOPs per weight per token
        compute_s += t
        devices.append({"node_id": node, "layers": [s, e], "memory_bytes": mem, "transfer_bytes": mem // 2,
                        "compute_ms": round(t * 1000, 1), "kind": w.get("kind")})

    def hub_leg(w):  # hub <-> device; the hub's own local worker is ~free
        return 0.5 if w.get("hub_machine") else (w.get("rtt_ms") or DEFAULT_RTT_MS) / 2

    net_ms = hub_leg(by_id[plan[0][0]]) + hub_leg(by_id[plan[-1][0]])
    for (a, _, _), (b, _, _) in zip(plan, plan[1:]):
        wa, wb = by_id[a], by_id[b]
        direct = (wa.get("kind") == "native" and wb.get("kind") == "native" and not wa.get("contributor")
                  and not wb.get("contributor") and wa.get("direct", True) and wb.get("direct", True))
        net_ms += DIRECT_HOP_MS if direct else hub_leg(wa) + hub_leg(wb)
    total = compute_s + net_ms / 1000
    return {"tok_per_s": round(1 / total, 2), "compute_ms": round(compute_s * 1000, 1),
            "network_ms": round(net_ms, 1), "devices": devices}


def best_plan(cfg: dict, workers: list[dict]) -> tuple[list, dict]:
    """(plan, estimate) using the subset of devices with the best estimated speed."""
    usable = usable_workers(workers, cfg)
    if not usable:
        if cfg.get("native_only") and usable_workers(workers):
            raise RuntimeError("this model runs on computers and the Android app only, not in browsers yet")
        raise RuntimeError("no devices are contributing (or all are paused by their usage limit)")
    by_id = {w["node_id"]: w for w in usable}
    forced = [w for w in usable if (w.get("limits") or {}).get("always")]
    rest = sorted((w for w in usable if w not in forced), key=lambda w: -w["score"])
    best, last_err = None, None
    for k in range(0 if forced else 1, len(rest) + 1):
        group = forced + rest[:k]
        try:
            plan = plan_layers(cfg["n_layer"], group, cfg, 4)
        except RuntimeError as e:
            last_err = e
            continue
        est = score_plan(cfg, plan, by_id)
        if best is None or est["tok_per_s"] > best[1]["tok_per_s"] * 1.05:  # need a real gain to add a hop
            best = (plan, est)
    if best is None:
        raise last_err or RuntimeError("no plan fits")
    return best


def estimate(cfg: dict, workers: list[dict], disk_bytes: int | None = None) -> dict:
    out = {"model": cfg.get("name"), "layers": cfg["n_layer"], "disk_bytes": disk_bytes,
           "memory_bytes_total": shard_bytes(cfg, 0, cfg["n_layer"], 4), "fits": False,
           "tok_per_s": None, "devices": [], "standby": [], "problem": None}
    try:
        plan, est = best_plan(cfg, workers)
    except RuntimeError as e:
        out["problem"] = str(e)
        return out
    used = {n for n, _, _ in plan}
    out.update(est, fits=True, standby=[w["node_id"] for w in usable_workers(workers, cfg) if w["node_id"] not in used])
    return out
