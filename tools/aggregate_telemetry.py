#!/usr/bin/env python3
"""Aggregate multiple runbook telemetry outputs.

This script looks for files named:
  runtime_artifacts/load_telemetry_{scenario}_run{n}.json

It computes mean/median/stdev for common metrics found under the
`load_summary` key and writes `runtime_artifacts/aggregated_results.json`.
"""
from __future__ import annotations
import json
import glob
import statistics
from pathlib import Path
from datetime import datetime
import sys

RUNTIME = Path("runtime_artifacts")

METRICS = [
    "overall_p50_ms",
    "overall_p95_ms",
    "provider_p50_ms",
    "provider_p95_ms",
    "throughput",
    "total_requests",
    "success_count",
    "error_count",
]


def load_json(p: Path):
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def find_run_files():
    return sorted(RUNTIME.glob("load_telemetry_*_run*.json"))


def scenario_from_name(name: str) -> str:
    # name like load_telemetry_increased_run1.json -> increased
    if not name.startswith("load_telemetry_"):
        return name
    tail = name[len("load_telemetry_"):]
    parts = tail.split("_run", 1)
    return parts[0]


def collect():
    files = find_run_files()
    if not files:
        print("No run files found: runtime_artifacts/load_telemetry_*_run*.json")
        sys.exit(2)

    grouped: dict[str, list[dict]] = {}
    for p in files:
        try:
            j = load_json(p)
            summary = j.get("load_summary") or j.get("summary") or j
            scenario = scenario_from_name(p.name)
            grouped.setdefault(scenario, []).append(summary)
        except Exception as e:
            print("failed to read", p, e)

    aggregated: dict[str, dict] = {}
    for scenario, runs in grouped.items():
        agg: dict = {"runs": len(runs), "per_run": []}
        # gather per-run metric lists
        per_metric: dict[str, list[float]] = {m: [] for m in METRICS}
        for r in runs:
            per_run = {}
            for m in METRICS:
                v = r.get(m)
                if v is None:
                    # skip missing
                    continue
                try:
                    fv = float(v)
                except Exception:
                    continue
                per_metric[m].append(fv)
                per_run[m] = fv
            agg["per_run"].append(per_run)

        stats = {}
        for m, vals in per_metric.items():
            if not vals:
                continue
            stats[m] = {
                "count": len(vals),
                "mean": statistics.mean(vals),
                "median": statistics.median(vals),
                "stdev": statistics.pstdev(vals) if len(vals) > 1 else 0.0,
                "values": vals,
            }
        aggregated[scenario] = {"summary": stats, **agg}

    out = {"generated_at": datetime.utcnow().isoformat() + "Z", "aggregated": aggregated}
    out_path = RUNTIME / "aggregated_results.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print("Wrote:", out_path)
    return out_path


if __name__ == '__main__':
    collect()
