#!/usr/bin/env python3
"""Generate simple plots from `runtime_artifacts/aggregated_results.json`.

Produces PNGs in `runtime_artifacts/plots`:
 - p50_p95.png  (lines: p50 and p95 by run)
 - success_error.png (bars: success/error counts by run)

Usage: python -u tools/plot_aggregated_metrics.py
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / "runtime_artifacts"
AGG = RUNTIME / "aggregated_results.json"
OUT = RUNTIME / "plots"
OUT.mkdir(parents=True, exist_ok=True)

def read_agg():
    if not AGG.exists():
        print("Missing aggregated file:", AGG)
        print("Run tools/aggregate_telemetry.py first.")
        sys.exit(2)
    with AGG.open("r", encoding="utf-8") as f:
        return json.load(f)

def try_import_matplotlib():
    try:
        import matplotlib.pyplot as plt
        return plt
    except Exception:
        return None

def plot_latency(plt, agg):
    plt.figure(figsize=(10,6))
    for scenario in sorted(agg.keys()):
        per = agg[scenario].get("per_run", [])
        x = list(range(1, len(per)+1))
        p50 = [r.get("overall_p50_ms", float('nan')) for r in per]
        p95 = [r.get("overall_p95_ms", float('nan')) for r in per]
        plt.plot(x, p50, marker='o', label=f"{scenario} p50")
        plt.plot(x, p95, marker='x', linestyle='--', label=f"{scenario} p95")
    plt.xlabel("Run #")
    plt.ylabel("Latency (ms)")
    plt.title("Aggregated p50/p95 across runs")
    plt.grid(True)
    plt.legend()
    out = OUT / "p50_p95.png"
    plt.tight_layout()
    plt.savefig(out)
    print("Wrote:", out)

def plot_counts(plt, agg):
    import numpy as np
    scenarios = sorted(agg.keys())
    max_runs = max(len(agg[s].get("per_run", [])) for s in scenarios)
    fig, ax = plt.subplots(figsize=(10,5))
    width = 0.35
    for idx, scenario in enumerate(scenarios):
        per = agg[scenario].get("per_run", [])
        runs = len(per)
        x = np.arange(1, runs+1)
        succ = [r.get("success_count", 0) for r in per]
        err = [r.get("error_count", 0) for r in per]
        offset = (idx - (len(scenarios)-1)/2) * width
        ax.bar(x + offset, succ, width=width, label=f"{scenario} success")
        ax.bar(x + offset, err, width=width, alpha=0.6, label=f"{scenario} error" if idx==0 else None)
    ax.set_xlabel("Run #")
    ax.set_ylabel("Count")
    ax.set_title("Success / Error counts per run")
    ax.legend()
    ax.grid(True)
    out = OUT / "success_error.png"
    fig.tight_layout()
    fig.savefig(out)
    print("Wrote:", out)

def main():
    data = read_agg()
    agg = data.get("aggregated") or {}
    if not agg:
        print("No aggregated data found in", AGG)
        sys.exit(2)

    plt = try_import_matplotlib()
    if plt is None:
        print("matplotlib is required to generate plots.")
        print("Install with: pip install matplotlib numpy")
        sys.exit(2)

    # create plots
    try:
        plot_latency(plt, agg)
    except Exception as e:
        print("Failed to create latency plot:", e)

    try:
        plot_counts(plt, agg)
    except Exception as e:
        print("Failed to create counts plot:", e)

if __name__ == '__main__':
    main()
