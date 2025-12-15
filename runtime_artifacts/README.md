**Automated Benchmark Comparison**

- Baseline: load_telemetry_baseline.json
  - requests: 200 succeeded=1 failed=199
  - overall p50=12.390449999656994 ms, p95=32.95754499686153 ms

- Increased limits: load_telemetry_increased.json
  - requests: 200 succeeded=200 failed=0
  - overall p50=139.847850001388 ms, p95=168.4641300020303 ms

- Interpretation:
  - Increasing the limits removed 429 failures and allowed full processing; provider latencies remained small while overall p50/p95 rose due to real work being executed.

- Next steps:
  - Use the increased limits for controlled benchmarks only; keep defaults in production.