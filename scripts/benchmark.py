"""
Benchmarking script for student academic standing prediction API.
Measures latency, throughput, and accuracy.

The model predicts at-risk vs good academic standing using two-feature labeling:
Pass requires GPA >= 2.0 AND AvgTestScore >= 73.
All other students are labeled as Fail (at-risk).
AttendanceRate is used as a predictive feature but not in target labeling.
"""
import time
import statistics
import csv
from pathlib import Path
from datetime import datetime
import pandas as pd
import requests
import argparse

PROC = Path("data/processed")
OUT = Path("results/benchmarks")
OUT.mkdir(parents=True, exist_ok=True)

def percentile(data: list, p: float) -> float:
    """Calculate percentile of a list."""
    if not data:
        return 0.0
    data_sorted = sorted(data)
    k = (len(data_sorted) - 1) * p
    f = int(k)
    c = min(f + 1, len(data_sorted) - 1)
    if f == c:
        return data_sorted[f]
    return data_sorted[f] + (data_sorted[c] - data_sorted[f]) * (k - f)

def prepare_payload(row: pd.Series) -> dict:
    """Prepare API payload from DataFrame row, excluding target and leakage features.
    
    Excludes:
    - pass_fail: Target variable
    - GPA: Used to create target (leakage)
    - Test scores: Directly tied to GPA (leakage)
    - Gender, Race: Protected attributes (excluded from model training)
    """
    # Drop target, leakage features, and protected attributes
    exclude_cols = [
        "pass_fail", 
        "GPA",
        "TestScore_Math",
        "TestScore_Reading", 
        "TestScore_Science",
        "Gender",  # Protected attribute - excluded from model
        "Race"     # Protected attribute - excluded from model
    ]
    payload = row.drop(labels=exclude_cols, errors="ignore").to_dict()
    
    # Convert NaN to None for JSON serialization
    payload = {k: (None if pd.isna(v) else v) for k, v in payload.items()}
    
    return payload

def main(endpoint: str, label: str, warmup: int, n: int, api_key: str = None):
    """Run benchmarking tests.
    
    Args:
        endpoint: API endpoint URL
        label: Label for this benchmark run
        warmup: Number of warmup requests
        n: Number of requests to benchmark
        api_key: Optional API key for API Gateway authentication
    """
    print("=" * 60)
    print("Benchmarking Student Performance Prediction API")
    print("=" * 60)
    print(f"Endpoint: {endpoint}")
    print(f"Label: {label}")
    print(f"Warmup requests: {warmup}")
    print(f"Benchmark requests: {n}")
    
    # Load probe dataset
    probe_path = PROC / "probe.csv"
    if not probe_path.exists():
        raise FileNotFoundError(f"Probe dataset not found: {probe_path}. Run prepare_data.py first.")
    
    probe = pd.read_csv(probe_path)
    rows = probe.head(n)
    
    print(f"\nLoaded {len(rows)} samples from probe dataset")
    print(f"True label distribution: Pass={rows['pass_fail'].sum()}, Fail={(~rows['pass_fail'].astype(bool)).sum()}")
    
    # Prepare headers
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["x-api-key"] = api_key
        print(f"\n[INFO] Using API key authentication for API Gateway")
    
    # Warmup phase
    print(f"\n[1/3] Warmup phase ({warmup} requests)...")
    for i, (_, r) in enumerate(rows.head(warmup).iterrows(), 1):
        payload = prepare_payload(r)
        try:
            requests.post(f"{endpoint}/predict", json=payload, headers=headers, timeout=30)
            if i % 5 == 0:
                print(f"  Warmup {i}/{warmup}...")
        except Exception as e:
            print(f"  Warning: Warmup request {i} failed: {e}")
    
    print("  Warmup complete")
    
    # Benchmark phase
    print(f"\n[2/3] Benchmarking ({n} requests)...")
    latencies = []
    y_true = rows["pass_fail"].tolist()
    y_pred = []
    errors = 0
    start_time_total = time.perf_counter()  # Track total execution time
    
    # Prepare headers for requests
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["x-api-key"] = api_key
    
    for i, (_, r) in enumerate(rows.iterrows(), 1):
        payload = prepare_payload(r)
        
        try:
            t0 = time.perf_counter_ns()
            resp = requests.post(f"{endpoint}/predict", json=payload, headers=headers, timeout=30)
            t1 = time.perf_counter_ns()
            
            latency_ms = (t1 - t0) / 1e6
            latencies.append(latency_ms)
            
            if resp.ok:
                result = resp.json()
                y_pred.append(int(result.get("pass_fail", 0)))
            else:
                errors += 1
                y_pred.append(0)  # Default prediction on error
                # Print detailed error for first few failures
                if i <= 3:
                    try:
                        error_detail = resp.json()
                        print(f"  Warning: Request {i} returned status {resp.status_code}: {error_detail.get('detail', resp.text[:200])}")
                    except:
                        print(f"  Warning: Request {i} returned status {resp.status_code}: {resp.text[:200]}")
                else:
                    print(f"  Warning: Request {i} returned status {resp.status_code}")
        
        except Exception as e:
            errors += 1
            latencies.append(0.0)  # Record error latency as 0
            y_pred.append(0)
            print(f"  Error on request {i}: {e}")
        
        # Progress indicator
        if i % 100 == 0:
            print(f"  Completed {i}/{n} requests...")
    
    end_time_total = time.perf_counter()
    total_execution_time = end_time_total - start_time_total  # Total wall clock time
    
    print("  Benchmarking complete")
    
    # Calculate metrics
    print(f"\n[3/3] Calculating metrics...")
    
    # Latency metrics
    valid_latencies = [l for l in latencies if l > 0]
    avg_latency = statistics.mean(valid_latencies) if valid_latencies else 0.0
    median_latency = statistics.median(valid_latencies) if valid_latencies else 0.0
    min_latency = min(valid_latencies) if valid_latencies else 0.0
    max_latency = max(valid_latencies) if valid_latencies else 0.0
    p50_latency = percentile(valid_latencies, 0.50)
    p95_latency = percentile(valid_latencies, 0.95)
    p99_latency = percentile(valid_latencies, 0.99)
    
    # Accuracy
    correct = sum(int(a == b) for a, b in zip(y_true, y_pred))
    accuracy = correct / max(1, len(y_true))
    
    # Time metrics
    total_execution_time_ms = total_execution_time * 1000  # Convert to milliseconds
    total_execution_time_sec = total_execution_time  # Keep in seconds
    
    # Throughput (requests per second)
    throughput = len(valid_latencies) / total_execution_time_sec if total_execution_time_sec > 0 else 0.0
    
    # Reliability metrics
    error_rate = errors / n if n > 0 else 0.0
    success_rate = (n - errors) / n if n > 0 else 0.0
    reliability = success_rate * 100  # Percentage
    
    # Print results
    print("\n" + "=" * 60)
    print("Benchmark Results")
    print("=" * 60)
    print(f"Total requests:      {n}")
    print(f"Successful:          {n - errors}")
    print(f"Errors:              {errors} ({error_rate*100:.2f}%)")
    print(f"\nTime Metrics:")
    print(f"  Total execution:   {total_execution_time_sec:.2f} seconds ({total_execution_time_ms:.2f} ms)")
    print(f"  Throughput:        {throughput:.2f} req/s")
    print(f"\nLatency (ms):")
    print(f"  Min:               {min_latency:.3f}")
    print(f"  Average:           {avg_latency:.3f}")
    print(f"  Median (p50):      {p50_latency:.3f}")
    print(f"  p95:               {p95_latency:.3f}")
    print(f"  p99:               {p99_latency:.3f}")
    print(f"  Max:               {max_latency:.3f}")
    print(f"\nAccuracy:            {accuracy:.4f} ({correct}/{len(y_true)} correct)")
    print(f"\nReliability:")
    print(f"  Success rate:      {success_rate:.4f} ({reliability:.2f}%)")
    print(f"  Error rate:        {error_rate:.4f} ({error_rate*100:.2f}%)")
    print("=" * 60)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_csv = OUT / f"benchmark_{label}_{timestamp}.csv"
    
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "label", "timestamp", "n_requests", "n_errors", "error_rate",
            "total_execution_time_sec", "total_execution_time_ms",
            "min_latency_ms", "avg_latency_ms", "median_latency_ms", "p50_latency_ms",
            "p95_latency_ms", "p99_latency_ms", "max_latency_ms",
            "throughput_req_per_sec", "accuracy", "success_rate", "reliability_percent"
        ])
        w.writerow([
            label, timestamp, n, errors, round(error_rate, 4),
            round(total_execution_time_sec, 3), round(total_execution_time_ms, 2),
            round(min_latency, 3), round(avg_latency, 3), round(median_latency, 3),
            round(p50_latency, 3), round(p95_latency, 3), round(p99_latency, 3),
            round(max_latency, 3), round(throughput, 2), round(accuracy, 4),
            round(success_rate, 4), round(reliability, 2)
        ])
    
    print(f"\n[OK] Results saved to {out_csv}")
    
    return {
        "label": label,
        "n_requests": n,
        "errors": errors,
        "total_execution_time_sec": total_execution_time_sec,
        "total_execution_time_ms": total_execution_time_ms,
        "min_latency_ms": min_latency,
        "avg_latency_ms": avg_latency,
        "p50_latency_ms": p50_latency,
        "p95_latency_ms": p95_latency,
        "p99_latency_ms": p99_latency,
        "max_latency_ms": max_latency,
        "throughput_req_per_sec": throughput,
        "accuracy": accuracy,
        "success_rate": success_rate,
        "reliability_percent": reliability
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark student performance prediction API")
    parser.add_argument("--endpoint", default="http://127.0.0.1:8000", help="API endpoint URL")
    parser.add_argument("--label", default="local", help="Label for this benchmark run")
    parser.add_argument("--warmup", type=int, default=10, help="Number of warmup requests")
    parser.add_argument("--n", type=int, default=1000, help="Number of requests to benchmark")
    parser.add_argument("--api-key", type=str, default=None, help="API key for API Gateway (required for AWS endpoint)")
    args = parser.parse_args()
    
    main(args.endpoint, args.label, args.warmup, args.n, args.api_key)

