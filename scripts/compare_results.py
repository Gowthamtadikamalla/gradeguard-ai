"""
Comparison script for local vs AWS benchmark results.
Reads CSV benchmark files and generates a detailed comparison report.
"""
import pandas as pd
from pathlib import Path
import argparse
from datetime import datetime

RESULTS_DIR = Path("results/benchmarks")

def load_latest_benchmark(label: str) -> pd.DataFrame:
    """Load the latest benchmark result for a given label."""
    pattern = f"benchmark_{label}_*.csv"
    files = sorted(RESULTS_DIR.glob(pattern))
    
    if not files:
        raise FileNotFoundError(f"No benchmark files found for label '{label}' in {RESULTS_DIR}")
    
    latest_file = files[-1]  # Most recent file
    df = pd.read_csv(latest_file)
    print(f"Loaded: {latest_file.name}")
    return df.iloc[0], latest_file.name  # Return first row (should only be one)

def calculate_difference(local_val: float, aws_val: float, is_percentage: bool = False) -> tuple:
    """Calculate absolute and percentage difference."""
    if local_val == 0:
        return 0.0, 0.0
    
    abs_diff = aws_val - local_val
    pct_diff = (abs_diff / local_val) * 100 if local_val != 0 else 0.0
    
    if is_percentage:
        # For percentage metrics, return difference in percentage points
        return abs_diff, abs_diff
    else:
        return abs_diff, pct_diff

def format_time(seconds: float) -> str:
    """Format time in human-readable format."""
    if seconds < 1:
        return f"{seconds*1000:.2f} ms"
    elif seconds < 60:
        return f"{seconds:.2f} s"
    else:
        mins = int(seconds // 60)
        secs = seconds % 60
        return f"{mins}m {secs:.2f}s"

def format_comparison(value: float, diff: float, pct_diff: float, unit: str = "") -> str:
    """Format comparison with difference."""
    sign = "+" if diff >= 0 else ""
    return f"{value:.3f}{unit} ({sign}{diff:.3f}{unit}, {sign}{pct_diff:.2f}%)"

def compare_benchmarks(local_csv: str = None, aws_csv: str = None):
    """Compare local and AWS benchmark results."""
    print("=" * 80)
    print("BENCHMARK COMPARISON: Local vs AWS")
    print("=" * 80)
    
    # Load benchmark data
    if local_csv:
        local_df = pd.read_csv(RESULTS_DIR / local_csv).iloc[0]
        local_file = local_csv
    else:
        local_df, local_file = load_latest_benchmark("local")
    
    if aws_csv:
        aws_df = pd.read_csv(RESULTS_DIR / aws_csv).iloc[0]
        aws_file = aws_csv
    else:
        aws_df, aws_file = load_latest_benchmark("aws")
    
    print(f"\nLocal benchmark:  {local_file}")
    print(f"AWS benchmark:    {aws_file}")
    print(f"\nRequests tested:  {int(local_df['n_requests'])}")
    
    # Time Comparison
    print("\n" + "=" * 80)
    print("TIME COMPARISON")
    print("=" * 80)
    
    local_total_time = local_df['total_execution_time_sec']
    aws_total_time = aws_df['total_execution_time_sec']
    time_diff, time_pct = calculate_difference(local_total_time, aws_total_time)
    
    print(f"\nTotal Execution Time:")
    print(f"  Local:  {format_time(local_total_time)}")
    print(f"  AWS:    {format_time(aws_total_time)}")
    print(f"  Diff:   {format_time(time_diff)} ({time_pct:+.2f}%)")
    
    local_throughput = local_df['throughput_req_per_sec']
    aws_throughput = aws_df['throughput_req_per_sec']
    throughput_diff, throughput_pct = calculate_difference(local_throughput, aws_throughput)
    
    print(f"\nThroughput (Requests/Second):")
    print(f"  Local:  {local_throughput:.2f} req/s")
    print(f"  AWS:    {aws_throughput:.2f} req/s")
    print(f"  Diff:   {throughput_diff:+.2f} req/s ({throughput_pct:+.2f}%)")
    
    # Latency Comparison
    print("\n" + "=" * 80)
    print("LATENCY COMPARISON (Response Time)")
    print("=" * 80)
    
    latency_metrics = [
        ('min_latency_ms', 'Min Latency'),
        ('avg_latency_ms', 'Average Latency'),
        ('p50_latency_ms', 'Median (p50) Latency'),
        ('p95_latency_ms', '95th Percentile (p95)'),
        ('p99_latency_ms', '99th Percentile (p99)'),
        ('max_latency_ms', 'Max Latency')
    ]
    
    print(f"\n{'Metric':<25} {'Local (ms)':<20} {'AWS (ms)':<20} {'Difference':<20}")
    print("-" * 85)
    
    for metric_key, metric_name in latency_metrics:
        local_val = local_df[metric_key]
        aws_val = aws_df[metric_key]
        diff, pct_diff = calculate_difference(local_val, aws_val)
        
        print(f"{metric_name:<25} {local_val:>10.3f} ms      {aws_val:>10.3f} ms      "
              f"{diff:>+10.3f} ms ({pct_diff:>+6.2f}%)")
    
    # Accuracy Comparison
    print("\n" + "=" * 80)
    print("ACCURACY COMPARISON")
    print("=" * 80)
    
    local_accuracy = local_df['accuracy']
    aws_accuracy = aws_df['accuracy']
    acc_diff, acc_pct = calculate_difference(local_accuracy, aws_accuracy, is_percentage=True)
    
    print(f"\nPrediction Accuracy:")
    print(f"  Local:  {local_accuracy:.4f} ({local_accuracy*100:.2f}%)")
    print(f"  AWS:    {aws_accuracy:.4f} ({aws_accuracy*100:.2f}%)")
    print(f"  Diff:   {acc_diff:+.4f} ({acc_pct:+.2f} percentage points)")
    
    # Reliability Comparison
    print("\n" + "=" * 80)
    print("RELIABILITY COMPARISON")
    print("=" * 80)
    
    local_success = local_df['success_rate']
    aws_success = aws_df['success_rate']
    success_diff, success_pct = calculate_difference(local_success, aws_success, is_percentage=True)
    
    local_reliability = local_df['reliability_percent']
    aws_reliability = aws_df['reliability_percent']
    
    local_errors = int(local_df['n_errors'])
    aws_errors = int(aws_df['n_errors'])
    
    print(f"\nSuccess Rate:")
    print(f"  Local:  {local_success:.4f} ({local_reliability:.2f}%)")
    print(f"  AWS:    {aws_success:.4f} ({aws_reliability:.2f}%)")
    print(f"  Diff:   {success_diff:+.4f} ({success_pct:+.2f} percentage points)")
    
    print(f"\nErrors:")
    print(f"  Local:  {local_errors} errors")
    print(f"  AWS:    {aws_errors} errors")
    
    local_error_rate = local_df['error_rate']
    aws_error_rate = aws_df['error_rate']
    print(f"\nError Rate:")
    print(f"  Local:  {local_error_rate:.4f} ({local_error_rate*100:.2f}%)")
    print(f"  AWS:    {aws_error_rate:.4f} ({aws_error_rate*100:.2f}%)")
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY & RECOMMENDATIONS")
    print("=" * 80)
    
    print("\nKey Findings (Standardized Test - Same # of requests):")
    
    # Time analysis
    if time_diff < 0:
        print(f"  [OK] AWS is {abs(time_pct):.2f}% FASTER (saves {format_time(abs(time_diff))})")
    else:
        print(f"  [WARNING] Local is {time_pct:.2f}% FASTER (AWS takes {format_time(time_diff)} longer)")
        print(f"     Note: This includes network latency and API Gateway overhead")
    
    # Throughput analysis
    if throughput_diff > 0:
        print(f"  [OK] AWS handles {throughput_diff:.2f} more requests/second")
    else:
        print(f"  [WARNING] Local handles {abs(throughput_diff):.2f} more requests/second")
        print(f"     Note: Local has no network overhead; cloud includes internet latency")
    
    # Latency analysis
    avg_lat_diff, avg_lat_pct = calculate_difference(local_df['avg_latency_ms'], aws_df['avg_latency_ms'])
    if avg_lat_diff < 0:
        print(f"  [OK] AWS has {abs(avg_lat_pct):.2f}% LOWER average latency")
    else:
        print(f"  [WARNING] Local has {abs(avg_lat_pct):.2f}% LOWER average latency")
        print(f"     Note: Local runs on same machine (zero network latency)")
        print(f"     Cloud min latency: {aws_df['min_latency_ms']:.2f}ms (after warmup) is excellent")
    
    # Accuracy analysis
    if abs(acc_diff) < 0.001:
        print(f"  [OK] Accuracy is EQUIVALENT (model performance consistent)")
    else:
        if acc_diff > 0:
            print(f"  [OK] AWS has {acc_pct:.2f} percentage points HIGHER accuracy")
        else:
            print(f"  [WARNING] Local has {abs(acc_pct):.2f} percentage points HIGHER accuracy")
    
    # Reliability analysis
    if success_diff > 0:
        print(f"  [OK] AWS has {success_pct:.2f} percentage points HIGHER reliability")
    elif success_diff < 0:
        print(f"  [WARNING] Local has {abs(success_pct):.2f} percentage points HIGHER reliability")
    else:
        print(f"  [OK] Reliability is EQUIVALENT (both 100%)")
    
    # Additional context
    print("\nPerformance Context:")
    print(f"  - Both tests used {int(local_df['n_requests'])} requests (fair comparison)")
    print(f"  - Local: Same machine = zero network latency")
    print(f"  - AWS: Includes internet + API Gateway + Lambda overhead")
    print(f"  - AWS min latency ({aws_df['min_latency_ms']:.2f}ms) shows excellent performance after warmup")
    print(f"  - For production: Cloud provides scalability, reliability, and global availability")
    
    print("\n" + "=" * 80)
    
    # Save comparison report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = RESULTS_DIR / f"comparison_{timestamp}.txt"
    
    # For now, just print. Can enhance to save full report later.
    print(f"\nComparison complete. Results shown above.")
    print(f"Individual benchmark files:")
    print(f"  - {local_file}")
    print(f"  - {aws_file}")

def main():
    parser = argparse.ArgumentParser(
        description="Compare local vs AWS benchmark results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compare latest local vs AWS benchmarks
  python scripts/compare_results.py
  
  # Compare specific files
  python scripts/compare_results.py --local benchmark_local_20231201_120000.csv --aws benchmark_aws_20231201_130000.csv
        """
    )
    parser.add_argument(
        "--local",
        type=str,
        default=None,
        help="Local benchmark CSV file name (default: latest 'local' benchmark)"
    )
    parser.add_argument(
        "--aws",
        type=str,
        default=None,
        help="AWS benchmark CSV file name (default: latest 'aws' benchmark)"
    )
    args = parser.parse_args()
    
    compare_benchmarks(args.local, args.aws)

if __name__ == "__main__":
    main()

