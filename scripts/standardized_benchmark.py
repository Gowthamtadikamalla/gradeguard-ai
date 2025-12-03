"""
Standardized benchmarking script that ensures fair comparison
between local and AWS deployments using identical test conditions.
"""
import subprocess
import sys
import time
from pathlib import Path
import argparse
from datetime import datetime

def run_standardized_benchmark(endpoint: str, label: str, n: int, warmup: int, api_key: str = None):
    """Run a standardized benchmark with consistent parameters."""
    print("\n" + "=" * 80)
    print(f"STANDARDIZED BENCHMARK: {label.upper()}")
    print("=" * 80)
    print(f"Endpoint: {endpoint}")
    print(f"Requests: {n}")
    print(f"Warmup: {warmup}")
    print("=" * 80)
    
    cmd = [
        sys.executable,
        "scripts/benchmark.py",
        "--endpoint", endpoint,
        "--label", label,
        "--n", str(n),
        "--warmup", str(warmup)
    ]
    
    if api_key:
        cmd.extend(["--api-key", api_key])
    
    result = subprocess.run(cmd, capture_output=False)
    return result.returncode == 0

def main():
    parser = argparse.ArgumentParser(
        description="Run standardized benchmarks for fair local vs AWS comparison",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run both local and AWS with 200 requests
  python scripts/standardized_benchmark.py --n 200 --aws-endpoint https://... --api-key KEY
  
  # Run only AWS (if local server not available)
  python scripts/standardized_benchmark.py --n 200 --aws-endpoint https://... --api-key KEY --skip-local
        """
    )
    parser.add_argument(
        "--local-endpoint",
        type=str,
        default="http://127.0.0.1:8000",
        help="Local API endpoint URL"
    )
    parser.add_argument(
        "--aws-endpoint",
        type=str,
        required=True,
        help="AWS API Gateway endpoint URL (required)"
    )
    parser.add_argument(
        "--api-key",
        type=str,
        required=True,
        help="API Gateway API key (required)"
    )
    parser.add_argument(
        "--n",
        type=int,
        default=200,
        help="Number of requests to benchmark (default: 200)"
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=10,
        help="Number of warmup requests (default: 10)"
    )
    parser.add_argument(
        "--skip-local",
        action="store_true",
        help="Skip local benchmark"
    )
    parser.add_argument(
        "--skip-aws",
        action="store_true",
        help="Skip AWS benchmark"
    )
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("STANDARDIZED BENCHMARKING: Local vs AWS")
    print("=" * 80)
    print(f"\nTest Configuration (IDENTICAL for both):")
    print(f"  Requests per test:  {args.n}")
    print(f"  Warmup requests:    {args.warmup}")
    print(f"  Local endpoint:     {args.local_endpoint}")
    print(f"  AWS endpoint:       {args.aws_endpoint}")
    print("=" * 80)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Run local benchmark
    local_success = False
    if not args.skip_local:
        print("\n[STEP 1/3] Running LOCAL benchmark...")
        local_label = f"local-std-{timestamp}"
        local_success = run_standardized_benchmark(
            args.local_endpoint,
            local_label,
            args.n,
            args.warmup
        )
        if local_success:
            print(f"[OK] Local benchmark completed: {local_label}")
        else:
            print(f"[ERROR] Local benchmark failed")
        time.sleep(2)
    else:
        print("\n[SKIPPED] Local benchmark (--skip-local)")
    
    # Run AWS benchmark
    aws_success = False
    if not args.skip_aws:
        print("\n[STEP 2/3] Running AWS benchmark...")
        aws_label = f"aws-std-{timestamp}"
        aws_success = run_standardized_benchmark(
            args.aws_endpoint,
            aws_label,
            args.n,
            args.warmup,
            args.api_key
        )
        if aws_success:
            print(f"[OK] AWS benchmark completed: {aws_label}")
        else:
            print(f"[ERROR] AWS benchmark failed")
    else:
        print("\n[SKIPPED] AWS benchmark (--skip-aws)")
    
    # Generate comparison
    if local_success and aws_success:
        print("\n[STEP 3/3] Generating standardized comparison...")
        
        # Find the benchmark files we just created
        results_dir = Path("results/benchmarks")
        local_files = sorted(results_dir.glob(f"benchmark_{local_label}_*.csv"))
        aws_files = sorted(results_dir.glob(f"benchmark_{aws_label}_*.csv"))
        
        if local_files and aws_files:
            local_file = local_files[-1].name
            aws_file = aws_files[-1].name
            
            result = subprocess.run(
                [sys.executable, "scripts/compare_results.py", "--local", local_file, "--aws", aws_file],
                capture_output=False
            )
            
            if result.returncode == 0:
                print("\n" + "=" * 80)
                print("STANDARDIZED COMPARISON COMPLETE")
                print("=" * 80)
                print(f"\nBenchmark files:")
                print(f"  Local: {local_file}")
                print(f"  AWS:   {aws_file}")
                print(f"\nBoth tests used IDENTICAL parameters:")
                print(f"  - Requests: {args.n}")
                print(f"  - Warmup: {args.warmup}")
                print(f"  - Same dataset samples")
            else:
                print("\n[WARNING] Comparison script had errors")
        else:
            print("\n[WARNING] Could not find benchmark files for comparison")
    elif args.skip_local or args.skip_aws:
        print("\n[INFO] Skipped comparison (one benchmark was skipped)")
    else:
        print("\n[WARNING] Could not generate comparison (one or both benchmarks failed)")

if __name__ == "__main__":
    main()

