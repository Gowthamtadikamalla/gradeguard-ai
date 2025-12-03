"""
Comprehensive benchmarking script that runs both local and AWS benchmarks
and generates a detailed comparison report.

This script:
1. Runs benchmark against local FastAPI server
2. Runs benchmark against AWS API Gateway endpoint
3. Generates a comprehensive comparison report
4. Creates visual comparison showing cloud advantages
"""
import subprocess
import sys
import time
from pathlib import Path
import argparse

def check_local_server(endpoint: str) -> bool:
    """Check if local server is running."""
    import requests
    try:
        resp = requests.get(f"{endpoint}/health", timeout=2)
        return resp.status_code == 200
    except:
        return False

def run_benchmark(endpoint: str, label: str, n: int, api_key: str = None, warmup: int = 10):
    """Run a single benchmark."""
    print("\n" + "=" * 80)
    print(f"Running {label.upper()} benchmark")
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
        description="Run comprehensive local vs AWS benchmarks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run benchmarks with default settings
  python scripts/run_benchmarks.py --aws-endpoint https://lde2uzyhu8.execute-api.us-east-2.amazonaws.com/prod --api-key YOUR_KEY
  
  # Run with custom request count
  python scripts/run_benchmarks.py --aws-endpoint https://... --api-key YOUR_KEY --n 2000
        """
    )
    parser.add_argument(
        "--local-endpoint",
        type=str,
        default="http://127.0.0.1:8000",
        help="Local API endpoint URL (default: http://127.0.0.1:8000)"
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
        default=1000,
        help="Number of requests to benchmark (default: 1000)"
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
        help="Skip local benchmark (only run AWS)"
    )
    parser.add_argument(
        "--skip-aws",
        action="store_true",
        help="Skip AWS benchmark (only run local)"
    )
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("COMPREHENSIVE BENCHMARKING: Local vs AWS")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Local endpoint:  {args.local_endpoint}")
    print(f"  AWS endpoint:    {args.aws_endpoint}")
    print(f"  Requests:        {args.n}")
    print(f"  Warmup:          {args.warmup}")
    print("=" * 80)
    
    # Check local server if not skipping
    if not args.skip_local:
        print("\n[CHECK] Verifying local server is running...")
        if not check_local_server(args.local_endpoint):
            print(f"\n[ERROR] Local server is not running at {args.local_endpoint}")
            print("Please start the local server first:")
            print("  python serve.py")
            print("\nOr use --skip-local to only run AWS benchmark")
            sys.exit(1)
        print("[OK] Local server is running")
    
    # Run local benchmark
    local_success = True
    if not args.skip_local:
        print("\n" + "=" * 80)
        print("STEP 1: Running LOCAL benchmark")
        print("=" * 80)
        local_success = run_benchmark(
            args.local_endpoint,
            "local",
            args.n,
            warmup=args.warmup
        )
        if not local_success:
            print("\n[WARNING] Local benchmark had errors, but continuing...")
        time.sleep(2)  # Brief pause between benchmarks
    else:
        print("\n[Skipping] Local benchmark (--skip-local)")
    
    # Run AWS benchmark
    aws_success = True
    if not args.skip_aws:
        print("\n" + "=" * 80)
        print("STEP 2: Running AWS benchmark")
        print("=" * 80)
        aws_success = run_benchmark(
            args.aws_endpoint,
            "aws",
            args.n,
            api_key=args.api_key,
            warmup=args.warmup
        )
        if not aws_success:
            print("\n[WARNING] AWS benchmark had errors, but continuing...")
    else:
        print("\n[Skipping] AWS benchmark (--skip-aws)")
    
    # Generate comparison
    if local_success and aws_success and not args.skip_local and not args.skip_aws:
        print("\n" + "=" * 80)
        print("STEP 3: Generating comparison report")
        print("=" * 80)
        
        result = subprocess.run(
            [sys.executable, "scripts/compare_results.py"],
            capture_output=False
        )
        
        if result.returncode == 0:
            print("\n" + "=" * 80)
            print("BENCHMARKING COMPLETE")
            print("=" * 80)
            print("\nNext steps:")
            print("1. Review the comparison report above")
            print("2. Check individual benchmark files in results/benchmarks/")
            print("3. Use the comparison to demonstrate cloud advantages")
        else:
            print("\n[WARNING] Comparison script had errors")
    elif args.skip_local or args.skip_aws:
        print("\n[INFO] Skipped comparison (one benchmark was skipped)")
    else:
        print("\n[WARNING] Could not generate comparison (one or both benchmarks failed)")

if __name__ == "__main__":
    main()

