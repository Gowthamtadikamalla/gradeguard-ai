"""
Real-time monitoring script for API performance.
Monitors API endpoint and displays metrics in real-time.
"""
import time
import requests
import argparse
from collections import deque
from datetime import datetime
import statistics

def percentile(data: list, p: float) -> float:
    """Calculate percentile."""
    if not data:
        return 0.0
    data_sorted = sorted(data)
    k = (len(data_sorted) - 1) * p
    f = int(k)
    c = min(f + 1, len(data_sorted) - 1)
    if f == c:
        return data_sorted[f]
    return data_sorted[f] + (data_sorted[c] - data_sorted[f]) * (k - f)

def monitor_api(endpoint: str, interval: float = 1.0, window: int = 100):
    """Monitor API endpoint with sliding window."""
    print("=" * 80)
    print(f"Real-time API Monitoring: {endpoint}")
    print("=" * 80)
    print(f"Monitoring interval: {interval}s | Window size: {window} requests")
    print(f"Press Ctrl+C to stop\n")
    
    latencies = deque(maxlen=window)
    successes = deque(maxlen=window)
    errors = deque(maxlen=window)
    start_time = time.time()
    
    request_count = 0
    
    try:
        while True:
            # Check API health first
            try:
                health_start = time.perf_counter()
                health_resp = requests.get(f"{endpoint}/health", timeout=5)
                health_latency = (time.perf_counter() - health_start) * 1000
                
                if health_resp.ok:
                    health_data = health_resp.json()
                    status = health_data.get("status", "unknown")
                    model = health_data.get("model", "unknown")
                    
                    # Try a prediction request
                    request_count += 1
                    pred_start = time.perf_counter()
                    
                    # Sample prediction payload (excludes Gender, Race, and test scores as they're not used by the model)
                    sample_payload = {
                        "Age": 17,
                        "Grade": 12,
                        "SES_Quartile": 3,
                        "ParentalEducation": "HS",
                        "SchoolType": "Public",
                        "Locale": "Suburban",
                        "AttendanceRate": 0.9,
                        "StudyHours": 2.0,
                        "InternetAccess": 1,
                        "Extracurricular": 1,
                        "PartTimeJob": 0,
                        "ParentSupport": 1,
                        "Romantic": 0,
                        "FreeTime": 3,
                        "GoOut": 2
                    }
                    
                    pred_resp = requests.post(
                        f"{endpoint}/predict",
                        json=sample_payload,
                        timeout=10
                    )
                    pred_latency = (time.perf_counter() - pred_start) * 1000
                    
                    if pred_resp.ok:
                        latencies.append(pred_latency)
                        successes.append(1)
                        errors.append(0)
                    else:
                        successes.append(0)
                        errors.append(1)
                        latencies.append(0)
                    
                else:
                    successes.append(0)
                    errors.append(1)
                    
            except requests.exceptions.RequestException as e:
                successes.append(0)
                errors.append(1)
                latencies.append(0)
            
            # Calculate and display metrics
            if len(latencies) > 0:
                valid_lats = [l for l in latencies if l > 0]
                if valid_lats:
                    avg_lat = statistics.mean(valid_lats)
                    p95_lat = percentile(valid_lats, 0.95)
                    p99_lat = percentile(valid_lats, 0.99)
                else:
                    avg_lat = p95_lat = p99_lat = 0.0
                
                success_count = sum(successes)
                error_count = sum(errors)
                total_in_window = len(successes)
                success_rate = success_count / total_in_window if total_in_window > 0 else 0.0
                
                uptime = time.time() - start_time
                
                # Clear line and print metrics
                print("\r" + " " * 80, end="")  # Clear line
                print(f"\r[{datetime.now().strftime('%H:%M:%S')}] "
                      f"Requests: {request_count} | "
                      f"Uptime: {uptime:.1f}s | "
                      f"Latency: avg={avg_lat:.1f}ms p95={p95_lat:.1f}ms p99={p99_lat:.1f}ms | "
                      f"Success: {success_rate*100:.1f}% ({success_count}/{total_in_window})",
                      end="", flush=True)
            
            time.sleep(interval)
            
    except KeyboardInterrupt:
        print("\n\n" + "=" * 80)
        print("Monitoring stopped")
        print("=" * 80)
        
        if len(latencies) > 0:
            valid_lats = [l for l in latencies if l > 0]
            if valid_lats:
                print(f"\nFinal Statistics (last {len(valid_lats)} requests):")
                print(f"  Total requests:    {request_count}")
                print(f"  Successful:        {sum(successes)}")
                print(f"  Errors:            {sum(errors)}")
                print(f"  Success rate:      {sum(successes)/len(successes)*100:.2f}%")
                print(f"  Avg latency:       {statistics.mean(valid_lats):.2f} ms")
                print(f"  p50 latency:       {percentile(valid_lats, 0.50):.2f} ms")
                print(f"  p95 latency:       {percentile(valid_lats, 0.95):.2f} ms")
                print(f"  p99 latency:       {percentile(valid_lats, 0.99):.2f} ms")
                print(f"  Min latency:       {min(valid_lats):.2f} ms")
                print(f"  Max latency:       {max(valid_lats):.2f} ms")

def main():
    parser = argparse.ArgumentParser(description="Monitor API performance in real-time")
    parser.add_argument(
        "--endpoint",
        default="http://127.0.0.1:8000",
        help="API endpoint URL (default: http://127.0.0.1:8000)"
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=1.0,
        help="Monitoring interval in seconds (default: 1.0)"
    )
    parser.add_argument(
        "--window",
        type=int,
        default=100,
        help="Sliding window size for metrics (default: 100)"
    )
    args = parser.parse_args()
    
    monitor_api(args.endpoint, args.interval, args.window)

if __name__ == "__main__":
    main()

