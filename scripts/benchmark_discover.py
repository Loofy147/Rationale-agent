import time
import numpy as np
from mas.engines.discover import DiscoverEngine

def benchmark_discover_engine():
    """
    Benchmarks the performance of the DiscoverEngine.
    """
    print("Initializing DiscoverEngine (this will load the synthesis model)...")
    engine = DiscoverEngine()

    benchmark_topics = ["image classification", "text-to-speech", "reinforcement learning"]
    latencies = []

    print("\nStarting benchmark...")
    for topic in benchmark_topics:
        start_time = time.time()
        try:
            engine.run(topic)
            end_time = time.time()
            latency = end_time - start_time
            latencies.append(latency)
            print(f"Topic '{topic}' processed in {latency:.2f} seconds.")
        except Exception as e:
            print(f"An error occurred while processing topic '{topic}': {e}")

    if latencies:
        p50 = np.percentile(latencies, 50)
        p90 = np.percentile(latencies, 90)
        p99 = np.percentile(latencies, 99)

        print("\n--- Benchmark Results ---")
        print(f"Number of runs: {len(latencies)}")
        print(f"p50 (Median) Latency: {p50:.2f} seconds")
        print(f"p90 Latency: {p90:.2f} seconds")
        print(f"p99 Latency: {p99:.2f} seconds")

        # Append results to BENCHMARKS.md
        with open("BENCHMARKS.md", "a") as f:
            f.write("## Discover Engine Performance\n\n")
            f.write(f"| Metric | Value (seconds) |\n")
            f.write(f"|---|---|\n")
            f.write(f"| p50 (Median) | {p50:.2f} |\n")
            f.write(f"| p90 | {p90:.2f} |\n")
            f.write(f"| p99 | {p99:.2f} |\n\n")

if __name__ == "__main__":
    # Note: This requires the `numpy` library.
    # Run `poetry add numpy` if not already installed.
    benchmark_discover_engine()
