import argparse
import time
from typing import Any, Dict, List, Optional

import requests


def _get_number(d: Dict[str, Any], key: str) -> Optional[float]:
    v = d.get(key)
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8000/stats")
    parser.add_argument("--interval", type=float, default=1.0)
    parser.add_argument("--duration", type=float, default=30.0)
    parser.add_argument("--out", default="stats.png")
    args = parser.parse_args()

    n = max(1, int(args.duration / args.interval))

    times: List[float] = []
    queue_size: List[Optional[float]] = []
    avg_latency: List[Optional[float]] = []
    avg_e2e_latency: List[Optional[float]] = []
    avg_batch_size: List[Optional[float]] = []
    median_of_medians: List[Optional[float]] = []
    inference_count: List[Optional[float]] = []
    request_count: List[Optional[float]] = []

    start = time.time()

    for i in range(n):
        t0 = time.time()
        try:
            r = requests.get(args.url, timeout=2)
            r.raise_for_status()
            payload = r.json()
        except Exception as e:
            payload = {}
            print(f"request_failed i={i} err={e}")

        times.append(time.time() - start)
        queue_size.append(_get_number(payload, "queue_size"))
        avg_latency.append(_get_number(payload, "avg_latency"))
        avg_e2e_latency.append(_get_number(payload, "avg_e2e_latency"))
        avg_batch_size.append(_get_number(payload, "avg_batch_size"))
        median_of_medians.append(_get_number(payload, "median_of_medians"))
        inference_count.append(_get_number(payload, "inference_count"))
        request_count.append(_get_number(payload, "request_count"))

        dt = time.time() - t0
        sleep_for = args.interval - dt
        if sleep_for > 0:
            time.sleep(sleep_for)

    events_per_sec: List[Optional[float]] = [None] * len(times)
    requests_per_sec: List[Optional[float]] = [None] * len(times)
    for i in range(1, len(times)):
        dt = times[i] - times[i - 1]
        if dt <= 0:
            continue

        if inference_count[i] is not None and inference_count[i - 1] is not None:
            events_per_sec[i] = (inference_count[i] - inference_count[i - 1]) / dt

        if request_count[i] is not None and request_count[i - 1] is not None:
            requests_per_sec[i] = (request_count[i] - request_count[i - 1]) / dt

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(6, 1, figsize=(10, 12), sharex=True)

    axes[0].plot(times, queue_size, label="queue_size")
    axes[0].set_ylabel("queue_size")
    axes[0].grid(True)

    axes[1].plot(times, avg_latency, label="avg_latency")
    axes[1].plot(times, avg_e2e_latency, label="avg_e2e_latency")
    axes[1].set_ylabel("latency")
    axes[1].grid(True)
    axes[1].legend(loc="upper right")

    axes[2].plot(times, avg_batch_size, label="avg_batch_size")
    axes[2].set_ylabel("avg_batch_size")
    axes[2].grid(True)

    axes[3].plot(times, events_per_sec, label="events_per_sec")
    axes[3].set_ylabel("events/sec")
    axes[3].grid(True)

    axes[4].plot(times, requests_per_sec, label="requests_per_sec")
    axes[4].set_ylabel("req/sec")
    axes[4].grid(True)

    axes[5].plot(times, median_of_medians, label="median_of_medians")
    axes[5].set_ylabel("median_of_medians")
    axes[5].set_xlabel("seconds")
    axes[5].grid(True)

    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"wrote_plot {args.out}")


if __name__ == "__main__":
    main()
