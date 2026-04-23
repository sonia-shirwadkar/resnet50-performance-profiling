import os
import time
import json
from typing import Callable, Dict, Any

import pandas as pd
import psutil


RESULTS_DIR = os.environ.get("RESULTS_DIR", "results")


def ensure_results_dir() -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)


def get_memory_mb() -> float:
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)


def benchmark_loop_fixed_steps(run_fn: Callable[[], None], warmup_steps: int, measure_steps: int):
    for _ in range(warmup_steps):
        run_fn()

    start = time.perf_counter()
    for _ in range(measure_steps):
        run_fn()
    end = time.perf_counter()

    total_time = end - start
    avg_step_time = total_time / measure_steps
    return total_time, avg_step_time, measure_steps


def benchmark_loop_timed(run_fn: Callable[[], None], warmup_steps: int, duration_sec: float):
    for _ in range(warmup_steps):
        run_fn()

    measured_steps = 0
    start = time.perf_counter()

    while True:
        elapsed = time.perf_counter() - start
        if measured_steps > 0 and elapsed >= duration_sec:
            break

        run_fn()
        measured_steps += 1

    end = time.perf_counter()
    total_time = end - start
    avg_step_time = total_time / measured_steps
    return total_time, avg_step_time, measured_steps


def run_benchmark(run_fn: Callable[[], None], warmup_steps: int, measure_steps: int, duration_sec):
    if duration_sec is not None:
        return benchmark_loop_timed(
            run_fn=run_fn,
            warmup_steps=warmup_steps,
            duration_sec=duration_sec,
        )

    return benchmark_loop_fixed_steps(
        run_fn=run_fn,
        warmup_steps=warmup_steps,
        measure_steps=measure_steps,
    )


def save_result_row(result: Dict[str, Any], csv_path: str) -> None:
    ensure_results_dir()
    path = os.path.join(RESULTS_DIR, csv_path)
    df = pd.DataFrame([result])
    df.to_csv(path, index=False)


def combine_csvs(output_filename: str = "combined_results.csv") -> str:
    ensure_results_dir()
    csv_files = [
        os.path.join(RESULTS_DIR, f)
        for f in os.listdir(RESULTS_DIR)
        if f.endswith(".csv") and f != output_filename
    ]
    if not csv_files:
        raise FileNotFoundError("No CSV benchmark results found in results/")

    dfs = [pd.read_csv(path) for path in csv_files]
    combined = pd.concat(dfs, ignore_index=True)
    output_path = os.path.join(RESULTS_DIR, output_filename)
    combined.to_csv(output_path, index=False)
    return output_path


def save_json(data: Dict[str, Any], filename: str) -> str:
    ensure_results_dir()
    path = os.path.join(RESULTS_DIR, filename)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    return path
