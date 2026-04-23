import argparse
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
RESULTS_ROOT = ROOT / "results"


def create_run_directory(label: str = None):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = f"_{label}" if label else ""
    run_dir = RESULTS_ROOT / f"run_{timestamp}{suffix}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def run_cmd(cmd, run_dir):
    env = dict(os.environ)
    env["RESULTS_DIR"] = str(run_dir)

    print("\\n" + "=" * 80)
    print("Running:", " ".join(cmd))
    print("Saving results to:", run_dir)
    print("=" * 80)

    subprocess.run(cmd, check=True, cwd=str(ROOT), env=env)


def split_duration(total_duration_sec: float, num_configs: int) -> float:
    if num_configs <= 0:
        raise ValueError("num_configs must be positive")
    return total_duration_sec / num_configs


def main():
    parser = argparse.ArgumentParser(description="Run ResNet-50 CPU benchmarks and generate report")
    parser.add_argument("--quick", action="store_true", help="Run fewer step-based experiments for a faster smoke test")
    parser.add_argument("--timed", action="store_true", help="Use timed benchmark groups instead of fixed measured steps")
    parser.add_argument(
        "--group-duration-sec",
        type=float,
        default=600,
        help="In timed mode, target measured time per group: pytorch synth, pytorch real, tensorflow synth, tensorflow real",
    )
    parser.add_argument("--label", type=str, default=None, help="Optional label appended to the run folder name")
    args = parser.parse_args()

    py = sys.executable
    run_dir = create_run_directory(args.label)

    print(f"\\nCreated benchmark results folder:\\n{run_dir}\\n")

    if args.timed:
        # Total measured time target is roughly 4 * group_duration_sec.
        # Each group duration is split across that group's configurations.
        synth_batches = [1, 8]
        real_batches = [8]
        pt_threads = [2, 4]
        tf_thread_pairs = [(2, 1), (4, 2)]
        pt_workers = [0]
        tf_prefetch_values = [False]

        pt_synth_configs = [(b, t) for b in synth_batches for t in pt_threads]
        pt_real_configs = [(b, t, w) for b in real_batches for t in pt_threads for w in pt_workers]
        tf_synth_configs = [(b, intra, inter) for b in synth_batches for intra, inter in tf_thread_pairs]
        tf_real_configs = [
            (b, intra, inter, prefetch)
            for b in real_batches
            for intra, inter in tf_thread_pairs
            for prefetch in tf_prefetch_values
        ]

        pt_synth_duration = split_duration(args.group_duration_sec, len(pt_synth_configs))
        pt_real_duration = split_duration(args.group_duration_sec, len(pt_real_configs))
        tf_synth_duration = split_duration(args.group_duration_sec, len(tf_synth_configs))
        tf_real_duration = split_duration(args.group_duration_sec, len(tf_real_configs))

        print("Timed mode:")
        print(f"  Target per group: {args.group_duration_sec:.1f} sec")
        print(f"  Approx total measured time: {4 * args.group_duration_sec:.1f} sec")
        print(f"  PyTorch synthetic configs: {len(pt_synth_configs)}, seconds/config: {pt_synth_duration:.1f}")
        print(f"  PyTorch real configs: {len(pt_real_configs)}, seconds/config: {pt_real_duration:.1f}")
        print(f"  TensorFlow synthetic configs: {len(tf_synth_configs)}, seconds/config: {tf_synth_duration:.1f}")
        print(f"  TensorFlow real configs: {len(tf_real_configs)}, seconds/config: {tf_real_duration:.1f}")

        for batch, thr in pt_synth_configs:
            run_cmd([
                py, str(SRC / "pytorch_synth.py"),
                "--batch-size", str(batch),
                "--threads", str(thr),
                "--duration-sec", str(pt_synth_duration),
                "--warmup-steps", "5",
            ], run_dir)

        for batch, thr, workers in pt_real_configs:
            run_cmd([
                py, str(SRC / "pytorch_realdata.py"),
                "--batch-size", str(batch),
                "--threads", str(thr),
                "--num-workers", str(workers),
                "--duration-sec", str(pt_real_duration),
                "--warmup-steps", "3",
                "--subset-size", "500",
            ], run_dir)

        for batch, intra, inter in tf_synth_configs:
            run_cmd([
                py, str(SRC / "tensorflow_synth.py"),
                "--batch-size", str(batch),
                "--intra-threads", str(intra),
                "--inter-threads", str(inter),
                "--duration-sec", str(tf_synth_duration),
                "--warmup-steps", "5",
            ], run_dir)

        for batch, intra, inter, prefetch in tf_real_configs:
            cmd = [
                py, str(SRC / "tensorflow_realdata.py"),
                "--batch-size", str(batch),
                "--intra-threads", str(intra),
                "--inter-threads", str(inter),
                "--duration-sec", str(tf_real_duration),
                "--warmup-steps", "3",
                "--subset-size", "500",
            ]
            if prefetch:
                cmd.append("--prefetch")
            run_cmd(cmd, run_dir)

    else:
        if args.quick:
            synth_batches = [1, 8]
            real_batches = [8]
            pt_threads = [2, 4]
            tf_thread_pairs = [(2, 1), (4, 2)]
            pt_workers = [0]
            tf_prefetch_values = [False]
            synth_measure_steps = 8
            real_measure_steps = 5
        else:
            synth_batches = [1, 8, 16]
            real_batches = [8, 16]
            pt_threads = [2, 4, 8]
            tf_thread_pairs = [(2, 1), (4, 2), (8, 2)]
            pt_workers = [0, 2]
            tf_prefetch_values = [False, True]
            synth_measure_steps = 20
            real_measure_steps = 10

        for batch in synth_batches:
            for thr in pt_threads:
                run_cmd([
                    py, str(SRC / "pytorch_synth.py"),
                    "--batch-size", str(batch),
                    "--threads", str(thr),
                    "--measure-steps", str(synth_measure_steps),
                    "--warmup-steps", "5",
                ], run_dir)

            for intra, inter in tf_thread_pairs:
                run_cmd([
                    py, str(SRC / "tensorflow_synth.py"),
                    "--batch-size", str(batch),
                    "--intra-threads", str(intra),
                    "--inter-threads", str(inter),
                    "--measure-steps", str(synth_measure_steps),
                    "--warmup-steps", "5",
                ], run_dir)

        for batch in real_batches:
            for thr in pt_threads[:2]:
                for workers in pt_workers:
                    run_cmd([
                        py, str(SRC / "pytorch_realdata.py"),
                        "--batch-size", str(batch),
                        "--threads", str(thr),
                        "--num-workers", str(workers),
                        "--measure-steps", str(real_measure_steps),
                        "--warmup-steps", "3",
                        "--subset-size", "500",
                    ], run_dir)

            for intra, inter in tf_thread_pairs[:2]:
                for prefetch in tf_prefetch_values:
                    cmd = [
                        py, str(SRC / "tensorflow_realdata.py"),
                        "--batch-size", str(batch),
                        "--intra-threads", str(intra),
                        "--inter-threads", str(inter),
                        "--measure-steps", str(real_measure_steps),
                        "--warmup-steps", "3",
                        "--subset-size", "500",
                    ]
                    if prefetch:
                        cmd.append("--prefetch")
                    run_cmd(cmd, run_dir)

    run_cmd([py, str(SRC / "plot_results.py")], run_dir)
    run_cmd([py, str(SRC / "report_results.py")], run_dir)

    print("\\nBenchmark complete.")
    print(f"Results saved in:\\n{run_dir}\\n")


if __name__ == "__main__":
    main()
