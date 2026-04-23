import argparse
import torch
from torchvision.models import resnet50

from common import get_memory_mb, save_result_row, run_benchmark


def main():
    parser = argparse.ArgumentParser(description="PyTorch synthetic ResNet-50 CPU benchmark")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--measure-steps", type=int, default=20)
    parser.add_argument("--duration-sec", type=float, default=None)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()

    torch.set_num_threads(args.threads)

    model = resnet50(weights=None)
    model.eval()
    batch = torch.randn(args.batch_size, 3, 224, 224)

    def run_inference():
        with torch.no_grad():
            _ = model(batch)

    memory_before = get_memory_mb()
    total_time, avg_batch_latency, actual_steps = run_benchmark(
        run_fn=run_inference,
        warmup_steps=args.warmup_steps,
        measure_steps=args.measure_steps,
        duration_sec=args.duration_sec,
    )
    memory_after = get_memory_mb()

    images_processed = args.batch_size * actual_steps
    throughput = images_processed / total_time

    result = {
        "framework": "pytorch",
        "mode": "inference",
        "data_type": "synthetic",
        "batch_size": args.batch_size,
        "threads": args.threads,
        "warmup_steps": args.warmup_steps,
        "requested_measure_steps": args.measure_steps,
        "duration_sec": args.duration_sec,
        "actual_measured_steps": actual_steps,
        "total_time_sec": total_time,
        "avg_batch_latency_sec": avg_batch_latency,
        "throughput_images_per_sec": throughput,
        "memory_before_mb": memory_before,
        "memory_after_mb": memory_after,
    }

    print(result)
    save_result_row(result, f"pytorch_synth_bs{args.batch_size}_thr{args.threads}.csv")


if __name__ == "__main__":
    main()
