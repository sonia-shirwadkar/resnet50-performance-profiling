import argparse
import torch
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CIFAR10
from torchvision.models import resnet50
from torchvision import transforms

from common import get_memory_mb, save_result_row, run_benchmark


def main():
    parser = argparse.ArgumentParser(description="PyTorch real-data ResNet-50 CPU benchmark")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--warmup-steps", type=int, default=3)
    parser.add_argument("--measure-steps", type=int, default=10)
    parser.add_argument("--duration-sec", type=float, default=None)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--subset-size", type=int, default=500)
    args = parser.parse_args()

    torch.set_num_threads(args.threads)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    dataset = CIFAR10(root="./data", train=False, download=True, transform=transform)
    dataset = Subset(dataset, range(0, min(args.subset_size, len(dataset))))

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    model = resnet50(weights=None)
    model.eval()
    batches = iter(loader)

    def run_inference():
        nonlocal batches
        try:
            images, _ = next(batches)
        except StopIteration:
            batches = iter(loader)
            images, _ = next(batches)

        with torch.no_grad():
            _ = model(images)

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
        "data_type": "real_cifar10",
        "batch_size": args.batch_size,
        "threads": args.threads,
        "num_workers": args.num_workers,
        "subset_size": args.subset_size,
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
    filename = f"pytorch_real_bs{args.batch_size}_thr{args.threads}_workers{args.num_workers}.csv"
    save_result_row(result, filename)


if __name__ == "__main__":
    main()
