import argparse
import tensorflow as tf

from common import get_memory_mb, save_result_row, run_benchmark


def main():
    parser = argparse.ArgumentParser(description="TensorFlow synthetic ResNet-50 CPU benchmark")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--measure-steps", type=int, default=20)
    parser.add_argument("--duration-sec", type=float, default=None)
    parser.add_argument("--intra-threads", type=int, default=4)
    parser.add_argument("--inter-threads", type=int, default=2)
    args = parser.parse_args()

    tf.config.threading.set_intra_op_parallelism_threads(args.intra_threads)
    tf.config.threading.set_inter_op_parallelism_threads(args.inter_threads)

    model = tf.keras.applications.ResNet50(weights=None, include_top=True)
    batch = tf.random.normal((args.batch_size, 224, 224, 3))

    @tf.function
    def run_one_batch():
        return model(batch, training=False)

    def run_inference():
        _ = run_one_batch()

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
        "framework": "tensorflow",
        "mode": "inference",
        "data_type": "synthetic",
        "batch_size": args.batch_size,
        "intra_threads": args.intra_threads,
        "inter_threads": args.inter_threads,
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
    filename = f"tensorflow_synth_bs{args.batch_size}_intra{args.intra_threads}_inter{args.inter_threads}.csv"
    save_result_row(result, filename)


if __name__ == "__main__":
    main()
