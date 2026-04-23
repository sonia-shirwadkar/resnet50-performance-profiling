import argparse
import tensorflow as tf

from common import get_memory_mb, save_result_row, run_benchmark


def main():
    parser = argparse.ArgumentParser(description="TensorFlow real-data ResNet-50 CPU benchmark")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--warmup-steps", type=int, default=3)
    parser.add_argument("--measure-steps", type=int, default=10)
    parser.add_argument("--duration-sec", type=float, default=None)
    parser.add_argument("--intra-threads", type=int, default=4)
    parser.add_argument("--inter-threads", type=int, default=2)
    parser.add_argument("--subset-size", type=int, default=500)
    parser.add_argument("--prefetch", action="store_true")
    args = parser.parse_args()

    tf.config.threading.set_intra_op_parallelism_threads(args.intra_threads)
    tf.config.threading.set_inter_op_parallelism_threads(args.inter_threads)

    (_, _), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    x_test = x_test[: args.subset_size]
    y_test = y_test[: args.subset_size]

    def preprocess(image, label):
        image = tf.image.resize(image, (224, 224))
        image = tf.keras.applications.resnet.preprocess_input(tf.cast(image, tf.float32))
        return image, label

    dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    dataset = dataset.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(args.batch_size)
    if args.prefetch:
        dataset = dataset.prefetch(tf.data.AUTOTUNE)

    model = tf.keras.applications.ResNet50(weights=None, include_top=True)
    iterator = iter(dataset)

    @tf.function
    def run_one_batch(images):
        return model(images, training=False)

    def run_inference():
        nonlocal iterator
        try:
            images, _ = next(iterator)
        except StopIteration:
            iterator = iter(dataset)
            images, _ = next(iterator)
        _ = run_one_batch(images)

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
        "data_type": "real_cifar10",
        "batch_size": args.batch_size,
        "intra_threads": args.intra_threads,
        "inter_threads": args.inter_threads,
        "prefetch": args.prefetch,
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
    filename = f"tensorflow_real_bs{args.batch_size}_intra{args.intra_threads}_inter{args.inter_threads}_prefetch{int(args.prefetch)}.csv"
    save_result_row(result, filename)


if __name__ == "__main__":
    main()
