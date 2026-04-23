import os
import pandas as pd
import matplotlib.pyplot as plt

from common import combine_csvs, RESULTS_DIR


def plot_grouped_by_framework(df, data_type, output_name, title):
    subdf = df[df["data_type"] == data_type]
    if subdf.empty:
        return

    plt.figure(figsize=(8, 5))
    for framework in subdf["framework"].dropna().unique():
        sub = subdf[subdf["framework"] == framework].sort_values("batch_size")
        plt.plot(sub["batch_size"], sub["throughput_images_per_sec"], marker="o", label=framework)

    plt.xlabel("Batch size")
    plt.ylabel("Throughput (images/sec)")
    plt.title(title)
    plt.legend()
    plt.savefig(os.path.join(RESULTS_DIR, output_name))
    plt.close()


def plot_pytorch_threads(df, data_type, output_name):
    pt = df[(df["framework"] == "pytorch") & (df["data_type"] == data_type)]
    if pt.empty or "threads" not in pt.columns:
        return

    plt.figure(figsize=(8, 5))
    for threads in sorted(pt["threads"].dropna().unique()):
        sub = pt[pt["threads"] == threads].sort_values("batch_size")
        plt.plot(sub["batch_size"], sub["throughput_images_per_sec"], marker="o", label=f"threads={int(threads)}")

    plt.xlabel("Batch size")
    plt.ylabel("Throughput (images/sec)")
    plt.title(f"PyTorch {data_type} throughput by thread count")
    plt.legend()
    plt.savefig(os.path.join(RESULTS_DIR, output_name))
    plt.close()


def plot_tensorflow_threads(df, data_type, output_name):
    tf = df[(df["framework"] == "tensorflow") & (df["data_type"] == data_type)]
    if tf.empty:
        return

    plt.figure(figsize=(8, 5))
    for (intra, inter), sub in tf.groupby(["intra_threads", "inter_threads"]):
        sub = sub.sort_values("batch_size")
        plt.plot(sub["batch_size"], sub["throughput_images_per_sec"], marker="o", label=f"intra={int(intra)} inter={int(inter)}")

    plt.xlabel("Batch size")
    plt.ylabel("Throughput (images/sec)")
    plt.title(f"TensorFlow {data_type} throughput by thread config")
    plt.legend()
    plt.savefig(os.path.join(RESULTS_DIR, output_name))
    plt.close()


def main():
    combined_path = combine_csvs()
    df = pd.read_csv(combined_path)

    plot_grouped_by_framework(df, "synthetic", "synthetic_throughput_vs_batch.png", "Synthetic throughput vs batch size")
    plot_grouped_by_framework(df, "real_cifar10", "realdata_throughput_vs_batch.png", "Real-data throughput vs batch size")
    plot_pytorch_threads(df, "synthetic", "pytorch_synthetic_threads.png")
    plot_tensorflow_threads(df, "synthetic", "tensorflow_synthetic_threads.png")
    plot_pytorch_threads(df, "real_cifar10", "pytorch_realdata_threads.png")
    plot_tensorflow_threads(df, "real_cifar10", "tensorflow_realdata_threads.png")

    print(f"Combined results saved to: {combined_path}")
    print(f"Plots saved to: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
