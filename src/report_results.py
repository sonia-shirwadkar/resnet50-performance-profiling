import os
import pandas as pd

from common import combine_csvs, RESULTS_DIR, save_json


def main():
    combined_path = combine_csvs()
    df = pd.read_csv(combined_path)

    measured_time_group = df.groupby(["framework", "data_type"])["total_time_sec"].sum()

    summary = {
        "num_runs": int(len(df)),
        "frameworks": sorted([str(x) for x in df["framework"].dropna().unique()]),
        "data_types": sorted([str(x) for x in df["data_type"].dropna().unique()]),
        "total_measured_time_sec": float(df["total_time_sec"].sum()),
        "best_throughput_run": df.sort_values("throughput_images_per_sec", ascending=False).head(1).to_dict(orient="records")[0],
        "lowest_latency_run": df.sort_values("avg_batch_latency_sec", ascending=True).head(1).to_dict(orient="records")[0],
        "average_throughput_by_framework": df.groupby("framework")["throughput_images_per_sec"].mean().to_dict(),
        "average_latency_by_framework": df.groupby("framework")["avg_batch_latency_sec"].mean().to_dict(),
        "measured_time_by_framework_and_data_type": {str(k): float(v) for k, v in measured_time_group.items()},
    }

    save_json(summary, "summary_report.json")

    lines = []
    lines.append("ResNet-50 CPU benchmark summary")
    lines.append("=" * 35)
    lines.append(f"Total runs: {summary['num_runs']}")
    lines.append(f"Total measured benchmark time: {summary['total_measured_time_sec']:.2f} sec")
    lines.append(f"Frameworks: {', '.join(summary['frameworks'])}")
    lines.append(f"Data types: {', '.join(summary['data_types'])}")
    lines.append("")
    lines.append("Measured time by framework/data type:")
    for key, value in summary["measured_time_by_framework_and_data_type"].items():
        lines.append(f"  {key}: {value:.2f} sec")
    lines.append("")
    lines.append("Best throughput run:")
    for k, v in summary["best_throughput_run"].items():
        lines.append(f"  {k}: {v}")
    lines.append("")
    lines.append("Lowest latency run:")
    for k, v in summary["lowest_latency_run"].items():
        lines.append(f"  {k}: {v}")
    lines.append("")
    lines.append("Average throughput by framework:")
    for k, v in summary["average_throughput_by_framework"].items():
        lines.append(f"  {k}: {v:.4f}")
    lines.append("")
    lines.append("Average latency by framework:")
    for k, v in summary["average_latency_by_framework"].items():
        lines.append(f"  {k}: {v:.6f}")

    report_path = os.path.join(RESULTS_DIR, "summary_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\\n".join(lines))

    print("\\n".join(lines))
    print(f"\\nSaved text report to: {report_path}")
    print(f"Saved JSON report to: {os.path.join(RESULTS_DIR, 'summary_report.json')}")
    print(f"Combined CSV: {combined_path}")


if __name__ == "__main__":
    main()
