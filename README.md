# ResNet-50 CPU Performance Benchmark
### PyTorch vs TensorFlow Inference Profiling Toolkit

This project benchmarks **CPU inference performance of ResNet-50** across **PyTorch and TensorFlow** using both **synthetic tensors** and **real image data pipelines**.

It is designed as a **performance engineering study** to analyze:

- framework efficiency
- threading scalability
- batch size impact
- data pipeline overhead
- latency vs throughput tradeoffs
- reproducible timed benchmarking methodology

The benchmark supports:

- quick smoke tests
- full configuration sweeps
- fixed-duration performance runs (recommended)

---

# Project Goals

This benchmark answers questions such as:

- Which framework achieves higher CPU throughput?
- How does batch size affect latency and throughput?
- How well does inference scale with thread count?
- How much overhead does real data loading introduce?
- How stable are measurements over fixed time windows?

The design mirrors workflows used in:

- MLPerf-style benchmarking
- CPU inference optimization
- framework comparison studies
- production inference profiling

---

# Repository Structure

```
resnet50-performance-profiling/
│
├── src/
│   ├── run_benchmarks.py
│   ├── pytorch_synth.py
│   ├── pytorch_realdata.py
│   ├── tensorflow_synth.py
│   ├── tensorflow_realdata.py
│   ├── plot_results.py
│   └── report_results.py
│
├── results/
│
├── requirements.txt
└── README.md
```

---

# Environment Setup

## Step 1 — Clone repository

```
git clone <your_repo_url>
cd resnet50-performance-profiling
```

---

## Step 2 — Create virtual environment

### Windows

```
python -m venv venv
venv\Scripts\activate
```

### Mac / Linux

```
python3 -m venv venv
source venv/bin/activate
```

---

## Step 3 — Install dependencies

```
pip install -r requirements.txt
```

---

# Benchmark Modes

The runner supports three execution styles:

| Mode | Purpose | Runtime |
|------|--------|---------|
| Quick | Smoke test | ~2–5 minutes |
| Full | Parameter sweep | ~10–30 minutes |
| Timed (recommended) | Stable measurements | user-controlled |

---

# Recommended Benchmark (Timed Mode)

Runs equal-duration experiments for:

- PyTorch synthetic inference
- PyTorch real-data inference
- TensorFlow synthetic inference
- TensorFlow real-data inference

Run:

```
python src/run_benchmarks.py --timed --group-duration-sec 600
```

This produces approximately:

```
10 min PyTorch synthetic
10 min PyTorch real
10 min TensorFlow synthetic
10 min TensorFlow real
```

Total measured benchmarking time:

```
≈ 40 minutes
```

Actual wall-clock runtime may be slightly longer due to dataset loading and model initialization overhead.

---

# Short Test Run (Recommended First)

To validate setup:

```
python src/run_benchmarks.py --timed --group-duration-sec 60
```

Runs a ~4-minute benchmark.

---

# Quick Mode (Fast Smoke Test)

```
python src/run_benchmarks.py --quick
```

Useful for:

- verifying environment setup
- testing scripts
- validating dataset downloads
- checking plotting pipeline

---

# Full Sweep Mode

Runs extended parameter combinations:

```
python src/run_benchmarks.py
```

Tests:

- multiple batch sizes
- multiple thread counts
- synthetic vs real data
- TensorFlow intra/inter thread configs
- PyTorch dataloader workers

---

# What Each Benchmark Measures

Each experiment records:

| Metric | Meaning |
|-------|---------|
| throughput_images_per_sec | images processed per second |
| avg_batch_latency_sec | latency per batch |
| total_time_sec | measured runtime |
| actual_measured_steps | iterations completed |
| memory_before_mb | RSS before execution |
| memory_after_mb | RSS after execution |

These enable:

- throughput comparison
- latency comparison
- memory footprint analysis
- scaling behavior evaluation

---

# Synthetic vs Real Data Benchmarks

Synthetic mode:

```
random tensors → model inference
```

Measures:

```
pure compute performance
```

Real-data mode:

```
CIFAR10 → resize → preprocess → inference
```

Measures:

```
compute + input pipeline overhead
```

Comparing both isolates:

```
framework pipeline efficiency
```

---

# Threading Parameters

## PyTorch

```
--threads N
```

Controls:

```
intra-op CPU parallelism
```

Equivalent to:

```
torch.set_num_threads(N)
```

---

## TensorFlow

```
--intra-threads N
--inter-threads M
```

Controls:

| Parameter | Role |
|----------|------|
| intra | threads per operation |
| inter | parallel operations |

Example:

```
--intra-threads 4 --inter-threads 2
```

---

# Output Files

Each benchmark run creates:

```
results/run_TIMESTAMP/
```

Containing:

```
combined_results.csv
summary_report.txt
summary_report.json
performance plots (.png)
```

---

# Generated Plots

Plots include:

```
synthetic_throughput_vs_batch.png
realdata_throughput_vs_batch.png
pytorch_synthetic_threads.png
tensorflow_synthetic_threads.png
pytorch_realdata_threads.png
tensorflow_realdata_threads.png
```

These visualize:

- scaling efficiency
- framework comparison
- pipeline overhead
- thread utilization

---

# Example Summary Report

Generated automatically:

```
results/run_TIMESTAMP/summary_report.txt
```

Contains:

- best throughput configuration
- lowest latency configuration
- framework averages
- total benchmark runtime
- per-framework timing breakdown

---

# Example Custom Experiments

Run PyTorch synthetic benchmark:

```
python src/pytorch_synth.py \
    --batch-size 8 \
    --threads 4 \
    --duration-sec 120
```

Run TensorFlow real-data benchmark:

```
python src/tensorflow_realdata.py \
    --batch-size 8 \
    --intra-threads 4 \
    --inter-threads 2 \
    --duration-sec 120
```

---

# Why Timed Benchmarks Matter

Fixed-step benchmarks vary with system load.

Timed benchmarks ensure:

```
equal runtime per framework
stable throughput estimates
fair comparisons
repeatable results
```

This mirrors production benchmarking workflows used in performance engineering.

---

# Future Extensions

Possible improvements:

- GPU benchmarking support
- ONNX Runtime comparison
- OpenVINO inference runs
- MLPerf-style reporting
- NUMA-aware thread pinning
- perf / VTune integration

---

# Author

Performance benchmarking toolkit created for studying:

```
CPU inference scaling behavior
framework efficiency differences
input pipeline bottlenecks
thread-level parallelism effects
```

Designed as a practical performance engineering reference workflow.