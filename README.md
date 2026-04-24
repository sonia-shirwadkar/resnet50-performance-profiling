# ResNet-50 CPU Performance Benchmark  
**PyTorch vs TensorFlow Inference Profiling Toolkit**

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
