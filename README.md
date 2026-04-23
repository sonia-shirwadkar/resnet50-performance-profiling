# ResNet-50 Performance Profiling on CPU

This project compares CPU inference performance of ResNet-50 in PyTorch and TensorFlow on a local laptop without a GPU.

## Goals

- run pretrained ResNet-50 locally
- measure inference latency
- estimate throughput
- observe warm-up effects
- compare PyTorch and TensorFlow behavior
- practice ML performance analysis in a small GitHub-ready project

## Project Structure

```text
resnet50-performance-profiling/
├── README.md
├── requirements.txt
├── .gitignore
├── scripts/
├── results/
├── profiling/
└── data/