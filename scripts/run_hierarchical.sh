#!/bin/bash

# Default device
DEVICE="cpu"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --device)
      DEVICE="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# Loop through all hierarchical tasks except bernoulli_glm
TASKS=(
  "hierarchical_gaussian_linear"
  "hierarchical_gaussian_linear_uniform"
  "hierarchical_gaussian_mixture"
  "hierarchical_lotka_volterra"
  "hierarchical_sir"
  "hierarchical_slcp"
  "hierarchical_two_moons"
)

ALGORITHMS=("bottom_up", "snpe" "deepset")

echo "Running benchmarks with device: $DEVICE"

for ALGORITHM in "${ALGORITHMS[@]}"; do
  for TASK in "${TASKS[@]}"; do
    echo "Running benchmark for task: $TASK with algorithm: $ALGORITHM"
    python scripts/run_hierarchical_benchmark.py \
      --task "$TASK" \
      --algorithm "$ALGORITHM" \
      --num_simulations 1000 \
      --num_observation 1 \
      --output_dir test_results \
      --device "$DEVICE" \
      --seed 42 \
      --num_samples 100

    python scripts/run_hierarchical_benchmark.py \
      --task "$TASK" \
      --algorithm "$ALGORITHM" \
      --num_simulations 5000 \
      --num_observation 1 \
      --output_dir test_results \
      --device "$DEVICE" \
      --seed 42 \
      --num_samples 100

    python scripts/run_hierarchical_benchmark.py \
      --task "$TASK" \
      --algorithm "$ALGORITHM" \
      --num_simulations 10000 \
      --num_observation 1 \
      --output_dir test_results \
      --device "$DEVICE" \
      --seed 42 \
      --num_samples 100

    echo "Generating visualization for task: $TASK"
    python scripts/visualize_hierarchical_posterior.py \
      --task "$TASK" \
      --algorithm "$ALGORITHM" \
      --output_path "test_results/${TASK}_${ALGORITHM}_posterior.png"
  done
done

echo "All tasks completed!"
