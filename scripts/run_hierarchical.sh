#!/bin/bash

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

ALGORITHMS=("snpe" "deepset")

for ALGORITHM in "${ALGORITHMS[@]}"; do
  for TASK in "${TASKS[@]}"; do
    echo "Running benchmark for task: $TASK with algorithm: $ALGORITHM"
    python scripts/run_hierarchical_benchmark.py \
      --task "$TASK" \
      --algorithm "$ALGORITHM" \
      --num_simulations 1000 \
      --num_observation 1 \
      --output_dir test_results \
      --seed 42 \
      --num_samples 100

    python scripts/run_hierarchical_benchmark.py \
      --task "$TASK" \
      --algorithm "$ALGORITHM" \
      --num_simulations 5000 \
      --num_observation 1 \
      --output_dir test_results \
      --seed 42 \
      --num_samples 100

    python scripts/run_hierarchical_benchmark.py \
      --task "$TASK" \
      --algorithm "$ALGORITHM" \
      --num_simulations 10000 \
      --num_observation 1 \
      --output_dir test_results \
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
