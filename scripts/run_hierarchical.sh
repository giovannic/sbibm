#!/bin/bash

# Default values
DEVICE="cpu"
N_L=5
NUM_OBSERVATIONS=10

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --device)
      DEVICE="$2"
      shift 2
      ;;
    --n_l)
      N_L="$2"
      shift 2
      ;;
    --num_observations)
      NUM_OBSERVATIONS="$2"
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
  "hierarchical_sir"
  "hierarchical_slcp"
  "hierarchical_two_moons"
)

ALGORITHMS=("bottom_up" "snpe" "deepset")

echo "Device: $DEVICE"
echo "n_l (local contexts): $N_L"
echo "num_observations: $NUM_OBSERVATIONS"
echo ""

# Regenerate observations if n_l != 5
if [[ $N_L -ne 5 ]]; then
  echo "Regenerating observations for n_l=$N_L..."
  for TASK in "${TASKS[@]}"; do
    echo "Regenerating observations for task: $TASK with n_l=$N_L"
    python sbibm/tasks/"$TASK"/task.py --n_l "$N_L"
    if [[ $? -ne 0 ]]; then
      echo "Error regenerating observations for $TASK"
      exit 1
    fi
  done
  echo "Observation regeneration completed!"
  echo ""
fi

echo "Running benchmarks with device: $DEVICE"

for OBS in $(seq 3 $NUM_OBSERVATIONS); do
  for ALGORITHM in "${ALGORITHMS[@]}"; do
    for TASK in "${TASKS[@]}"; do
      echo "Running benchmark for task: $TASK with algorithm: $ALGORITHM, observation: $OBS"
      python scripts/run_hierarchical_benchmark.py \
        --task "$TASK" \
        --algorithm "$ALGORITHM" \
        --num_simulations 1000 \
        --num_observation "$OBS" \
        --output_dir test_results \
        --device "$DEVICE" \
        --n_l "$N_L" \
        --seed 42 \
        --num_samples 1000

      python scripts/run_hierarchical_benchmark.py \
        --task "$TASK" \
        --algorithm "$ALGORITHM" \
        --num_simulations 5000 \
        --num_observation "$OBS" \
        --output_dir test_results \
        --device "$DEVICE" \
        --n_l "$N_L" \
        --seed 42 \
        --num_samples 1000

      python scripts/run_hierarchical_benchmark.py \
        --task "$TASK" \
        --algorithm "$ALGORITHM" \
        --num_simulations 10000 \
        --num_observation "$OBS" \
        --output_dir test_results \
        --device "$DEVICE" \
        --n_l "$N_L" \
        --seed 42 \
        --num_samples 1000
    done
  done
done

echo "All tasks completed!"
