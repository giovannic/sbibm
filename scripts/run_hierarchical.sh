#!/bin/bash

# Default values
DEVICE="cpu"
N_L=5
START_OBS=1
END_OBS=10
CUDA_DEVICE=""

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
    --start_obs)
      START_OBS="$2"
      shift 2
      ;;
    --end_obs)
      END_OBS="$2"
      shift 2
      ;;
    --cuda_device)
      CUDA_DEVICE="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# Set CUDA_VISIBLE_DEVICES if specified
if [[ -n "$CUDA_DEVICE" ]]; then
  export CUDA_VISIBLE_DEVICES="$CUDA_DEVICE"
  echo "CUDA_VISIBLE_DEVICES: $CUDA_DEVICE"
fi

# Loop through all hierarchical tasks except bernoulli_glm
TASKS=(
  "hierarchical_gaussian_linear"
  "hierarchical_gaussian_linear_uniform"
  "hierarchical_gaussian_mixture"
  "hierarchical_sir"
  "hierarchical_slcp"
  "hierarchical_two_moons"
)

ALGORITHMS=("bottom_up" "snpe" "snpe_5r" "deepset" "fmpe" "fmpe_transformer" "simformer")

echo "Device: $DEVICE"
echo "n_l (local contexts): $N_L"
echo "Observations: $START_OBS to $END_OBS"
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

for OBS in $(seq $START_OBS $END_OBS); do
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
