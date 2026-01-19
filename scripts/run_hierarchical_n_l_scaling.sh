#!/bin/bash

# Default values
DEVICE="cpu"
NUM_OBSERVATIONS=10

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --device)
      DEVICE="$2"
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

# n_l values to benchmark
N_L_VALUES=(1 2 5 10 20 50 100)

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
echo "num_observations: $NUM_OBSERVATIONS"
echo "n_l values to test: ${N_L_VALUES[*]}"
echo ""

for N_L in "${N_L_VALUES[@]}"; do
  echo "========================================"
  echo "Running benchmarks for n_l=$N_L"
  echo "========================================"

  # Regenerate observations for this n_l value
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

  # Run benchmarks for all algorithms and tasks with this n_l
  for ALGORITHM in "${ALGORITHMS[@]}"; do
    for TASK in "${TASKS[@]}"; do
      for OBS in $(seq 1 $NUM_OBSERVATIONS); do
        echo "Running benchmark for task: $TASK with algorithm: $ALGORITHM, n_l: $N_L, observation: $OBS"
        python scripts/run_hierarchical_benchmark.py \
          --task "$TASK" \
          --algorithm "$ALGORITHM" \
          --num_simulations 10000 \
          --num_observation "$OBS" \
          --output_dir test_results_n_l_scaling \
          --device "$DEVICE" \
          --n_l "$N_L" \
          --seed 42 \
          --num_samples 1000
      done
    done
  done

  echo ""
done

echo "All n_l scaling benchmarks completed!"
