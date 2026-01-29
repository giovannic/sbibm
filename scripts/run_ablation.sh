#!/bin/bash

# Default values
DEVICE="cpu"
N_L=50
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

echo "Device: $DEVICE"
echo "n_l (local contexts): $N_L"
echo "Observations: $START_OBS to $END_OBS"
echo ""

# Regenerate observations for n_l=50
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

echo "Running ablation study with device: $DEVICE"

for TASK in "${TASKS[@]}"; do
  for OBS in $(seq $START_OBS $END_OBS); do
    # Run baseline (no extra flags)
    echo "Running baseline for task: $TASK, observation: $OBS"
    python scripts/run_hierarchical_benchmark.py \
      --task "$TASK" \
      --algorithm "bottom_up" \
      --num_simulations 10000 \
      --num_observation "$OBS" \
      --output_dir results \
      --device "$DEVICE" \
      --n_l "$N_L" \
      --seed 42 \
      --num_samples 1000

    # Run with --mlp flag
    echo "Running with --mlp for task: $TASK, observation: $OBS"
    python scripts/run_hierarchical_benchmark.py \
      --task "$TASK" \
      --algorithm "bottom_up" \
      --num_simulations 10000 \
      --num_observation "$OBS" \
      --output_dir results_mlp \
      --device "$DEVICE" \
      --n_l "$N_L" \
      --seed 42 \
      --num_samples 1000 \
      --mlp

    # Run with --fit_directly flag
    echo "Running with --fit_directly for task: $TASK, observation: $OBS"
    python scripts/run_hierarchical_benchmark.py \
      --task "$TASK" \
      --algorithm "bottom_up" \
      --num_simulations 10000 \
      --num_observation "$OBS" \
      --output_dir results_fit_directly \
      --device "$DEVICE" \
      --n_l "$N_L" \
      --seed 42 \
      --num_samples 1000 \
      --fit_directly
  done
done

echo "Ablation study completed!"
