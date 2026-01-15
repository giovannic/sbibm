#!/bin/bash

# Default values
N_L=5
NUM_SAMPLES=1000
MAX_LOCAL_CONTEXTS=2
OUTPUT_DIR="results/prior_predictive"
SEED=42

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --n_l)
      N_L="$2"
      shift 2
      ;;
    --num_samples)
      NUM_SAMPLES="$2"
      shift 2
      ;;
    --max_local_contexts)
      MAX_LOCAL_CONTEXTS="$2"
      shift 2
      ;;
    --output_dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --seed)
      SEED="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# Loop through all hierarchical tasks
TASKS=(
  "hierarchical_gaussian_linear"
  "hierarchical_gaussian_linear_uniform"
  "hierarchical_gaussian_mixture"
  "hierarchical_lotka_volterra"
  "hierarchical_sir"
  "hierarchical_slcp"
  "hierarchical_two_moons"
)

echo "Prior Checks (Parameters + Observations)"
echo "========================================"
echo "n_l (local contexts): $N_L"
echo "num_samples: $NUM_SAMPLES"
echo "max_local_contexts: $MAX_LOCAL_CONTEXTS"
echo "output_dir: $OUTPUT_DIR"
echo "seed: $SEED"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Regenerate observations to ensure consistency with current prior
echo "Regenerating observations for n_l=$N_L..."
for TASK in "${TASKS[@]}"; do
  echo "Regenerating observations for task: $TASK with n_l=$N_L"
  python sbibm/tasks/"$TASK"/task.py --n_l "$N_L"
done
echo "Observation regeneration completed!"
echo ""

for TASK in "${TASKS[@]}"; do
  echo "Running prior check (parameters) for task: $TASK"
  python scripts/visualize_hierarchical_prior.py \
    --task "$TASK" \
    --n_l "$N_L" \
    --num_samples "$NUM_SAMPLES" \
    --num_observation 1 \
    --max_local_contexts "$MAX_LOCAL_CONTEXTS" \
    --output_path "${OUTPUT_DIR}/${TASK}_prior.png" \
    --seed "$SEED"

  if [[ $? -ne 0 ]]; then
    echo "Error running prior check for $TASK"
    exit 1
  fi
  echo ""
done

echo "Prior checks (parameters) completed!"
echo ""

# Run prior predictive checks (observations)
for TASK in "${TASKS[@]}"; do
  echo "Running prior predictive check (observations) for task: $TASK"
  python scripts/visualize_hierarchical_prior_predictive.py \
    --task "$TASK" \
    --n_l "$N_L" \
    --num_samples "$NUM_SAMPLES" \
    --num_observation 1 \
    --max_local_contexts "$MAX_LOCAL_CONTEXTS" \
    --output_path "${OUTPUT_DIR}/${TASK}_prior_predictive.png" \
    --seed "$SEED"

  if [[ $? -ne 0 ]]; then
    echo "Error running prior predictive check for $TASK"
    exit 1
  fi
  echo ""
done

echo "All prior checks completed!"
echo "Results saved to: $OUTPUT_DIR"
echo "  - *_prior.png: Prior distribution of parameters"
echo "  - *_prior_predictive.png: Prior predictive distribution of observations"
