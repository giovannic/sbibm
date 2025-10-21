"""Snapshot test for SIR task ODE solver numerical correctness.

This test ensures that the Diffrax implementation of the SIR ODE
solver produces numerically equivalent results to the baseline
torchdiffeq implementation.

Snapshots are generated from randomly sampled parameter sets with
a fixed seed to ensure reproducible testing across implementations.
"""

import json
from pathlib import Path

import pyro
import pytest
import torch

from sbibm.tasks import get_task

SNAPSHOT_FILE = Path(__file__).parent / "sir_snapshot.json"
TOLERANCE_RTOL = 1e-5
TOLERANCE_ATOL = 1e-8


@pytest.fixture
def sir_snapshot():
    """Load or generate SIR ODE snapshot for testing.

    This fixture:
    1. Checks if snapshot file exists on disk
    2. If not, generates snapshot using current implementation
    3. Validates snapshot has consistent shapes and is finite
    4. Returns (parameters, trajectories) tuple

    The snapshot is saved as JSON with fixed random seed (123)
    for reproducibility across implementations.
    """
    if SNAPSHOT_FILE.exists():
        # Load snapshot from file
        with open(SNAPSHOT_FILE, "r") as f:
            data = json.load(f)
        parameters = torch.tensor(data["parameters"], dtype=torch.float32)
        trajectories = torch.tensor(
            data["trajectories"], dtype=torch.float32
        )
    else:
        # Generate snapshot using current implementation with fixed seed
        pyro.util.set_rng_seed(123)
        torch.manual_seed(123)

        task = get_task("sir")
        prior = task.get_prior()

        # Sample 5 parameter sets from prior with fixed seed
        num_samples = 5
        parameters = prior(num_samples=num_samples)

        # Solve ODE trajectories deterministically
        trajectories = task.solve_ode_trajectories(parameters)

        # Save snapshot to file
        with open(SNAPSHOT_FILE, "w") as f:
            json.dump(
                {
                    "parameters": parameters.tolist(),
                    "trajectories": trajectories.tolist(),
                },
                f,
                indent=2,
            )

    # Validate snapshot consistency and finiteness
    assert parameters.shape == (5, 2), (
        f"Expected parameters shape (5, 2), got {parameters.shape}"
    )
    assert trajectories.shape[0] == 5, (
        f"Expected 5 trajectory samples, got {trajectories.shape[0]}"
    )
    assert trajectories.shape[1] == 3, (
        f"Expected 3 state dims (S, I, R), got {trajectories.shape[1]}"
    )
    assert torch.isfinite(parameters).all(), (
        "Snapshot parameters contain NaN or Inf"
    )
    assert torch.isfinite(trajectories).all(), (
        "Snapshot trajectories contain NaN or Inf"
    )

    return parameters, trajectories


def test_sir_snapshot_trajectories(sir_snapshot):
    """Test SIR ODE solver produces trajectories matching baseline.

    This snapshot test verifies that the current implementation
    produces numerically equivalent ODE trajectories compared to
    the baseline snapshot.

    Uses allclose with rtol=1e-5 and atol=1e-8 for comparison.
    """
    baseline_parameters, baseline_trajectories = sir_snapshot

    task = get_task("sir")

    # Solve ODE on fixed baseline parameters
    trajectories = task.solve_ode_trajectories(baseline_parameters)

    # Check numerical equivalence using allclose
    assert torch.allclose(
        trajectories,
        baseline_trajectories,
        rtol=TOLERANCE_RTOL,
        atol=TOLERANCE_ATOL,
    ), (
        f"Trajectories differ from baseline beyond tolerance "
        f"(rtol={TOLERANCE_RTOL}, atol={TOLERANCE_ATOL})"
    )
