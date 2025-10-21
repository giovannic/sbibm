"""Benchmark test for SIR task ODE solver performance.

This test measures the performance of the SIR ODE solver on solving
multiple parameter sets. It serves as a baseline to measure the
speedup gained by migrating from torchdiffeq to Diffrax.

Run with: pytest tests/tasks/sir/test_sir_benchmark.py -v -m benchmark
"""

import pyro
import pytest
import torch

from sbibm.tasks import get_task

pyro.util.set_rng_seed(42)


@pytest.mark.benchmark(
    group="sir_ode_solver",
    min_rounds=3,
)
@pytest.mark.skip_by_default
def test_sir_ode_solver_benchmark(benchmark):
    """Benchmark SIR ODE solver on 10 parameter sets from prior.

    This test:
    - Loads the SIR task
    - Samples 10 parameter sets from the prior distribution
    - Benchmarks the ODE solver execution time
    - Provides baseline for performance comparison

    The benchmark measures wall-clock time for solving 10 ODE systems
    deterministically, which is the computational bottleneck for
    SIR inference.
    """
    task = get_task("sir")
    prior = task.get_prior()

    # Generate 10 parameter sets from prior with fixed seed
    num_samples = 10
    parameters = prior(num_samples=num_samples)

    # Benchmark the ODE solver execution
    def run_ode_solver():
        trajectories = task.solve_ode_trajectories(parameters)
        return trajectories

    benchmark(run_ode_solver)
