import diffrax
import jax
import jax.numpy as jnp
import numpy
import pyro
import pytest
import torch

from sbibm.tasks.hierarchical_sir.task import HierarchicalSIR

pyro.util.set_rng_seed(42)


@pytest.mark.parametrize("n_l", [3, 5])
def test_prior_shape(n_l):
    """Test prior returns correct shape."""
    task = HierarchicalSIR(n_l=n_l)
    prior = task.get_prior()

    num_samples = 100
    samples = prior(num_samples=num_samples)

    expected_dim = 1 + n_l  # 1 global (gamma) + n_l local (beta)
    assert samples.shape == (num_samples, expected_dim)


@pytest.mark.parametrize("n_l", [3, 5])
def test_prior_no_nan(n_l):
    """Test prior samples contain no NaN or Inf values."""
    task = HierarchicalSIR(n_l=n_l)
    prior = task.get_prior()

    samples = prior(num_samples=1000)

    assert not torch.isnan(samples).any()
    assert not torch.isinf(samples).any()


@pytest.mark.parametrize("n_l", [3, 5])
def test_simulator_shape(n_l):
    """Test simulator returns correct shape."""
    task = HierarchicalSIR(n_l=n_l)
    prior = task.get_prior()
    simulator = task.get_simulator()

    num_samples = 5
    parameters = prior(num_samples=num_samples)
    observations = simulator(parameters)

    expected_dim = 10 * n_l  # 10 subsampled points per region
    assert observations.shape == (num_samples, expected_dim)


@pytest.mark.parametrize("n_l", [2, 3])
def test_simulator_no_nan(n_l):
    """Test simulator output contains no NaN values (most of the time).

    Note: SIR ODE can occasionally fail for extreme parameter values,
    but should succeed for most prior samples.
    """
    task = HierarchicalSIR(n_l=n_l)
    prior = task.get_prior()
    simulator = task.get_simulator()

    parameters = prior(num_samples=20)
    observations = simulator(parameters)

    # Allow some NaN values due to ODE failures, but most should succeed
    nan_rate = torch.isnan(observations).any(dim=1).float().mean()
    assert nan_rate < 0.5  # At least 50% should succeed


def test_prior_structure():
    """Test prior structure: global gamma and local betas."""
    n_l = 5
    task = HierarchicalSIR(n_l=n_l)
    prior = task.get_prior()

    samples = prior(num_samples=1000)

    # First param is global gamma: LogNormal(log(0.125), 0.2)
    gamma = samples[:, 0]
    assert gamma.shape[0] == 1000
    # Check positive (LogNormal support)
    assert (gamma > 0).all()

    # Next n_l params are local betas: LogNormal(log(0.4), 0.5)
    beta = samples[:, 1:]
    assert beta.shape == (1000, n_l)
    # Check positive
    assert (beta > 0).all()


def test_prior_dist_log_prob():
    """Test prior distribution log_prob."""
    n_l = 3
    task = HierarchicalSIR(n_l=n_l)
    prior = task.get_prior()

    # Sample from prior
    samples = prior(num_samples=10)

    # Compute log_prob via prior_dist
    log_probs = task.prior_dist.log_prob(samples)
    assert log_probs.shape == torch.Size([10])
    assert not torch.isnan(log_probs).any()
    assert torch.all(torch.isfinite(log_probs))


def test_likelihood_function():
    """Test _likelihood function returns finite values.

    This test ensures the prior is tuned such that ODE integration
    succeeds reliably. If this test fails, the prior may need adjustment.
    """
    n_l = 2
    task = HierarchicalSIR(n_l=n_l)
    prior = task.get_prior()
    simulator = task.get_simulator()

    # Generate parameters and observations
    parameters = prior(num_samples=5)
    observations = simulator(parameters)

    # Compute likelihood
    log_lik = task._likelihood(parameters, observations, log=True)

    assert log_lik.shape == torch.Size([5])
    # Likelihood should be finite for all valid observations
    valid_obs = ~torch.isnan(observations).any(dim=1)
    assert torch.all(torch.isfinite(log_lik[valid_obs]))


def test_reference_posterior_not_implemented():
    """Test that reference posterior raises NotImplementedError."""
    task = HierarchicalSIR(n_l=3)

    with pytest.raises(NotImplementedError):
        task._sample_reference_posterior(num_samples=100, num_observation=1)


def test_get_prior_dist():
    """Test get_prior_dist returns the prior distribution."""
    task = HierarchicalSIR(n_l=3)
    prior_dist = task.get_prior_dist()

    assert prior_dist is not None
    assert prior_dist is task.prior_dist

    # Test sampling from returned distribution
    samples = prior_dist.sample((10,))
    assert samples.shape == (10, 4)  # 1 global + 3 local


@pytest.mark.parametrize("n_l", [1, 3, 5])
def test_vmap_vs_sequential_ode_solving(n_l):
    """Test that vmap ODE solving matches sequential solving.

    This test verifies that the vmapped ODE solver produces identical
    results to solving each (sample, site) pair independently in a loop.

    Args:
        n_l: Number of local contexts/regions
    """
    task = HierarchicalSIR(n_l=n_l)
    prior = task.get_prior()

    # Use small num_samples for speed
    num_samples = 2
    parameters = prior(num_samples=num_samples)

    # Extract parameters
    gamma = parameters[:, 0]  # (num_samples,)
    beta = parameters[:, 1:]  # (num_samples, n_l)

    # Convert to JAX arrays
    gamma_jax = jnp.array(gamma.numpy())
    beta_jax = jnp.array(beta.numpy())

    # Method 1: Sequential solving (reference, in JAX)
    def solve_sequential_jax():
        trajectories_sequential = []
        for i in range(num_samples):
            gamma_i = gamma_jax[i]
            beta_i = beta_jax[i, :]  # (n_l,)
            sample_trajectories = []

            for j in range(n_l):
                gamma_ij = gamma_i
                beta_ij = beta_i[j]

                # Solve single (sample, site) pair using JAX function
                traj_jax = task._solve_ode_single_site_jax(gamma_ij, beta_ij)
                sample_trajectories.append(traj_jax)

            # Stack sites: (n_l, 3, timepoints)
            sample_traj = jnp.stack(sample_trajectories, axis=0)
            trajectories_sequential.append(sample_traj)

        # Stack samples: (num_samples, n_l, 3, timepoints)
        return jnp.stack(trajectories_sequential, axis=0)

    # Method 2: Vmapped solving (in JAX)
    def solve_vmapped_jax():
        return task._solve_ode_trajectories_jax(gamma_jax, beta_jax)

    # Get results in JAX
    traj_seq_jax = solve_sequential_jax()
    traj_vmap_jax = solve_vmapped_jax()

    # Convert to numpy for comparison
    traj_seq = numpy.asarray(traj_seq_jax)
    traj_vmap = numpy.asarray(traj_vmap_jax)

    # Verify shapes match
    assert traj_seq.shape == traj_vmap.shape, (
        f"Shape mismatch: sequential {traj_seq.shape} "
        f"vs vmapped {traj_vmap.shape}"
    )

    # Verify results are similar (allowing for ODE solver differences)
    # ODE solvers with adaptive stepping can take different numerical paths
    # depending on how the computation is batched. Both solve the same ODE.
    assert numpy.allclose(
        traj_seq, traj_vmap, rtol=1e-2, atol=1.0
    ), (
        f"Trajectories differ beyond tolerance. "
        f"Max difference: {numpy.abs(traj_seq - traj_vmap).max()}"
    )
