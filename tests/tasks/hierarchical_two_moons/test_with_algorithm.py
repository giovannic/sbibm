"""Test hierarchical_two_moons with SNPE algorithm and reference-free metrics.

This module tests the end-to-end integration of hierarchical_two_moons with:
- SNPE algorithm from sbi
- Reference-free metrics (reverse_kl and lc2st)
"""

import logging

import pytest
import torch

from sbibm.algorithms.sbi.snpe import run as run_snpe
from sbibm.algorithms.sbi.utils import (
    wrap_posterior,
    wrap_prior_dist,
    wrap_simulator_fn,
)
from sbibm.algorithms.tfmpe.bottom_up import run as run_bottom_up
from sbibm.metrics.lc2st import lc2st
from sbibm.metrics.reverse_kl import reverse_kl
from sbibm.tasks.hierarchical_two_moons.task import HierarchicalTwoMoons

log = logging.getLogger(__name__)


class MockPosterior:
    """Mock posterior that operates in unbounded (unconstrained) space.

    This simulates what a neural density estimator returns before wrapping.
    """

    def __init__(self, mean, std):
        # Normal distribution - unbounded support
        self.dist = torch.distributions.Normal(mean, std)

    def sample(self, shape):
        return self.dist.sample(shape)

    def log_prob(self, samples):
        return self.dist.log_prob(samples).sum(dim=-1)


@pytest.mark.parametrize("automatic_transforms_enabled", [True, False])
def test_both_metrics_on_same_posterior(automatic_transforms_enabled):
    """Test both metrics on the same trained posterior.

    This test verifies that both reverse_kl and lc2st can be computed
    on the same posterior object, demonstrating the complete reference-free
    evaluation workflow.

    Args:
        automatic_transforms_enabled: Whether to use automatic transforms
    """
    n_l = 5
    task = HierarchicalTwoMoons(n_l=n_l)
    n_samples = 100

    # Run SNPE with minimal budget for testing
    samples, num_sims, log_prob_true, posterior = run_snpe(
        task=task,
        num_samples=n_samples,
        num_simulations=100,
        num_observation=1,
        num_rounds=1,
        neural_net="nsf",
        hidden_features=50,
        max_num_epochs=10,
        automatic_transforms_enabled=automatic_transforms_enabled,
    )

    # Verify samples shape
    expected_dim = 4 + 2 * n_l
    assert samples.shape == (n_samples, expected_dim), (
        f"Expected samples shape ({n_samples}, {expected_dim}), " f"got {samples.shape}"
    )

    # Verify no NaN values in samples
    assert not torch.isnan(samples).any(), "Samples contain NaN values"

    # Verify posterior object is returned
    assert posterior is not None, "Posterior object should be returned"

    # Verify posterior has required methods
    assert hasattr(posterior, "sample"), "Posterior should have sample method"
    assert hasattr(posterior, "log_prob"), "Posterior should have log_prob method"

    # Test that we can sample from posterior
    new_samples = posterior.sample((50,))
    assert new_samples.shape == (50, expected_dim)
    assert not torch.isnan(new_samples).any()

    # Test that we can compute log_prob
    log_probs = posterior.log_prob(samples)
    assert not torch.isnan(log_probs).any()

    # Verify log_prob_true is computed when num_observation is provided
    assert log_prob_true is not None
    assert torch.isfinite(log_prob_true)

    # Compute reverse KL
    rkl = reverse_kl(
        posterior=posterior,
        task=task,
        num_observation=1,
        num_samples=1000,
    )

    # Verify reverse KL metric is finite
    assert torch.isfinite(rkl), f"Reverse KL should be finite, got {rkl}"
    assert rkl.numel() == 1, "Reverse KL should be a scalar"

    # Compute LC2ST
    lc2st_result = lc2st(
        posterior=posterior,
        task=task,
        num_observation=1,
        num_calibration_samples=10,
        num_posterior_samples=10,
        num_trials=10,
    )

    # Verify LC2ST result is a dictionary
    assert isinstance(lc2st_result, dict), "LC2ST should return a dictionary"
    assert (
        "test_statistic" in lc2st_result
    ), "Result should contain 'test_statistic' key"

    # Verify LC2ST values are finite
    for key, value in lc2st_result.items():
        if isinstance(value, (torch.Tensor, float)):
            if isinstance(value, torch.Tensor):
                assert torch.isfinite(
                    value
                ).all(), f"LC2ST result[{key}] contains non-finite values"
            else:
                assert not (
                    value != value or abs(value) == float("inf")
                ), f"LC2ST result[{key}] is not finite"

    log.info(
        f"Complete reference-free evaluation "
        f"(transforms={'enabled' if automatic_transforms_enabled else 'disabled'}):"
        f"\n  Reverse KL: {rkl.item():.3f}"
        f"\n  LC2ST test_statistic: {lc2st_result.get('test_statistic', 'N/A')}"
        f"\n  Log prob (true params): {log_prob_true.item():.3f}"
    )


def test_snpe_samples_in_valid_range():
    """Test that SNPE samples are within the valid parameter ranges.

    Verifies that samples from the trained posterior respect the parameter
    constraints:
    - Global loc parameters: [-1, 1]
    - Global scale parameters: [0, inf)
    - Local parameters: [-1, 1]
    """
    automatic_transforms_enabled = True
    n_l = 5
    task = HierarchicalTwoMoons(n_l=n_l)

    # Run SNPE with minimal budget for testing
    samples, num_sims, log_prob_true, posterior = run_snpe(
        task=task,
        num_samples=100,
        num_simulations=1000,
        num_observation=1,
        num_rounds=1,
        neural_net="nsf",
        hidden_features=50,
        max_num_epochs=10,
        automatic_transforms_enabled=automatic_transforms_enabled,
    )

    # Check global loc parameters (dims 0-1, should be in [-1, 1])
    global_loc = samples[:, :2]
    assert (
        global_loc >= -1.0
    ).all(), f"Global loc below -1: min={global_loc.min().item()}"
    assert (
        global_loc <= 1.0
    ).all(), f"Global loc above 1: max={global_loc.max().item()}"

    # Check global scale parameters (dims 2-3, should be positive)
    global_scale = samples[:, 2:4]
    assert (
        global_scale >= 0.0
    ).all(), f"Global scale negative: min={global_scale.min().item()}"

    # Check local parameters (dims 4+, should be in [-1, 1])
    local_params = samples[:, 4:]
    assert (
        local_params >= -1.0
    ).all(), f"Local params below -1: min={local_params.min().item()}"
    assert (
        local_params <= 1.0
    ).all(), f"Local params above 1: max={local_params.max().item()}"

    log.info(
        f"Sample ranges (transforms={'enabled' if automatic_transforms_enabled else 'disabled'}):"
        f"\n  Global loc: [{global_loc.min().item():.3f}, {global_loc.max().item():.3f}]"
        f"\n  Global scale: [{global_scale.min().item():.3f}, {global_scale.max().item():.3f}]"
        f"\n  Local params: [{local_params.min().item():.3f}, {local_params.max().item():.3f}]"
    )


def test_wrapped_prior_domain_codomain():
    """Test that wrapped prior produces unconstrained samples.

    The wrapped prior should output samples in unconstrained space (R^n).
    When transformed back via transform.inv(), they should satisfy the
    parameter constraints.
    """
    n_l = 5
    task = HierarchicalTwoMoons(n_l=n_l)

    # Get unwrapped and wrapped priors
    prior_unwrapped = task.get_prior_dist()
    transforms = task._get_transforms(automatic_transforms_enabled=True)["parameters"]
    prior_wrapped = wrap_prior_dist(prior_unwrapped, transforms)

    # Sample from wrapped prior (should be unconstrained)
    num_samples = 1000
    samples_unconstrained = prior_wrapped.sample((num_samples,))

    assert samples_unconstrained.shape == (
        num_samples,
        task.dim_parameters,
    ), f"Expected shape ({num_samples}, {task.dim_parameters})"

    # Wrapped prior samples should be unbounded (can have any value)
    # Check that some samples are outside constrained bounds
    # (this would fail if prior wasn't actually transformed)
    assert (
        samples_unconstrained.abs().max() > 1.0
    ), "Unconstrained samples should extend beyond constrained bounds"

    # Transform back to constrained space
    samples_constrained = transforms.inv(samples_unconstrained)

    # Check constrained samples satisfy bounds
    # Global loc (dims 0-1): [-1, 1]
    global_loc = samples_constrained[:, :2]
    assert (global_loc >= -1.0).all(), (
        f"Global loc should be >= -1, " f"got min={global_loc.min().item()}"
    )
    assert (global_loc <= 1.0).all(), (
        f"Global loc should be <= 1, " f"got max={global_loc.max().item()}"
    )

    # Global scale (dims 2-3): > 0
    global_scale = samples_constrained[:, 2:4]
    assert (global_scale > 0.0).all(), (
        f"Global scale should be > 0, " f"got min={global_scale.min().item()}"
    )

    # Local params (dims 4+): [-1, 1]
    local_params = samples_constrained[:, 4:]
    assert (local_params >= -1.0).all(), (
        f"Local params should be >= -1, " f"got min={local_params.min().item()}"
    )
    assert (local_params <= 1.0).all(), (
        f"Local params should be <= 1, " f"got max={local_params.max().item()}"
    )


def test_wrapped_prior_domain_codomain_with_gaussian():
    """Test that wrapped prior produces unconstrained samples.

    The wrapped prior should output samples in unconstrained space (R^n).
    When transformed back via transform.inv(), they should satisfy the
    parameter constraints.
    """
    n_l = 5
    task = HierarchicalTwoMoons(n_l=n_l)

    # Get unwrapped and wrapped priors
    transforms = task._get_transforms(automatic_transforms_enabled=True)["parameters"]

    # Sample from wrapped prior (should be unconstrained)
    num_samples = 1000
    samples_unconstrained = torch.distributions.Normal(
        loc=torch.zeros(num_samples, task.dim_parameters),
        scale=torch.ones(num_samples, task.dim_parameters),
    ).sample()

    assert samples_unconstrained.shape == (
        num_samples,
        task.dim_parameters,
    ), f"Expected shape ({num_samples}, {task.dim_parameters})"

    # Wrapped prior samples should be unbounded (can have any value)
    # Check that some samples are outside constrained bounds
    # (this would fail if prior wasn't actually transformed)
    assert (
        samples_unconstrained.abs().max() > 1.0
    ), "Unconstrained samples should extend beyond constrained bounds"

    # Transform back to constrained space
    samples_constrained = transforms.inv(samples_unconstrained)

    # Check constrained samples satisfy bounds
    # Global loc (dims 0-1): [-1, 1]
    global_loc = samples_constrained[:, :2]
    assert (global_loc >= -1.0).all(), (
        f"Global loc should be >= -1, " f"got min={global_loc.min().item()}"
    )
    assert (global_loc <= 1.0).all(), (
        f"Global loc should be <= 1, " f"got max={global_loc.max().item()}"
    )

    # Global scale (dims 2-3): > 0
    global_scale = samples_constrained[:, 2:4]
    assert (global_scale > 0.0).all(), (
        f"Global scale should be > 0, " f"got min={global_scale.min().item()}"
    )

    # Local params (dims 4+): [-1, 1]
    local_params = samples_constrained[:, 4:]
    assert (local_params >= -1.0).all(), (
        f"Local params should be >= -1, " f"got min={local_params.min().item()}"
    )
    assert (local_params <= 1.0).all(), (
        f"Local params should be <= 1, " f"got max={local_params.max().item()}"
    )


def test_wrapped_prior_log_prob():
    """Test that wrapped prior computes log_prob correctly.

    The wrapped prior should compute log probabilities in unconstrained space
    that match the unwrapped prior's log probabilities in constrained space
    (accounting for the transformation Jacobian automatically).
    """
    n_l = 5
    task = HierarchicalTwoMoons(n_l=n_l)

    # Get unwrapped and wrapped priors
    prior_unwrapped = task.get_prior_dist()
    transforms = task._get_transforms(automatic_transforms_enabled=True)["parameters"]
    prior_wrapped = wrap_prior_dist(prior_unwrapped, transforms)

    # Sample from unwrapped prior (constrained space)
    num_samples = 1000
    samples_constrained = prior_unwrapped.sample((num_samples,))

    assert samples_constrained.shape == (
        num_samples,
        task.dim_parameters,
    ), f"Expected shape ({num_samples}, {task.dim_parameters})"

    # Transform to unconstrained space
    samples_unconstrained = transforms(samples_constrained)

    # Compute log probs in both spaces
    log_prob_unwrapped = prior_unwrapped.log_prob(samples_constrained)
    log_prob_wrapped = prior_wrapped.log_prob(samples_unconstrained)

    # Compute Jacobian for change of variables
    log_abs_det_jac = transforms.log_abs_det_jacobian(
        samples_constrained, samples_unconstrained
    )

    # Check shapes
    assert log_prob_unwrapped.shape == (num_samples,), (
        f"Expected log_prob shape ({num_samples},), " f"got {log_prob_unwrapped.shape}"
    )
    assert log_prob_wrapped.shape == (num_samples,), (
        f"Expected log_prob shape ({num_samples},), " f"got {log_prob_wrapped.shape}"
    )
    assert log_abs_det_jac.shape == (num_samples,), (
        f"Expected Jacobian shape ({num_samples},), " f"got {log_abs_det_jac.shape}"
    )

    # Check all values are finite
    assert torch.isfinite(
        log_prob_unwrapped
    ).all(), "Unwrapped log_prob contains non-finite values"
    assert torch.isfinite(
        log_prob_wrapped
    ).all(), "Wrapped log_prob contains non-finite values"
    assert torch.isfinite(log_abs_det_jac).all(), "Jacobian contains non-finite values"

    # Check 1: Change-of-variables formula
    # log p_Y(y) = log p_X(x) - log|det J|
    # where y = transform(x) (constrained -> unconstrained)
    expected_log_prob = log_prob_unwrapped - log_abs_det_jac
    assert torch.allclose(log_prob_wrapped, expected_log_prob, atol=1e-2), (
        f"Change-of-variables formula failed: "
        f"max diff = {(log_prob_wrapped - expected_log_prob).abs().max()}"
    )

    # Check 2: Verify wrapped prior can compute log_prob and it's close
    # to unwrapped (accounting for Jacobian)
    # This verifies: wrapped.log_prob(y) + log|det J| ≈ unwrapped.log_prob(x)
    recovered_log_prob = log_prob_wrapped + log_abs_det_jac
    assert torch.allclose(recovered_log_prob, log_prob_unwrapped, atol=1e-5), (
        f"Recovered log prob doesn't match unwrapped: "
        f"max diff = {(recovered_log_prob - log_prob_unwrapped).abs().max()}"
    )


def test_wrapped_simulator_domain_codomain():
    """Test that wrapped simulator accepts unconstrained input.

    The wrapped simulator should accept parameters in unconstrained space
    and produce valid observations.
    """
    n_l = 5
    task = HierarchicalTwoMoons(n_l=n_l)

    # Get wrapped prior and simulator
    prior_unwrapped = task.get_prior_dist()
    transforms = task._get_transforms(automatic_transforms_enabled=True)["parameters"]
    prior_wrapped = wrap_prior_dist(prior_unwrapped, transforms)
    simulator_unwrapped = task.get_simulator()
    simulator_wrapped = wrap_simulator_fn(simulator_unwrapped, transforms)

    # Sample from wrapped prior (unconstrained)
    num_samples = 1000
    samples_unconstrained = prior_wrapped.sample((num_samples,))

    # Pass unconstrained samples to wrapped simulator
    observations = simulator_wrapped(samples_unconstrained)

    # Check observations shape
    assert observations.shape == (num_samples, task.dim_data), (
        f"Expected observations shape ({num_samples}, {task.dim_data}), "
        f"got {observations.shape}"
    )

    # Check no NaNs or Infs
    assert not torch.isnan(observations).any(), "Observations contain NaN values"
    assert torch.isfinite(observations).all(), "Observations contain non-finite values"

    # Verify simulator wrapper internally transforms to constrained space
    # by checking that results are similar to unwrapped simulator with
    # constrained samples
    samples_constrained = transforms.inv(samples_unconstrained[:10])
    observations_unwrapped = simulator_unwrapped(samples_constrained)

    # These should produce valid observations (not necessarily identical
    # due to stochasticity in simulator)
    assert observations_unwrapped.shape == (10, task.dim_data)
    assert torch.isfinite(observations_unwrapped).all()


def test_wrapped_posterior_domain_codomain():
    """Test that wrapped posterior operates in constrained space.

    The wrapped posterior should:
    - sample() returns samples in constrained space
    - log_prob() accepts samples in constrained space
    """
    n_l = 5
    task = HierarchicalTwoMoons(n_l=n_l)

    # Get transforms
    transforms = task._get_transforms(automatic_transforms_enabled=True)["parameters"]

    # Create mock posterior in unconstrained space
    # (simulating what density estimator returns)
    true_params = task.get_true_parameters(num_observation=1)
    true_params_unconstrained = transforms(true_params.squeeze())

    mock_posterior_unconstrained = MockPosterior(
        mean=true_params_unconstrained,
        std=torch.ones(task.dim_parameters) * 2.0,
    )

    # Wrap posterior (should now operate in constrained space)
    posterior_wrapped = wrap_posterior(mock_posterior_unconstrained, transforms)

    # Test 1: sample() returns constrained samples
    num_samples = 1000
    samples = posterior_wrapped.sample((num_samples,))

    assert samples.shape == (num_samples, task.dim_parameters)

    # Check samples satisfy constraints
    # Global loc (dims 0-1): [-1, 1]
    global_loc = samples[:, :2]
    assert (global_loc >= -1.0).all(), (
        f"Global loc should be >= -1, " f"got min={global_loc.min().item()}"
    )
    assert (global_loc <= 1.0).all(), (
        f"Global loc should be <= 1, " f"got max={global_loc.max().item()}"
    )

    # Global scale (dims 2-3): > 0
    global_scale = samples[:, 2:4]
    assert (global_scale > 0.0).all(), (
        f"Global scale should be > 0, " f"got min={global_scale.min().item()}"
    )

    # Local params (dims 4+): [-1, 1]
    local_params = samples[:, 4:]
    assert (local_params >= -1.0).all(), (
        f"Local params should be >= -1, " f"got min={local_params.min().item()}"
    )
    assert (local_params <= 1.0).all(), (
        f"Local params should be <= 1, " f"got max={local_params.max().item()}"
    )

    # Test 2: log_prob() accepts constrained samples
    log_probs = posterior_wrapped.log_prob(samples[:10])

    assert log_probs.shape == (
        10,
    ), f"Expected log_probs shape (10,), got {log_probs.shape}"
    assert torch.isfinite(log_probs).all(), "log_probs should be finite"


def test_snpe_two_rounds():
    """Test that SNPE can run with two rounds of training.

    Verifies that multi-round SNPE training works correctly with
    the hierarchical two moons task and produces valid posteriors.
    """
    n_l = 5
    task = HierarchicalTwoMoons(n_l=n_l)
    n_samples = 100

    # Run SNPE with two rounds
    samples, num_sims, log_prob_true, posterior = run_snpe(
        task=task,
        num_samples=n_samples,
        num_simulations=500,
        num_observation=1,
        num_rounds=2,
        neural_net="nsf",
        hidden_features=50,
        max_num_epochs=10,
        automatic_transforms_enabled=True,
    )

    # Verify samples shape
    expected_dim = 4 + 2 * n_l
    assert samples.shape == (n_samples, expected_dim), (
        f"Expected samples shape ({n_samples}, {expected_dim}), " f"got {samples.shape}"
    )

    # Verify no NaN values in samples
    assert not torch.isnan(samples).any(), "Samples contain NaN values"

    # Verify posterior object is returned
    assert posterior is not None, "Posterior object should be returned"

    # Verify posterior has required methods
    assert hasattr(posterior, "sample"), "Posterior should have sample method"
    assert hasattr(posterior, "log_prob"), "Posterior should have log_prob method"

    # Test that we can sample from posterior
    new_samples = posterior.sample((50,))
    assert new_samples.shape == (50, expected_dim)
    assert not torch.isnan(new_samples).any()

    # Test that we can compute log_prob
    log_probs = posterior.log_prob(samples[:10])
    assert log_probs.shape == (10,)
    assert not torch.isnan(log_probs).any()

    # Verify samples respect parameter constraints
    # Global loc (dims 0-1): [-1, 1]
    global_loc = samples[:, :2]
    assert (
        global_loc >= -1.0
    ).all(), f"Global loc below -1: min={global_loc.min().item()}"
    assert (
        global_loc <= 1.0
    ).all(), f"Global loc above 1: max={global_loc.max().item()}"

    # Global scale (dims 2-3): > 0
    global_scale = samples[:, 2:4]
    assert (
        global_scale >= 0.0
    ).all(), f"Global scale negative: min={global_scale.min().item()}"

    # Local params (dims 4+): [-1, 1]
    local_params = samples[:, 4:]
    assert (
        local_params >= -1.0
    ).all(), f"Local params below -1: min={local_params.min().item()}"
    assert (
        local_params <= 1.0
    ).all(), f"Local params above 1: max={local_params.max().item()}"

    log.info(
        f"Two-round SNPE completed successfully:"
        f"\n  Num simulations: {num_sims}"
        f"\n  Sample shape: {samples.shape}"
        f"\n  Global loc range: [{global_loc.min().item():.3f}, "
        f"{global_loc.max().item():.3f}]"
        f"\n  Global scale range: [{global_scale.min().item():.3f}, "
        f"{global_scale.max().item():.3f}]"
        f"\n  Local params range: [{local_params.min().item():.3f}, "
        f"{local_params.max().item():.3f}]"
    )


def test_bottom_up_hierarchical_two_moons(
    task_name="hierarchical_two_moons",
    num_observation=1,
    num_samples=100,
    num_simulations=100,
):
    """Integration test for TFMPE bottom-up on hierarchical two moons.

    Validates that the bottom_up algorithm wrapper:
    - Loads the task and observation
    - Runs TFMPE training
    - Returns samples with correct shape
    - Returns execution time and metadata
    - Samples fall within prior bounds
    """
    task = HierarchicalTwoMoons(n_l=5)

    # Run the algorithm
    samples, execution_time, metadata = run_bottom_up(
        task=task,
        num_observation=num_observation,
        num_samples=num_samples,
        num_simulations=num_simulations,
        automatic_transforms_enabled=True,
    )

    # Validate output shape
    assert isinstance(samples, torch.Tensor)
    assert samples.shape == (num_samples, task.dim_parameters)

    # Validate execution time was recorded
    assert isinstance(execution_time, float)
    assert execution_time > 0.0

    # Validate metadata
    assert isinstance(metadata, dict)
    assert "losses" in metadata
    assert "n_samples_per_round" in metadata

    # Validate samples are not NaN or Inf
    assert not torch.isnan(samples).any()
    assert not torch.isinf(samples).any()

    # Validate scales are positive
    scales = samples[:, 2:4]
    assert scales.min() >= 0, f"scales must be positive but found {scales.min()}"

    # Validate local samples fall strictly within prior bounds [-1, 1]
    local_samples = samples[:, 4:]
    assert local_samples.min() >= -1.0, f"local sample min {local_samples.min()} < -1.0"
    assert local_samples.max() <= 1.0, f"local sample max {local_samples.max()} > 1.0"

    log.info(
        f"TFMPE bottom-up completed on hierarchical_two_moons:"
        f"\n  Num simulations: {num_simulations}"
        f"\n  Sample shape: {samples.shape}"
        f"\n  Scales range: [{scales.min().item():.3f}, "
        f"{scales.max().item():.3f}]"
        f"\n  Local params range: [{local_samples.min().item():.3f}, "
        f"{local_samples.max().item():.3f}]"
    )
