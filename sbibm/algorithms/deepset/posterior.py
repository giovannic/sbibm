"""
Posterior distribution from hierarchical DeepSet.

Provides sample() and log_prob() interface compatible with
sbibm metrics and benchmarking.
"""

import torch
from .utils import reshape_observations
from einops import rearrange


class HierarchicalPosterior:
    """
    Posterior distribution from trained hierarchical DeepSet.

    Encodes observations and provides sampling and density
    evaluation for posterior metrics.
    """

    def __init__(self, model, observation, task, device="cpu"):
        """
        Initialize posterior from trained model.

        Args:
            model: Trained HierarchicalDeepSetInference instance
            observation: Test observation ensemble,
                        shape (num_events, dim_per_event)
            task: Task instance with hierarchical structure
            device: torch device ("cpu" or "cuda")
        """
        self.model = model.to(device)
        self.model.eval()
        self.task = task
        self.device = device

        # Store observation and compute context
        if observation.dim() == 1:
            observation = observation.unsqueeze(0)
        self.observation = observation.to(device)

        # Encode observation to get contexts for posterior
        self._encode_observation()

    def _get_context(self, observations):
        with torch.no_grad():
            # Create batch of size 1 with observation
            x_batch = observations # (n_obs, n_events, dim)
            n_batch = x_batch.shape[0]

            # Pad observation to match n_set_max if needed
            n_set_max = self.model.deep_set.n_set_max
            if x_batch.shape[1] < n_set_max:
                padding = torch.zeros(
                    n_batch,
                    n_set_max - x_batch.shape[1],
                    x_batch.shape[2],
                    device=self.device,
                )
                x_batch = torch.cat([x_batch, padding], dim=1)

            # Forward pass through encoder
            x_enc = self.model.deep_set.enc(rearrange(x_batch, "b n d -> (b n) d"))
            x_enc = rearrange(
                x_enc,
                "(b n) d -> b n d",
                b=n_batch,
                n=n_set_max,
            )

            # Create mask for valid observations
            mask = torch.ones(n_batch, n_set_max, device=self.device)
            mask[:, observations.shape[1] :] = 0  # noqa: E203

            # Split encoder output in half: global vs local features
            # enc_dim = encoder output dim, split_dim = enc_dim // 2
            x, x_local = torch.chunk(x_enc, 2, -1)  # (b, n, split_dim) each

            # Global aggregation
            x_global = x.sum(-2) / mask.sum(1)[:, None]  # (b, split_dim)
            x_global = torch.cat([x_global, mask.sum(1, keepdim=True)], -1)

            # Local features
            obs_len = observations.shape[1]
            local_contexts = x_local[:, :obs_len]  # noqa: E203

            # global_context: (b, dec_out), local_contexts: (b, obs_len, split_dim)
            return self.model.deep_set.dec(x_global), local_contexts

    def _encode_observation(self):
        """Encode test observation to get global/local contexts."""
        (
            self.global_context,
            self.local_contexts
        ) = self._get_context(self.observation.unsqueeze(0))

    def _sample(self, num_samples, observations):
        global_context, local_contexts = self._get_context(observations)
        # Need to batch the contexts
        # then avoid the reshaping
        # but not break single sample
        with torch.no_grad():
            # Sample global parameters
            # Note: flow.sample(num_samples, context) returns shape
            # (context_batch, num_samples, features)
            global_samples = self.model.deep_set.flow_global.sample(
                num_samples,
                context=global_context,
            )

            # Sample local parameters (conditioned on global if applicable)
            num_events = observations.shape[1]
            n_batch = observations.shape[0]
            local_samples = []

            if self.model.deep_set.condition_local_on_global:
                # When conditioning local on global, need to sample each
                # local parameter with the actual global sample values
                for i in range(num_events):
                    local_ctx = local_contexts[:, i : i + 1]  # noqa: E203
                    # Concatenate with global samples
                    local_ctx_expanded = torch.cat(
                        [local_ctx.expand(n_batch, num_samples, -1), global_samples],
                        dim=-1,
                    )
                    # Now sample 1 sample per global sample
                    local_ctx_expanded = rearrange(local_ctx_expanded, "b n d -> (b n) d")
                    local_sample = self.model.deep_set.flow_local.sample(
                        1,
                        context=local_ctx_expanded,
                    )
                    local_sample = rearrange(
                        local_sample,
                        "(b n) i d -> b (n i) d",
                        b=n_batch,
                        n=num_samples,
                        i=1,
                    )
                    local_samples.append(local_sample)
                    # Shape is (batch, num_samples, 1, dim) -> squeeze to
                    # (batch, num_samples, dim)
                    # local_samples.append(local_sample.squeeze(2))
            else:
                # Sample all local parameters independently
                for i in range(num_events):
                    local_ctx = local_contexts[:, i : i + 1]  # noqa: E203
                    local_sample = self.model.deep_set.flow_local.sample(
                        num_samples,
                        context=local_ctx,
                    )
                    # Reshape from (1, num_samples, dim) to (num_samples, dim)
                    local_samples.append(local_sample.squeeze(0))

            # Concatenate global and all local samples
            all_local = torch.cat(local_samples, dim=-1)
            samples = torch.cat([global_samples, all_local], dim=-1)

            return samples.squeeze(1)


    def sample(self, shape=(), x=None):
        """
        Sample from posterior.

        Args:
            shape: Shape of sample batch, e.g. (1000,)

        Returns:
            samples: shape (*shape, dim_parameters)
        """
        # Handle shape argument
        if isinstance(shape, int):
            shape = (shape,)

        num_samples = 1
        for dim in shape:
            num_samples *= dim

        if x is None:
            samples = self._sample(num_samples, self.observation.unsqueeze(0))
        else:
            x = reshape_observations(self.task, x)
            samples = self._sample(1, x)
            return samples

        return samples.reshape(*shape, -1)

    def log_prob(self, theta):
        """
        Compute log probability under posterior.

        Args:
            theta: Parameters, shape (..., dim_parameters)

        Returns:
            log_prob: shape (...)
        """
        # Flatten batch dimensions
        original_shape = theta.shape[:-1]
        theta_flat = theta.reshape(-1, theta.shape[-1])
        batch_size = theta_flat.shape[0]

        with torch.no_grad():
            # Get dimensions from model
            flow_global = self.model.deep_set.flow_global
            dim_global = flow_global._distribution._shape[0]
            global_params = theta_flat[:, :dim_global]
            local_params_flat = theta_flat[:, dim_global:]

            # Evaluate global flow
            global_context = self.global_context.expand(batch_size, -1)
            log_prob_global = self.model.deep_set.flow_global.log_prob(
                global_params, global_context
            )

            # Evaluate local flows
            num_events = self.observation.shape[0]
            dim_local = local_params_flat.shape[1] // num_events
            log_prob_local_total = torch.zeros(batch_size, device=self.device)

            for i in range(num_events):
                local_context = self.local_contexts[0, i : i + 1]  # noqa: E203
                local_context_expanded = local_context.expand(batch_size, -1)

                if self.model.deep_set.condition_local_on_global:
                    local_context_expanded = torch.cat(
                        [local_context_expanded, global_params], dim=-1
                    )

                local_params = local_params_flat[
                    :, i * dim_local : (i + 1) * dim_local  # noqa: E203
                ]
                log_prob_local = self.model.deep_set.flow_local.log_prob(
                    local_params, local_context_expanded
                )
                log_prob_local_total += log_prob_local

            # Total log prob
            log_prob = log_prob_global + log_prob_local_total

            # Reshape to original shape
            return log_prob.reshape(original_shape)
