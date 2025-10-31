"""
Posterior distribution from hierarchical DeepSet.

Provides sample() and log_prob() interface compatible with
sbibm metrics and benchmarking.
"""

import torch


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

    def _encode_observation(self):
        """Encode test observation to get global/local contexts."""
        with torch.no_grad():
            # Create batch of size 1 with observation
            x_batch = self.observation.unsqueeze(0)  # (1, n_events, dim)
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
            from einops import rearrange

            x_enc = self.model.deep_set.enc(rearrange(x_batch, "b n d -> (b n) d"))
            x_enc = rearrange(
                x_enc,
                "(b n) d -> b n d",
                b=n_batch,
                n=n_set_max,
            )

            # Create mask for valid observations
            mask = torch.ones(n_batch, n_set_max, device=self.device)
            mask[:, self.observation.shape[0] :] = 0  # noqa: E203

            # Split features
            x, x_local = torch.chunk(x_enc, 2, -1)

            # Global aggregation
            x_global = x.sum(-2) / mask.sum(1)[:, None]
            x_global = torch.cat([x_global, mask.sum(1, keepdim=True)], -1)
            self.global_context = self.model.deep_set.dec(x_global)

            # Local features
            self.local_contexts = x_local[0, : self.observation.shape[0]]  # noqa: E203

    def sample(self, shape=()):
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

        with torch.no_grad():
            # Sample global parameters
            global_samples = self.model.deep_set.flow_global.sample(
                sample_shape=(num_samples,),
                context=self.global_context.expand(num_samples, -1),
            )

            # Sample local parameters (conditioned on global if applicable)
            num_events = self.observation.shape[0]
            local_samples = []

            for i in range(num_events):
                local_context = self.local_contexts[i : i + 1].expand(  # noqa: E203
                    num_samples, -1
                )
                if self.model.deep_set.condition_local_on_global:
                    local_context = torch.cat([local_context, global_samples], dim=-1)

                local_sample = self.model.deep_set.flow_local.sample(
                    sample_shape=(num_samples,),
                    context=local_context,
                )
                local_samples.append(local_sample)

            # Concatenate global and all local samples
            all_local = torch.cat(local_samples, dim=-1)
            samples = torch.cat([global_samples, all_local], dim=-1)

            # Reshape to requested shape
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
            # Split into global and local
            dim_global = self.model.deep_set.flow_global.distribution.shape[0]
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
                local_context = self.local_contexts[i : i + 1].expand(  # noqa: E203
                    batch_size, -1
                )
                if self.model.deep_set.condition_local_on_global:
                    local_context = torch.cat([local_context, global_params], dim=-1)

                local_params = local_params_flat[
                    :, i * dim_local : (i + 1) * dim_local  # noqa: E203
                ]
                log_prob_local = self.model.deep_set.flow_local.log_prob(
                    local_params, local_context
                )
                log_prob_local_total += log_prob_local

            # Total log prob
            log_prob = log_prob_global + log_prob_local_total

            # Reshape to original shape
            return log_prob.reshape(original_shape)
