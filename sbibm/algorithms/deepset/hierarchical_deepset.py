"""
Hierarchical DeepSet model and training module.

Source: https://github.com/smsharma/hierarchical-inference/blob/main/notebooks/05_lensing.ipynb
Extracted from lensing notebook in hierarchical-inference repository.
"""

import pytorch_lightning as pl
import torch
import torch.nn as nn
from einops import rearrange, repeat

from .flows import build_maf
from .resnet import ResNetEstimator
from .utils import build_mlp


class HierarchicalDeepSet(nn.Module):
    """
    Backbone to the hierarchical deep set model, using a ResNet embedder and MAF flows for
    local and global parameter posterior density estimators.
    """

    def __init__(
        self, dim_hidden=128, condition_local_on_global=True, n_set_max=None
    ):
        super(HierarchicalDeepSet, self).__init__()

        if n_set_max is None:
            raise ValueError("n_set_max must be specified")

        self.n_set_max = n_set_max

        inference_net_kwargs = {"cfg": 50}
        self.enc = ResNetEstimator(n_out=dim_hidden, **inference_net_kwargs)
        self.dec = build_mlp(
            input_dim=int(dim_hidden / 2) + 1,
            hidden_dim=int(2 * dim_hidden),
            output_dim=int(dim_hidden / 2),
            layers=4,
        )

        # Condition local flow on global params if local loss is turned on
        extra_context = 2 if condition_local_on_global else 0
        self.condition_local_on_global = condition_local_on_global

        self.flow_local = build_maf(
            dim=4,
            num_transforms=6,
            context_features=int(dim_hidden / 2) + extra_context,
            hidden_features=int(2 * dim_hidden),
        )
        self.flow_global = build_maf(
            dim=2,
            num_transforms=6,
            context_features=int(dim_hidden / 2),
            hidden_features=int(2 * dim_hidden),
        )

    def forward(self, x, y_local, y_global):
        n_batch = x.shape[0]

        lens = torch.randint(
            low=1,
            high=self.n_set_max + 1,
            size=(n_batch,),
            dtype=torch.float,
        )
        mask = (
            torch.arange(self.n_set_max).expand(len(lens), self.n_set_max)
            < torch.Tensor(lens)[:, None]
        ).to(x.device)

        x = rearrange(
            x, "batch n_set h w -> (batch n_set)  h w", n_set=self.n_set_max
        )
        x = self.enc(x)

        x = rearrange(
            x, "(batch n_set) n_out -> batch n_set n_out", n_set=self.n_set_max
        )

        idx_setperm = torch.randperm(self.n_set_max)  # Permutation indices
        x = (
            x[:, idx_setperm, :] * mask[:, :, None]
        )  # Permute set elements and mask
        y_local = y_local[:, idx_setperm, :]

        x, x_cond_local = torch.chunk(x, 2, -1)

        x = x.sum(-2) / mask.sum(1)[:, None]

        x = torch.cat(
            [x, lens[:, None].to(x.device)], -1
        )  # Add cardinality for rho network
        x_cond_global = self.dec(x)

        x_cond_local = rearrange(
            x_cond_local,
            "batch n_set n_out -> (batch n_set) n_out",
            n_set=self.n_set_max,
        )

        if self.condition_local_on_global:
            y_global_repeat = repeat(
                y_global,
                "batch glob -> (batch n_set) glob",
                n_set=self.n_set_max,
            )
            x_cond_local = torch.cat([x_cond_local, y_global_repeat], -1)

        y_local = rearrange(
            y_local,
            "batch n_set n_param -> (batch n_set) n_param",
            n_set=self.n_set_max,
        )

        log_prob_local = self.flow_local.log_prob(y_local, x_cond_local)
        log_prob_local = rearrange(
            log_prob_local,
            "(batch n_set) -> batch n_set",
            n_set=self.n_set_max,
        )
        log_prob_local = (log_prob_local * mask).sum(-1)

        log_prob_global = self.flow_global.log_prob(y_global, x_cond_global)

        return log_prob_local, log_prob_global


class HierarchicalDeepSetInference(pl.LightningModule):
    """
    Hierarchical deep set lightning module for training and inference.
    """

    def __init__(
        self,
        optimizer=torch.optim.AdamW,
        optimizer_kwargs={"weight_decay": 5e-5},
        lr=3e-4,
        max_epochs=50,
        scheduler=torch.optim.lr_scheduler.CosineAnnealingLR,
        local_loss=True,
        global_loss=True,
        n_set_max=None,
    ):
        super().__init__()

        if n_set_max is None:
            raise ValueError("n_set_max must be specified")

        self.optimizer = optimizer
        self.optimizer_kwargs = optimizer_kwargs
        self.scheduler = scheduler
        self.scheduler_kwargs = {"T_max": max_epochs}
        self.lr = lr

        self.local_loss = local_loss
        self.global_loss = global_loss

        # Condition local flow on global params only if both are turned on
        condition_local_on_global = (
            True if (local_loss and global_loss) else False
        )

        self.deep_set = HierarchicalDeepSet(
            condition_local_on_global=condition_local_on_global,
            n_set_max=n_set_max,
        )

    def forward(self, x, y_local, y_global):
        log_prob = self.deep_set(x, y_local, y_global)
        return log_prob

    def configure_optimizers(self):
        optimizer = self.optimizer(
            self.parameters(), lr=self.lr, **self.optimizer_kwargs
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": self.scheduler(
                    optimizer, **self.scheduler_kwargs
                ),
                "interval": "epoch",
                "monitor": "val_loss",
                "frequency": 1,
            },
        }

    def training_step(self, batch, batch_idx):
        x, y_local, y_global = batch
        log_prob_local, log_prob_global = self(x, y_local, y_global)
        log_prob = torch.zeros_like(log_prob_local).to(log_prob_local.device)
        if self.local_loss:
            log_prob += log_prob_local
        if self.global_loss:
            log_prob += log_prob_global
        loss = -log_prob.mean()
        self.log("train_loss", loss, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y_local, y_global = batch
        log_prob_local, log_prob_global = self(x, y_local, y_global)
        log_prob = torch.zeros_like(log_prob_local).to(log_prob_local.device)
        if self.local_loss:
            log_prob += log_prob_local
        if self.global_loss:
            log_prob += log_prob_global
        loss = -log_prob.mean()
        self.log("val_loss", loss, on_epoch=True)
        return loss
