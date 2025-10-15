"""
Training utilities for HierarchicalDeepSet.

Provides training loops using PyTorch Lightning for
the hierarchical deep set model.
"""

import torch
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader, TensorDataset


def train_hierarchical_deepset(
    model,
    x_set,
    y_local,
    y_global,
    num_epochs=50,
    batch_size=128,
    learning_rate=3e-4,
    validation_split=0.1,
    device="cpu",
    verbose=False,
):
    """
    Train a HierarchicalDeepSetInference model.

    Args:
        model: HierarchicalDeepSetInference instance
        x_set: Training observations, shape
               (num_datasets, num_events, height, width)
        y_local: Local parameters, shape
                 (num_datasets, num_events, dim_local)
        y_global: Global parameters, shape
                  (num_datasets, dim_global)
        num_epochs: Number of training epochs
        batch_size: Training batch size
        learning_rate: Adam learning rate
        validation_split: Fraction of data for validation
        device: "cpu" or "cuda"
        verbose: Print training progress

    Returns:
        model: Trained HierarchicalDeepSetInference instance
    """
    # Move data to device
    x_set = x_set.to(device)
    y_local = y_local.to(device)
    y_global = y_global.to(device)

    # Precompute set mask
    n_set_max = model.deep_set.n_set_max
    set_size = torch.randint(
        low=1,
        high=n_set_max + 1,
        size=(x_set.shape[0],),
        dtype=torch.float,
    )
    mask = (
        torch.arange(n_set_max).expand(len(set_size), n_set_max)
        < torch.Tensor(set_size)[:, None]
    ).to(x_set.device)

    # Create train/val split
    num_train = int(len(x_set) * (1 - validation_split))
    indices = torch.randperm(len(x_set))
    train_indices = indices[:num_train]
    val_indices = indices[num_train:]

    # Create datasets
    train_dataset = TensorDataset(
        x_set[train_indices],
        y_local[train_indices],
        y_global[train_indices],
        set_size[train_indices],
        mask[train_indices],
    )
    val_dataset = TensorDataset(
        x_set[val_indices],
        y_local[val_indices],
        y_global[val_indices],
        set_size[val_indices],
        mask[val_indices],
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    # Update model learning rate
    model.lr = learning_rate

    # Create trainer
    trainer_kwargs = {
        "max_epochs": num_epochs,
        "enable_progress_bar": verbose,
        "enable_model_summary": verbose,
        "logger": False,
        "enable_checkpointing": False,
    }
    if device == "cuda":
        trainer_kwargs["devices"] = [0]
    else:
        trainer_kwargs["devices"] = 1

    trainer = Trainer(**trainer_kwargs)

    # Train
    trainer.fit(
        model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
    )

    return model
