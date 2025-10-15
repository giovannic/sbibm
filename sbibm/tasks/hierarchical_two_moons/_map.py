import math

import torch


def two_moons_map(parameters: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    ang = torch.tensor([-math.pi / 4.0]).to(device=parameters.device)
    c = torch.cos(ang)
    s = torch.sin(ang)
    z0 = (c * parameters[:, 0] - s * parameters[:, 1]).reshape(-1, 1)
    z1 = (s * parameters[:, 0] + c * parameters[:, 1]).reshape(-1, 1)
    return p + torch.cat((-torch.abs(z0), z1), dim=1)


def two_moons_map_inv(parameters: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    ang = torch.tensor([-math.pi / 4.0]).to(device=parameters.device)
    c = torch.cos(ang)
    s = torch.sin(ang)
    z0 = (c * parameters[:, 0] - s * parameters[:, 1]).reshape(-1, 1)
    z1 = (s * parameters[:, 0] + c * parameters[:, 1]).reshape(-1, 1)
    return x - torch.cat((-torch.abs(z0), z1), dim=1)
