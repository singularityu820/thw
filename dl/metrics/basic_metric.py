import torch

from dl.utils.registry import METRIC_REGISTRY


@METRIC_REGISTRY.register()
def calculate_rmse(gt: torch.Tensor, pred: torch.Tensor, transform=True, **kwargs):
    if transform:
        gt = gt * 0.5 + 0.5
        pred = pred * 0.5 + 0.5
    metric = torch.sqrt(torch.mean((gt - pred) ** 2))
    return metric


@METRIC_REGISTRY.register()
def calculate_pixel(gt: torch.Tensor, pred: torch.Tensor, transform=True, **kwargs):
    if transform:
        gt = gt * 0.5 + 0.5
        pred = pred * 0.5 + 0.5
    metric = torch.mean(torch.abs(gt - pred))
    return metric
