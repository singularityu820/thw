import importlib
import os.path as osp
from copy import deepcopy

from dl.utils import scandir
from dl.utils.registry import METRIC_REGISTRY

from .niqe import calculate_niqe
from .psnr_ssim import calculate_psnr, calculate_ssim

metric_folder = osp.dirname(osp.abspath(__file__))
metric_filenames = [
    osp.splitext(osp.basename(v))[0] for v in scandir(metric_folder) if v.endswith("_metric.py")
]
# import all the metric modules
_model_modules = [
    importlib.import_module(f"dl.metrics.{file_name}") for file_name in metric_filenames
]


__all__ = ["calculate_psnr", "calculate_ssim", "calculate_niqe"]


def calculate_metric(data, opt):
    """Calculate metric from data and options.

    Args:
        opt (dict): Configuration. It must contain:
            type (str): Model type.
    """
    opt = deepcopy(opt)
    metric_type = opt.pop("type")
    metric = METRIC_REGISTRY.get(metric_type)(**data, **opt)
    return metric
