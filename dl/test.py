import logging
import os
import warnings
from os import path as osp

warnings.filterwarnings("ignore", category=UserWarning)

import torch
from setproctitle import setproctitle

from dl.data import build_dataloader, build_dataset
from dl.models import build_model
from dl.utils import get_env_info, get_root_logger, get_time_str, make_exp_dirs
from dl.utils.options import dict2str, parse_options


def test_pipeline(root_path):
    # parse options, set distributed setting, set ramdom seed
    opt, _ = parse_options(root_path, is_train=False)

    torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True

    # mkdir and initialize loggers
    make_exp_dirs(opt)
    log_file = osp.join(opt["path"]["log"], f"test_{opt['name']}_{get_time_str()}.log")
    logger = get_root_logger(logger_name="dl", log_level=logging.INFO, log_file=log_file)
    logger.info(get_env_info())
    logger.info(dict2str(opt))

    # create test dataset and dataloader
    test_loaders = []
    for _, dataset_opt in sorted(opt["datasets"].items()):
        test_set = build_dataset(dataset_opt)
        test_loader = build_dataloader(
            test_set,
            dataset_opt,
            num_gpu=opt["num_gpu"],
            dist=opt["dist"],
            sampler=None,
            seed=opt["manual_seed"],
        )
        logger.info(f"Number of test images in {dataset_opt['name']}: {len(test_set)}")
        test_loaders.append(test_loader)

    # create model
    model = build_model(opt)

    for test_loader in test_loaders:
        test_set_name = test_loader.dataset.opt["name"]
        logger.info(f"Testing {test_set_name}...")
        model.validation(
            test_loader,
            current_iter=opt["name"],
            tb_logger=None,
        )


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    proc_title = "ZJJTest"
    setproctitle(proc_title)
    root_path = "/home/zjj/exps/hdr"
    test_pipeline(root_path)
