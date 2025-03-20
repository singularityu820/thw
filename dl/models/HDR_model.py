import json
import os.path as osp
from copy import deepcopy
from typing import OrderedDict

import torch
from torchvision.utils import save_image
from tqdm import tqdm

from dl.archs import build_network
from dl.losses import build_loss
from dl.metrics import calculate_metric
from dl.models.base_model import BaseModel
from dl.utils.logger import get_root_logger
from dl.utils.registry import MODEL_REGISTRY
from dl.utils.render_util import svBRDF


@MODEL_REGISTRY.register()
class HDRModel(BaseModel):
    def __init__(self, opt):
        super(HDRModel, self).__init__(opt)
        self.renderer = svBRDF(opt["brdf_args"])
        self.init_rendering()
        self.init_network()
        if self.is_train:
            self.init_training_settings()

    def init_rendering(self):
        # init lighting direction
        self.surface = self.renderer.surface(256, 1).to("cuda")
        self.pos = torch.tensor((0, 0, self.opt["distance"]), device="cuda").unsqueeze(0)
        self.light_dir, _, self.light_dis, _ = self.renderer.torch_generate(
            self.pos, self.pos, pos=self.surface
        )
        if self.opt["val"].get("IsTest", False):
            self.light_poses = [
                torch.tensor([-0.5, -0.5, 2.414], device="cuda"),
                torch.tensor([0, -0.5, 2.414], device="cuda"),
                torch.tensor([0.5, -0.5, 2.414], device="cuda"),
                torch.tensor([-0.5, 0, 2.414], device="cuda"),
                torch.tensor([0, 0, 2.414], device="cuda"),
                torch.tensor([0.5, 0, 2.414], device="cuda"),
                torch.tensor([-0.5, 0.5, 2.414], device="cuda"),
                torch.tensor([0, 0.5, 2.414], device="cuda"),
                torch.tensor([0.5, 0.5, 2.414], device="cuda"),
            ]
            self.light_dirs = []
            self.light_diss = []
            for light_pos in self.light_poses:
                light_dir, _, light_dis, _ = self.renderer.torch_generate(
                    light_pos.unsqueeze(0), light_pos.unsqueeze(0), pos=self.surface
                )
                self.light_dirs.append(light_dir.unsqueeze(1))
                self.light_diss.append(light_dis.unsqueeze(1))

    def init_network(self):
        self.net_g = build_network(self.opt["network_g"])
        self.net_g = self.model_to_device(self.net_g)
        load_path_net_g = self.opt["path"].get("pretrain_network_g", None)
        if load_path_net_g is not None:
            self.load_network(
                self.net_g,
                load_path_net_g,
                self.opt["path"].get("strict_load_g", True),
                "params",
            )
        self.net_g = torch.compile(self.net_g)

    def init_training_settings(self):
        self.net_g.train()

        train_opt = self.opt["train"]
        self.cri_pix = build_loss(train_opt["pixel_opt"]).to(self.device)

        self.setup_optimizers()
        self.setup_schedulers()

    def setup_optimizers(self):
        train_opt = self.opt["train"]
        optim_params = [param for param in self.net_g.parameters()]

        optim_type = train_opt["optim_g"].pop("type")
        self.optimizer = self.get_optimizer(optim_type, optim_params, **train_opt["optim_g"])
        self.optimizers.append(self.optimizer)
        for p in self.net_g.parameters():
            p.requires_grad = True

    def log_normalization(self, img, eps=1e-2):
        return (torch.log(img + eps) - torch.log(torch.ones((1,), device="cuda") * eps)) / (
            torch.log(1 + torch.ones((1,), device="cuda") * eps)
            - torch.log(torch.ones((1,), device="cuda") * eps)
        )

    def minmax_normalization(self, img):
        return (img - torch.amin(img, dim=(2, 3), keepdim=True)) / (
            torch.amax(img, dim=(2, 3), keepdim=True) - torch.amin(img, dim=(2, 3), keepdim=True)
        )

    def render(self, svbrdf, hdr=True):
        point = self.renderer._render(
            svbrdf,
            self.light_dir.unsqueeze(1),
            self.light_dir.unsqueeze(1),
            self.light_dis.unsqueeze(1),
            hdr=hdr,
        )

        return point

    def preprocess_svbrdf(self, svbrdf):
        n, d, r, s = torch.split(svbrdf, 256, dim=-1)
        svbrdf = torch.cat([n, d, r[:, :1], s], dim=1) * 2 - 1
        return svbrdf

    def prepare_input(self):
        input = self.render(self.gt_svbrdf)
        input_log = self.minmax_normalization(input)
        input = torch.cat([input, input_log], dim=1)
        return input

    def feed_data(self, data, random=False):
        self.name, svbrdf = data["name"], data["svbrdf"].cuda()
        self.gt_svbrdf = self.preprocess_svbrdf(svbrdf)
        self.input = self.prepare_input()

    def optimize_parameters(self, current_iter):
        loss_dict = OrderedDict()

        self.optimizer.zero_grad()
        self.pred_svbrdf = self.net_g(self.input)

        l_total = 0

        l_pixel = self.cri_pix(self.pred_svbrdf, self.gt_svbrdf)
        l_total += l_pixel
        loss_dict["l_pixel"] = l_pixel

        l_total.backward()
        self.optimizer.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)

    def test(self):
        self.net_g.eval()
        with torch.no_grad():
            self.pred_svbrdf = self.net_g(self.input)
        self.net_g.train()

    def validation(self, dataloader, current_iter, tb_logger):
        dataset_name = dataloader.dataset.opt["name"]
        IsTest = self.opt["val"].get("IsTest", False)
        with_metrics = self.opt["val"].get("metrics") is not None
        if with_metrics:
            if IsTest:
                self.metric_results = {
                    metric: {key: 0 for key in ["n", "d", "r", "s", "svbrdf", "render"]}
                    for metric in self.opt["val"]["metrics"].keys()
                }
            else:
                self.metric_results = {metric: 0 for metric in self.opt["val"]["metrics"].keys()}
        if self.opt.get("pbar", True):
            pbar = tqdm(total=len(dataloader), unit="image")
        if not IsTest:
            svbrdfs_vis = []
        for idx, val_data in enumerate(dataloader):
            self.feed_data(val_data)
            self.test()
            if not IsTest:
                svbrdfs_vis.append(self.save_visuals())

            if with_metrics and not self.opt["datasets"]["val"].get("real_input", False):
                # calculate metrics
                opt_metric = deepcopy(self.opt["val"]["metrics"])
                for name, opt_ in opt_metric.items():
                    if IsTest:
                        self.metric_results[name]["n"] += calculate_metric(
                            {"pred": self.pred_svbrdf[:, :3], "gt": self.gt_svbrdf[:, :3]}, opt_
                        ).item() / len(dataloader)
                        self.metric_results[name]["d"] += calculate_metric(
                            {"pred": self.pred_svbrdf[:, 3:6], "gt": self.gt_svbrdf[:, 3:6]}, opt_
                        ).item() / len(dataloader)
                        self.metric_results[name]["r"] += calculate_metric(
                            {"pred": self.pred_svbrdf[:, 6:7], "gt": self.gt_svbrdf[:, 6:7]}, opt_
                        ).item() / len(dataloader)
                        self.metric_results[name]["s"] += calculate_metric(
                            {"pred": self.pred_svbrdf[:, 7:], "gt": self.gt_svbrdf[:, 7:]}, opt_
                        ).item() / len(dataloader)
                        self.metric_results[name]["svbrdf"] += calculate_metric(
                            {"pred": self.pred_svbrdf, "gt": self.gt_svbrdf}, opt_
                        ).item() / len(dataloader)
                        errors = 0
                        self.gt_render = []
                        self.pred_render = []
                        for light_dir, light_dis in zip(self.light_dirs, self.light_diss):
                            gt_ = self.renderer._render(
                                self.gt_svbrdf,
                                light_dir,
                                light_dir,
                                light_dis,
                                hdr=False,
                            )
                            pred_ = self.renderer._render(
                                self.pred_svbrdf,
                                light_dir,
                                light_dir,
                                light_dis,
                                hdr=False,
                            )
                            self.gt_render.append(gt_)
                            self.pred_render.append(pred_)
                            errors += calculate_metric(
                                {"pred": pred_ * 2 - 1, "gt": gt_ * 2 - 1},
                                opt_,
                            ).item() / len(dataloader)
                        self.metric_results[name]["render"] += errors / len(self.light_dirs)
                    else:
                        self.metric_results[name] += calculate_metric(
                            {"pred": self.pred_svbrdf, "gt": self.gt_svbrdf}, opt_
                        ) / len(dataloader)
            if IsTest:
                self.save_single_visual()
            torch.cuda.empty_cache()
            if self.opt.get("pbar", True):
                pbar.update(1)
                pbar.set_description("Testing")
        if self.opt.get("pbar", True):
            pbar.close()

        if with_metrics:
            if IsTest:
                # save metrics to json file at the parent folder of self.opt["path"]["visualization"]
                save_path = osp.join(
                    osp.dirname(self.opt["path"]["visualization"]),
                    "metrics.json",
                )
                with open(save_path, "w") as f:
                    json.dump(self.metric_results, f)

            else:
                self._log_validation_metric_values(current_iter, dataset_name, tb_logger)

        if not self.opt["datasets"]["val"].get("real_input", False) and not IsTest:
            svbrdfs_vis = torch.cat(svbrdfs_vis, dim=0)
            save_image(
                svbrdfs_vis,
                osp.join(
                    self.opt["path"]["visualization"],
                    f"{current_iter}.jpg",
                ),
                nrow=4,
                padding=5,
                pad_value=1,
            )

    def _log_validation_metric_values(self, current_iter, dataset_name, tb_logger):
        log_str = f"Validation {dataset_name};\t"
        for metric, value in self.metric_results.items():
            log_str += f"\t # {metric}: {value:.4f}\t"
        logger = get_root_logger()
        logger.info(log_str)
        if tb_logger:
            for metric, value in self.metric_results.items():
                tb_logger.add_scalar(f"metrics/{metric}", value, current_iter)

    def save_visuals(self):
        gt_svbrdf = self.gt_svbrdf * 0.5 + 0.5
        gt_n, gt_d, gt_r, gt_s = torch.split(gt_svbrdf, [3, 3, 1, 3], dim=1)
        gt_svbrdf = torch.cat([gt_n, gt_d, gt_r.repeat(1, 3, 1, 1), gt_s], dim=-1)

        pred_svbrdf = self.pred_svbrdf * 0.5 + 0.5
        pred_n, pred_d, pred_r, pred_s = torch.split(pred_svbrdf, [3, 3, 1, 3], dim=1)
        pred_svbrdf = torch.cat([pred_n, pred_d, pred_r.repeat(1, 3, 1, 1), pred_s], dim=-1)
        return torch.cat([gt_svbrdf, pred_svbrdf], dim=-2)

    def save_single_visual(self):
        gt_svbrdf = self.gt_svbrdf * 0.5 + 0.5
        gt_n, gt_d, gt_r, gt_s = torch.split(gt_svbrdf, [3, 3, 1, 3], dim=1)
        gt_svbrdf = torch.cat([gt_n, gt_d, gt_r.repeat(1, 3, 1, 1), gt_s], dim=-1)
        self.gt_render = [torch.clip(img**0.4545, 0, 1) for img in self.gt_render]

        gt = torch.cat([gt_svbrdf, *self.gt_render], dim=-1)

        pred_svbrdf = self.pred_svbrdf * 0.5 + 0.5
        pred_n, pred_d, pred_r, pred_s = torch.split(pred_svbrdf, [3, 3, 1, 3], dim=1)
        pred_svbrdf = torch.cat([pred_n, pred_d, pred_r.repeat(1, 3, 1, 1), pred_s], dim=-1)
        self.pred_render = [torch.clip(img**0.4545, 0, 1) for img in self.pred_render]
        pred = torch.cat([pred_svbrdf, *self.pred_render], dim=-1)
        result = torch.cat([gt, pred], dim=-2)
        save_image(
            result,
            osp.join(
                self.opt["path"]["visualization"],
                f"{self.name[0][:-4]}.jpg",
            ),
        )

    def save(self, epoch, current_iter):
        self.save_network(self.net_g, "net_g", current_iter)
        self.save_training_state(epoch, current_iter)
