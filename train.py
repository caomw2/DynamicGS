#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
os.environ["OMP_NUM_THREADS"] = "1"  # noqa
os.environ["MKL_NUM_THREADS"] = "1"  # noqa
import sys
import uuid
from argparse import ArgumentParser
from random import randint

import torch

from arguments import ModelParams, PipelineParams, OptimizationParams
from gaussian_renderer import render
from options.gaussian_option import Gaussian_Options
from scene import Scene, GaussianModel
from utils.image_utils import psnr
from utils.loss_utils import l1_loss, ssim


class BatchState:
    def __init__(self):
        self.losses = []
        self.radii = []
        self.visibility_filters = []
        self.viewspace_points = []

    def add(self, loss, radii, visibility_filter, viewspace_point_tensor):
        self.losses.append(loss)
        self.radii.append(radii.unsqueeze(0))
        self.visibility_filters.append(visibility_filter.unsqueeze(0))
        self.viewspace_points.append(viewspace_point_tensor)

    def ready(self, batch_size):
        return len(self.losses) == batch_size

    def backward_and_collect(self):
        total_loss = torch.stack(self.losses, dim=0).sum()
        total_loss.backward()

        radii = torch.cat(self.radii, dim=0).max(dim=0).values
        visibility_filter = torch.cat(self.visibility_filters, dim=0).any(dim=0)
        viewspace_point_tensor = self.viewspace_points[-1]

        self.clear()
        return radii, visibility_filter, viewspace_point_tensor

    def clear(self):
        self.losses.clear()
        self.radii.clear()
        self.visibility_filters.clear()
        self.viewspace_points.clear()


def ensure_model_path(dataset):
    if dataset.model_path:
        os.makedirs(dataset.model_path, exist_ok=True)
        return

    if os.getenv("OAR_JOB_ID"):
        unique_str = os.getenv("OAR_JOB_ID")
    else:
        unique_str = str(uuid.uuid4())
    dataset.model_path = os.path.join("./output/", unique_str[0:10])
    os.makedirs(dataset.model_path, exist_ok=True)


def get_input_dimensions(opt):
    return 2 * opt.time_freq, 6 * opt.xyz_freq


def load_checkpoint_metadata(gaussians, checkpoint_path):
    model_params, opt_dict, first_iter = torch.load(checkpoint_path)
    gaussians.N_pcd_init = model_params["_xyz"].shape[0]
    gaussians.active_sh_degree = gaussians.max_sh_degree
    gaussians.final_kpts_num = model_params["super_gaussians"].shape[0] if "super_gaussians" in model_params else None
    return model_params, opt_dict, first_iter


def create_gaussians(dataset, opt, args):
    time_input_dim, xyz_input_dim = get_input_dimensions(opt)
    gaussians = GaussianModel(dataset.sh_degree, args)
    gaussians.set_inputDim(time_input_dim, xyz_input_dim)
    first_iter = 0
    checkpoint_state = None

    if args.start_checkpoint:
        model_params, opt_dict, first_iter = load_checkpoint_metadata(gaussians, args.start_checkpoint)
        checkpoint_state = (model_params, opt_dict)

    return gaussians, first_iter, checkpoint_state


def restore_training_state(gaussians, opt, checkpoint_state, first_iter):
    if checkpoint_state is None:
        return
    model_params, opt_dict = checkpoint_state
    gaussians.restore(opt_dict, opt, first_iter)
    gaussians.load_state_dict(model_params, strict=False)


def get_background_tensor(dataset):
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    return torch.tensor(bg_color, dtype=torch.float32, device="cuda")


def sample_training_view(scene, viewpoint_stack):
    if not viewpoint_stack:
        viewpoint_stack = scene.getTrainCameras().copy()
    viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))
    return viewpoint_cam, viewpoint_stack


def compute_time_noise(iteration, args, gaussians, max_frame):
    decay_noise = torch.randn([1], device="cuda") * args.time_noise_ratio / max_frame
    if args.use_time_decay:
        if iteration >= gaussians.second_stage_iter:
            decay_noise *= 1 - min(1, (iteration - gaussians.second_stage_iter) / (args.time_noise_iteration * 2))
        else:
            decay_noise *= 1 - min(1, iteration / args.time_noise_iteration)
    else:
        decay_noise = torch.zeros_like(decay_noise)
    return decay_noise


def render_training_view(viewpoint_cam, gaussians, pipe, background, iteration, args, scene):
    time_ = torch.from_numpy(viewpoint_cam.time).to(torch.float32).to("cuda")
    time_ = time_ + compute_time_noise(iteration, args, gaussians, scene.total_frame)
    return render(viewpoint_cam, gaussians, pipe, background, delta=None, time=time_, it=iteration)


def compute_training_loss(image, gt_image, gaussians, opt, iteration):
    l1_value = l1_loss(image, gt_image)
    psnr_value = psnr(image, gt_image).mean().double()
    loss = (1.0 - opt.lambda_dssim) * l1_value + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
    loss += gaussians.get_loss(iteration)
    return loss, l1_value, psnr_value


def update_densification(gaussians, dataset, scene, opt, args, iteration, radii, visibility_filter, viewspace_point_tensor):
    if iteration < opt.densify_until_iter:
        gaussians.max_radii2D[visibility_filter] = torch.max(
            gaussians.max_radii2D[visibility_filter],
            radii[visibility_filter],
        )
        gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

        should_adjust_density = iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0
        size_threshold = 20 if iteration > opt.opacity_reset_interval else None

        if should_adjust_density and gaussians.get_xyz.shape[0] < args.max_gaussian_size:
            gaussians.densify(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)

        if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
            gaussians.reset_opacity()

        if should_adjust_density:
            gaussians.prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)


def update_adaptive_densification(gaussians, opt, args, iteration, radii, visibility_filter, viewspace_point_tensor):
    adaptive_limit = args.adaptive_end_iter + gaussians.second_stage_iter
    max_super_gaussians = args.max_points + args.adaptive_points_num

    if iteration >= adaptive_limit or gaussians.super_gaussians.shape[0] >= max_super_gaussians:
        return

    if gaussians.second_stage:
        gaussians.max_radii2D[visibility_filter] = torch.max(
            gaussians.max_radii2D[visibility_filter],
            radii[visibility_filter],
        )
        gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

    should_update_kpts = iteration > args.adaptive_from_iter + gaussians.second_stage_iter
    should_update_kpts = should_update_kpts and iteration % args.adaptive_interval == 0

    if not should_update_kpts:
        return

    if gaussians.new_xyz is not None:
        gaussians.densification_motion_postfix(gaussians.new_xyz, gaussians.new_motion_feature)
        gaussians.new_kpts_init()
    if args.densify_from_grad == "True":
        gaussians.densify_kpts(opt.densify_grad_threshold, mode="down_sampling")
    print(f"At iteration {iteration}, there are {gaussians.super_gaussians.shape[0]} super Gaussians!")


def maybe_save_scene(scene, iteration, save_iterations):
    if iteration in save_iterations:
        print("\n[ITER {}] Saving Gaussians".format(iteration))
        scene.save(iteration)


def maybe_save_checkpoint(scene, gaussians, iteration, checkpoint_iterations):
    if iteration in checkpoint_iterations:
        print("\n[ITER {}] Saving Checkpoint".format(iteration))
        torch.save(
            (gaussians.state_dict(), gaussians.optimizer.state_dict(), iteration),
            scene.model_path + "/chkpnt" + str(iteration) + ".pth",
        )


def training(dataset, opt, pipe, args, batch):
    ensure_model_path(dataset)
    gaussians, first_iter, checkpoint_state = create_gaussians(dataset, opt, args)
    scene = Scene(dataset, gaussians, ratio=args.ratio)
    gaussians.training_setup(opt)
    restore_training_state(gaussians, opt, checkpoint_state, first_iter)

    background = get_background_tensor(dataset)
    batch_state = BatchState()
    viewpoint_stack = None
    ema_loss_for_log = 0.0
    ema_psnr_for_log = 0.0
    first_iter += 1

    for iteration in range(first_iter, opt.iterations + 1):
        gaussians.update_learning_rate(iteration)

        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        viewpoint_cam, viewpoint_stack = sample_training_view(scene, viewpoint_stack)
        render_pkg = render_training_view(viewpoint_cam, gaussians, pipe, background, iteration, args, scene)
        image = render_pkg["render"]
        viewspace_point_tensor = render_pkg["viewspace_points"]
        visibility_filter = render_pkg["visibility_filter"]
        radii = render_pkg["radii"]

        gt_image = viewpoint_cam.original_image.cuda()
        loss, _, psnr_value = compute_training_loss(image, gt_image, gaussians, opt, iteration)

        batch_state.add(loss, radii, visibility_filter, viewspace_point_tensor)
        if not batch_state.ready(batch):
            continue

        radii, visibility_filter, viewspace_point_tensor = batch_state.backward_and_collect()

        with torch.no_grad():
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_psnr_for_log = 0.4 * psnr_value + 0.6 * ema_psnr_for_log

            if iteration % 10 == 0:
                print(
                    "[ITER {}] Loss {:.7f} PSNR {:.3f} P_NUM {}".format(
                        iteration,
                        ema_loss_for_log,
                        ema_psnr_for_log,
                        gaussians.get_xyz.shape[0],
                    )
                )

            maybe_save_scene(scene, iteration, args.save_iterations)
            update_densification(gaussians, dataset, scene, opt, args, iteration, radii, visibility_filter, viewspace_point_tensor)
            update_adaptive_densification(gaussians, opt, args, iteration, radii, visibility_filter, viewspace_point_tensor)

            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)

            maybe_save_checkpoint(scene, gaussians, iteration, args.checkpoint_iterations)


if __name__ == "__main__":
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    options = Gaussian_Options(parser)
    options.initial()

    parser = options.get_parser()

    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    print("Optimizing " + args.model_path)

    seed = 2024 * args.seed
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    training(lp.extract(args), op.extract(args), pp.extract(args), args, batch=args.batch)

    print("\nTraining complete.")
