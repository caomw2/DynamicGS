from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from PIL import Image
import torch
import torchvision.transforms.functional as tf
from pytorch_msssim import ms_ssim

from lpipsPyTorch import lpips
from utils.image_utils import psnr
from utils.loss_utils import ssim


TensorList = List[torch.Tensor]


def load_image_tensors(
    image_dir: str,
    resize: bool = False,
    resize_ratio: float = 0.5,
    device: str = "cuda",
    skip_keywords: Iterable[str] = (),
) -> Tuple[TensorList, List[str]]:
    images: TensorList = []
    image_names: List[str] = []
    directory = Path(image_dir)

    for image_path in sorted(directory.iterdir()):
        if not image_path.is_file():
            continue
        if any(keyword in image_path.name for keyword in skip_keywords):
            continue

        image = Image.open(image_path)
        if resize:
            width, height = image.size
            image = image.resize((int(width * resize_ratio), int(height * resize_ratio)))

        images.append(tf.to_tensor(image).unsqueeze(0)[:, :3, :, :].to(device))
        image_names.append(image_path.name)

    return images, image_names


def compute_metrics(render: torch.Tensor, gt: torch.Tensor) -> Dict[str, torch.Tensor]:
    ms_ssim_score = ms_ssim(render, gt, data_range=1, size_average=True)
    return {
        "SSIM": ssim(render, gt),
        "PSNR": psnr(render, gt),
        "LPIPS-vgg": lpips(render, gt, net_type="vgg"),
        "LPIPS-alex": lpips(render, gt, net_type="alex"),
        "MS-SSIM": ms_ssim_score,
        "D-SSIM": (1 - ms_ssim_score) / 2,
    }


def tensor_metric_dict_to_floats(metrics: Dict[str, List[torch.Tensor]]) -> Dict[str, float]:
    return {
        name: torch.tensor(values).mean().item()
        for name, values in metrics.items()
    }


def per_view_metric_dict(metrics: Dict[str, List[torch.Tensor]], image_names: List[str]) -> Dict[str, Dict[str, float]]:
    return {
        metric_name: {
            image_name: metric_value
            for metric_value, image_name in zip(torch.tensor(metric_values).tolist(), image_names)
        }
        for metric_name, metric_values in metrics.items()
    }
