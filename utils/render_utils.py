import time
from typing import Any, Tuple

import torch


def unwrap_view(view: Any):
    if isinstance(view, dict):
        return view["cam"]
    return view


def view_time_to_cuda(view: Any) -> torch.Tensor:
    return torch.from_numpy(unwrap_view(view).time).to(torch.float32).cuda()


def timed_render(render_fn, *args, **kwargs) -> Tuple[dict, float]:
    torch.cuda.synchronize()
    start_time = time.time()
    render_pkg = render_fn(*args, **kwargs)
    torch.cuda.synchronize()
    elapsed = time.time() - start_time
    return render_pkg, elapsed
