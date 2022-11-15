from functools import partial
import torch

grid_sample = partial(
    torch.nn.functional.grid_sample,
    padding_mode='zeros',
    align_corners=True,
    mode="bilinear"
)