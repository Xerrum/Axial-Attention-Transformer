

import numpy as np
import torch

def compute_dist(pos1, pos2, spherical_dist=True, r=6371000):
    if spherical_dist:
        pos1 = pos1 * np.pi / 180
        pos2 = pos2 * np.pi / 180
        cos_lat1 = np.cos(pos1[..., 0])
        cos_lat2 = np.cos(pos2[..., 0])
        cos_lat_d = np.cos(pos1[..., 0] - pos2[..., 0])
        cos_lon_d = np.cos(pos1[..., 1] - pos2[..., 1])
        return r * np.arccos(cos_lat_d - cos_lat1 * cos_lat2 * (1 - cos_lon_d))
    else:
        return np.sqrt(np.sum((pos1 - pos2) ** 2, axis=-1))


def window_view(x:torch.Tensor, window_size: int):
    shape = list(x.shape)
    shape[-1] = int(shape[-1]*window_size)
    shape[-2] = int(shape[-2]/window_size)
    return x.view(shape)