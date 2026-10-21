import math
import numpy as np
import torch
import torchvision


def transform_rescale(scale, voxels, disps=None, poses=None, intrinsics=None):
    """Transform voxels/images, depth maps, poses and intrinsics (n_frames,*)"""
    H, W = voxels.shape[-2:]
    nH, nW = math.floor(scale * H), math.floor(scale * W)
    resize = torchvision.transforms.Resize((nH, nW))

    voxels = resize(voxels)
    if disps is not None:
        disps = resize(disps)
    if poses is not None:
        poses = transform_rescale_poses(scale, poses)
    if intrinsics is not None:
        intrinsics = scale * intrinsics

    return voxels, disps, poses, intrinsics

def transform_rescale_poses(scale, poses):
    # import local : lietorch charge les extensions CUDA, inutiles au reste du module
    # (prétraitements et itérateurs d'events, utilisables sans GPU)
    from devo.lietorch import SE3
    s = torch.tensor(scale)
    poses = SE3(poses).scale(s).data
    return poses