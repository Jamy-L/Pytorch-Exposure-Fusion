# -*- coding:utf-8 -*-
"""
Created on Fri Nov 24 12:45:47 2023

@author: jamyl
"""

import numpy as np
import torch as th
import cv2
import matplotlib. pyplot as plt
import os
import glob

from pathlib import Path

burst_path = Path("data/")

burst = []

im_path_list = glob.glob(os.path.join(burst_path.as_posix(), '*.jpg'))
assert len(im_path_list) != 0, 'At least one .jpg file must be present in the burst folder.'


for im_path in im_path_list:
    burst.append(
        cv2.cvtColor(
            cv2.imread(im_path, cv2.IMREAD_UNCHANGED),
            cv2.COLOR_BGR2RGB)
        ) # flag to keep the same bit depth as original


burst = th.Tensor(burst)/255 # TODO adapt the value based on open cv dtype
burst = burst.movedim(-1, 1) # batch, channel, H, W format


def exposure_fusion(burst, w_sat=1, w_cont=1, w_exp=1):
    gray_burst = th.mean(burst, dim=1, keepdim=True)
    
    cont = compute_contrast(gray_burst)
    sat = compute_saturation(burst, gray_burst)
    exp = compute_well_exposedness(burst)
    
    weights = cont ** w_cont + sat ** w_sat + exp ** w_exp  # batched operation per image
    
    # normalise weights
    weights = weights/weights.sum(dim=0, keepdim=True)
    
    # Get gaussian pyramid for weights and images
    img_gaussian_pyramid = compute_gaussian_pyramid(burst)
    img_laplacian_pyramid = compute_laplacian_pyramid(img_gaussian_pyramid)

    weight_gaussian_pyramid = compute_gaussian_pyramid(weights)
    
    fused_laplacian_pyramid = merge_laplacian_pyramid(img_laplacian_pyramid,
                                                      weight_gaussian_pyramid)
    
    fused_image = collapse(fused_laplacian_pyramid).squeeze(0)
    
    return fused_image


def compute_contrast(gray_burst):
    
    k_laplacian = th.Tensor([[0,  1, 0],
                             [1, -4, 1],
                             [0,  1, 0]]).unsqueeze(0).unsqueeze(0)
    
    contrast = th.abs(
        th.nn.functional.conv2d(gray_burst, k_laplacian, padding="same")
        )
    
    return contrast

def compute_saturation(burst, gray_burst):
    sat = th.sqrt(
            th.mean(
                (burst - gray_burst)**2 , dim=1, keepdim=True))
    return sat

def compute_well_exposedness(burst):
    sigma = 0.2
    well_exposedness = th.exp(
        -th.sum(
            (burst-0.5)**2, dim=1, keepdim=True
            )/(2*sigma))
    return well_exposedness
    
def compute_gaussian_pyramid(tensor, n_levels=4):
    b, c, _, _ = tensor.size()
    assert c in [1, 3]
    downsample_kernel = th.Tensor([[1, 2, 1],
                                   [2, 4, 2],
                                   [1, 2, 1]]).unsqueeze(0).unsqueeze(0)/16
    
    # treat color channels as batch
    if c == 3:
        tensor = th.cat([tensor[:, 0],
                           tensor[:, 1],
                           tensor[:, 2]
                           ], dim=0).unsqueeze(1)
    
    
    pyramid = [tensor]
    for lvl in range(1, n_levels):
        # Manually enforce padding = same on top and left sides.
        # Bottoms and right sides only when necessary.

        padded = pad_for_downsample(pyramid[-1])
        pyramid.append(
                th.nn.functional.conv2d(padded, downsample_kernel,
                                        padding='valid', stride=2))
        
    if c == 3:
    # unpack color channels
        for lvl, stage in enumerate(pyramid):
            pyramid[lvl] = th.cat([stage[:b],
                                   stage[b:2*b],
                                   stage[2*b:3*b]], dim=1)
        
    return pyramid

def pad_for_downsample(prev_stage):
        _, _, h, w = prev_stage.size()
        if h%2 == 0:
            pad_vertical = (1, 0)
        else:
            pad_vertical = (1, 1)
        if w%2 == 0:
            pad_horizontal = (1, 0)
        else:
            pad_horizontal = (1, 1)
        pad = (*pad_vertical, *pad_horizontal)
            
        padded = th.nn.functional.pad(prev_stage,
                                      pad=pad,
                                      mode="replicate")
        return padded



def expand_level(tensor, target_shape=None):
    """
    Given a tensor, this function does a 2x upsampling by interpolating
    between samples. If target_shape is give, the function may pad a remaining
    row or column using paddin=same.
    

    Parameters
    ----------
    level : tensor [b, c, h, w]
        the tensor to expand
    target_shape : tuple (h, w), optional
        The target shape. The default is None, meaning that the target shape is
        (2h-1. 2w-1). If the target is bigger, the outputed interpolation will
        be padded

    Returns
    -------
    out : the expanded tensor

    """
    _, _, h, w = tensor.size()
    H = 2*h - 1
    W = 2*w - 1
    out = th.nn.functional.interpolate(tensor, size=(H, W),
                                       mode='bilinear',
                                       align_corners=True)
    
    if target_shape is not None:
        pad = (0, target_shape[0] - H,
               0, target_shape[1] - W)
        
        out = th.nn.functional.pad(out,
                                   pad=pad,
                                   mode="replicate")
        
        
    return out
    
def compute_laplacian_pyramid(gaussian_pyramid):
    n_levels = len(gaussian_pyramid)
    laplacian_pyramid = [None] * n_levels # init empty pyramid
    laplacian_pyramid[-1] = gaussian_pyramid[-1] # Coarsest level is just gaussian coarse.
    
    for lvl in range(n_levels-2, -1, -1):
        _, _, *target_shape = gaussian_pyramid[lvl].size()
        expanded_g = expand_level(gaussian_pyramid[lvl+1],
                                  target_shape=target_shape)
        laplacian_pyramid[lvl] = gaussian_pyramid[lvl] - expanded_g
        
    return laplacian_pyramid

def merge_laplacian_pyramid(img_pyramid, weight_pyramid):
    n_stages = len(img_pyramid)
    assert n_stages == len(weight_pyramid)
    
    merged_pyramid = [None] * n_stages
    for lvl, (weight, img) in enumerate(zip(weight_pyramid, img_pyramid)):
        merged_pyramid[lvl] = th.sum(weight * img,
                                     dim=0, keepdim=True)
    
    return merged_pyramid


def collapse(laplacian_pyramid):
    
    curr = laplacian_pyramid[-1]
    for stage in laplacian_pyramid[-2::-1]: # Reverse order, starting from the penultimate
        _, _, *target_shape = stage.size()
        curr = expand_level(curr,
                            target_shape=target_shape)
        curr = curr + stage
    return curr
    
out = exposure_fusion(burst, 1, 1, 1)
_, h, w = out.size()

plt.figure("exposure fusion")
plt.imshow(out.movedim(0, -1))


for i in range(burst.shape[0]):
    plt.figure(str(i))
    plt.imshow(burst[i, :].movedim(0, -1))
