# -*- coding: utf-8 -*-
"""
Created on Thu Dec 21 17:21:08 2023

@author: jamyl
"""

import torch as th

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

        padded = pad_stage_to_downsample(pyramid[-1])
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

def pad_stage_to_downsample(stage):
        _, _, h, w = stage.size()
        if h%2 == 0:
            pad_vertical = (1, 0)
        else:
            pad_vertical = (1, 1)
        if w%2 == 0:
            pad_horizontal = (1, 0)
        else:
            pad_horizontal = (1, 1)
        pad = (*pad_horizontal, *pad_vertical)
            
        padded = th.nn.functional.pad(stage,
                                      pad=pad,
                                      mode="replicate")
        return padded



def expand_stage(tensor, target_shape=None):
    """
    Given a tensor, this function does a 2x upsampling by interpolating
    between samples. If target_shape is given, the function may pad a remaining
    row or column using paddin=replicate.
    

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
        pad = (0, target_shape[1] - W,
               0, target_shape[0] - H)
        
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
        expanded_g = expand_stage(gaussian_pyramid[lvl+1],
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

def collapse_pyramid(laplacian_pyramid):
    
    curr = laplacian_pyramid[-1]
    for stage in laplacian_pyramid[-2::-1]: # Reverse order, starting from the penultimate
        _, _, *target_shape = stage.size()
        curr = expand_stage(curr,
                            target_shape=target_shape)
        curr = curr + stage
    return curr