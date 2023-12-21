# -*- coding:utf-8 -*-
"""
Created on Fri Nov 24 12:45:47 2023

@author: jamyl
"""


import torch as th

from pyramids import compute_gaussian_pyramid, compute_laplacian_pyramid, merge_laplacian_pyramid, collapse_pyramid


def exposure_fusion(burst, w_sat=1, w_cont=1, w_exp=1):
    gray_burst = th.mean(burst, dim=1, keepdim=True)
    
    cont = compute_contrast(gray_burst)
    sat = compute_saturation(burst, gray_burst)
    exp = compute_well_exposedness(burst)
    
    weights = cont ** w_cont + sat ** w_sat + exp ** w_exp
    
    # normalise weights
    weights = weights/weights.sum(dim=0, keepdim=True)
    
    # Get gaussian pyramid for weights and images
    img_gaussian_pyramid = compute_gaussian_pyramid(burst)
    img_laplacian_pyramid = compute_laplacian_pyramid(img_gaussian_pyramid)

    weight_gaussian_pyramid = compute_gaussian_pyramid(weights)
    
    fused_laplacian_pyramid = merge_laplacian_pyramid(img_laplacian_pyramid,
                                                      weight_gaussian_pyramid)
    
    fused_image = collapse_pyramid(fused_laplacian_pyramid).squeeze(0)
    
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
    
