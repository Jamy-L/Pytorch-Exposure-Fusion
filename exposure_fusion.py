# -*- coding:utf-8 -*-
"""
Created on Fri Nov 24 12:45:47 2023

@author: jamyl
"""


import torch as th

from pyramids import compute_gaussian_pyramid, compute_laplacian_pyramid, merge_laplacian_pyramid, collapse_pyramid


def exposure_fusion(burst, w_sat=1, w_cont=1, w_exp=1, n_levels=4):
    """

    Parameters
    ----------
    burst : Tensor [N, C, H, W]
        Input sequence of N images, C color channels, H by W images.
    w_sat : int, optional
        The saturation importance weight. The default is 1.
    w_cont : int, optional
        The contrast importance weight. The default is 1.
    w_exp : int, optional
        The well-exposed importance weight. The default is 1.
    n_levels: int optional
        The number of levels in the pyramids. The default is 4

    Returns
    -------
    fused_image : Tensor [C, H, W]
        The fused image, with compressed dynamic range.

    """
    gray_burst = th.mean(burst, dim=1, keepdim=True)
    
    cont = compute_contrast(gray_burst)
    sat = compute_saturation(burst, gray_burst)
    exp = compute_well_exposedness(burst)
    
    weights = (cont ** w_cont) * (sat ** w_sat) * (exp ** w_exp)
    # normalise weights
    weights = weights/weights.sum(dim=0, keepdim=True)
    # Normalisation will give Nan if all frames have 0 weight at 1 pixel.
    # In this case, all of them get the same weight
    weights =  weights.nan_to_num(nan = 1/burst.size(0))

    # Get gaussian pyramid for weights and images
    img_gaussian_pyramid = compute_gaussian_pyramid(burst, n_levels=n_levels)
    img_laplacian_pyramid = compute_laplacian_pyramid(img_gaussian_pyramid)

    weight_gaussian_pyramid = compute_gaussian_pyramid(weights, n_levels=n_levels)
    
    fused_laplacian_pyramid = merge_laplacian_pyramid(img_laplacian_pyramid,
                                                      weight_gaussian_pyramid)
    
    fused_image = collapse_pyramid(fused_laplacian_pyramid).squeeze(0)
    
    return fused_image


def compute_contrast(gray_burst):
    
    k_laplacian = th.tensor([[0,  1, 0],
                             [1, -4, 1],
                             [0,  1, 0]],
                            device=gray_burst.device,
                            dtype=gray_burst.dtype 
                            ).unsqueeze(0).unsqueeze(0)
    
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
    

