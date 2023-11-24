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

# for im in burst:
#     plt.figure()
#     plt.imshow(im)




def exposure_fusion(burst, w_sat, w_cont, w_exp):
    
    cont, gray_burst = compute_contrast(burst)
    sat = compute_saturation(burst, gray_burst)
    exp = compute_well_exposedness(burst)
    
    
    # TO visualize masks
    # n_frames = cont.size(0)
    # for i in range(n_frames):
    #     plt.figure("exp "+str(i))
    #     plt.imshow(exp[i])
    #     plt.colorbar()
    #     plt.figure("sat "+str(i))
    #     plt.imshow(sat[i])
    #     plt.colorbar()
    #     plt.figure("contrast "+str(i))
    #     plt.imshow(cont[i], vmin=0, vmax=2)
    #     plt.colorbar()
    
    
    weights = cont ** w_cont + sat ** w_sat + exp ** w_exp  # batched operation per image
    
    # normalise weights
    weights = weights/weights.sum(axix=0, keepdim=True)
    
    # TODO generate laplacian pyramids
    laplacian_pyramid = None
    
    # TODO fusion
    
    fused_image = None
    return fused_image


def compute_contrast(burst):
    gray_burst = burst.mean(dim=-1) # Average color channels
    
    k_laplacian = th.Tensor([[0, 1, 0],
                             [1, -4, 1],
                             [0, 1, 0]]).unsqueeze(0).unsqueeze(0)
    
    contrast = th.abs(
        th.nn.functional.conv2d(gray_burst.unsqueeze(1), k_laplacian)
        ).squeeze(1)
    
    return contrast, gray_burst

def compute_saturation(burst, gray_burst):
    
    sat = th.sqrt(
        th.mean(
            (burst - gray_burst.unsqueeze(-1))**2 
            ,dim=-1
            )
        )
    return sat

def compute_well_exposedness(burst):
    sigma = 0.2
    well_exposedness = th.exp(
        -th.sum(
            (burst-0.5)**2
            ,dim=-1
            )/(2*sigma)
        )
    
    return well_exposedness
    
    


exposure_fusion(burst, 1, 1, 1)



