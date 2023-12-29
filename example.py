# -*- coding: utf-8 -*-
"""
Created on Thu Dec 21 17:18:51 2023

@author: jamyl
"""

import os
import glob
import torch as th
import cv2
import matplotlib. pyplot as plt
from pathlib import Path

from exposure_fusion import exposure_fusion

# Read image burst
burst_path = Path("data/mask")

im_path_list = glob.glob(os.path.join(burst_path.as_posix(), '*.jpg'))
im_path_list += glob.glob(os.path.join(burst_path.as_posix(), '*.png'))
assert len(im_path_list) != 0, 'At least one .jpg or .png file must be present in the burst folder.'

burst = []
for im_path in im_path_list:
    burst.append(
        cv2.cvtColor(
            cv2.imread(im_path, cv2.IMREAD_UNCHANGED),
            cv2.COLOR_BGR2RGB)
        ) # flag to keep the same bit depth as original

# Normalise the burst between 0 and 1
burst = th.Tensor(burst)/255
burst = burst.movedim(-1, 1) # batch, channel, H, W format


out = exposure_fusion(burst)

out = out.clamp(0, 1)
out = out.movedim(0, -1).cpu().numpy() # [H, W, C] format for matplotlib
plt.imsave("out/result.png", out, vmin=0, vmax=1)


