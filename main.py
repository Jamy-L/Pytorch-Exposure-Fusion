# -*- coding: utf-8 -*-
"""
Created on Thu Dec 21 17:18:51 2023

@author: jamyl
"""

import os
import glob
import numpy as np
import torch as th
import cv2
import matplotlib. pyplot as plt
from pathlib import Path

from exposure_fusion import exposure_fusion

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


burst = th.Tensor(burst)/255
burst = burst.movedim(-1, 1) # batch, channel, H, W format


out = exposure_fusion(burst)
_, h, w = out.size()

#%%
def highlight_sat(img, vmin=0, vmax=1):
    sat_max = (img > 1).sum(dim=0, keepdim=True) # 1 channel saturates at least
    sat_min = (img < 0).sum(dim=0, keepdim=True) # 1 channel saturates at least
    out = img * (1 - sat_max) + sat_max * th.ones_like(img) * vmin
    out = out * (1 - sat_min) + sat_min * th.ones_like(img) * vmax
    return out

out2 = highlight_sat(out)

plt.figure("exposure fusion")
plt.imshow(out2.movedim(0, -1),
           vmin=0,
           vmax=1)


for i in range(burst.shape[0]):
    plt.figure(str(i))
    plt.imshow(burst[i, :].movedim(0, -1),
               vmin=0, vmax=1)
    
#%%


