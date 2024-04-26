# MIT License
#
# Copyright (c) 2024 Mohammad Zunaed, mHealth Lab, BUET
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import torch

DEVICE = torch.device("cuda:0")

NIH_DATASET_ROOT_DIR = './datasets/NIH_CXR/NIH_CXR_512_align/'
NIH_MASK_ROOT_DIR = './datasets/NIH_CXR/NIH_CXR_256_align_lung_masks_convex_hull/'

NIH_SPLIT_INFO_DICT_DIR = './label_info/NIH_CXR_img_level_split_info_dict_oct_2023_split_ratio_no_bbox_imgs.npy'
NIH_BBOX_INFO_DICT_DIR = './label_info/NIH_CXR_bbox_info_x1y1x2y2_256_align.npy'
NIH_ABNORMALITY_MASK_DIR = './label_info/all_abnormality_masks_nih_cxr_align.npy'

all_configs = {
    'exp_512_r48':{
        'weight_saving_path': './weights/exp_512_r48/',
        'epochs': 50,
        'checkpoint_path': None,
        'model_type': 'densenet121',
        'use_eight_class': False,
        'mask_resize_dim': 48,
        },
    }