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
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.cuda.amp import autocast
import numpy as np

class MaskedAttention(nn.Module):
    def __init__(self, in_channels:int, r:float=1.0):
        super().__init__()
        self.in_channels = in_channels
        self.r = r
        self.conv_list = nn.ModuleList()
        for i in range(4):
            self.conv_list.append(nn.Sequential(
                nn.Conv2d(self.in_channels, int(self.in_channels * r), kernel_size=1),
                nn.LeakyReLU(negative_slope=0.2, inplace=True)
            ))
        self.final_block = nn.Sequential(
            nn.Conv2d(int(r * self.in_channels), self.in_channels, kernel_size=1),
            nn.Sigmoid(),
            nn.Dropout(p=0.1)
        )
        self.bn3 = nn.BatchNorm2d(self.in_channels)

    @autocast()
    def forward(self, feature_map, mask):
        N, C, H, W = feature_map.shape
        mask = F.interpolate(mask, size=(H, W), mode="bilinear", align_corners=False) # (N, 1, H, W) # bilinear
        masked_feature_map = feature_map * mask # Spatial Attention (N, C, H, W)
        avg_pooled_fm = F.adaptive_avg_pool2d(feature_map, output_size=1) # (N, C, 1, 1)
        max_pooled_fm = F.adaptive_max_pool2d(feature_map, output_size=1) # (N, C)
        avg_pooled_mfm = F.adaptive_avg_pool2d(masked_feature_map, output_size=1) # (N, C, 1, 1)
        max_pooled_mfm = F.adaptive_max_pool2d(masked_feature_map, output_size=1) # (N, C)
        channel_weight = self.final_block(self.conv_list[0](avg_pooled_fm) + self.conv_list[1](max_pooled_fm) +\
                         self.conv_list[2](max_pooled_mfm) + self.conv_list[3](avg_pooled_mfm))
        channel_weight =  channel_weight.view(N, C, 1, 1)
        feature_map = self.bn3(channel_weight * masked_feature_map + (1 - channel_weight) * feature_map)
        return feature_map
    
class ThoraXPriorNet(nn.Module):
    def __init__(self, num_classes:int, model_type: str='densenet121', mask_resize_dim=None):
        super().__init__()
        assert model_type in ['resnet50', 'densenet121']
        self.model_type = model_type
        self.num_classes = num_classes
        self.pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.mask_resize_dim = mask_resize_dim
        
        if model_type == 'densenet121':
            self.backbone = torchvision.models.densenet121(weights=torchvision.models.DenseNet121_Weights.IMAGENET1K_V1).features
            self.classifiers = nn.ModuleList()
            self.lung_attention_module = MaskedAttention(1024)
            self.attention_modules = nn.ModuleList()
            for i in range(num_classes):
                self.attention_modules.append(MaskedAttention(1024))
                self.classifiers.append(nn.Conv2d(in_channels=2048, out_channels=1, kernel_size=1))
                nn.init.xavier_normal_(self.classifiers[i].weight)
                
        elif model_type == 'resnet50':
            backbone = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1)
            modules = list(backbone.children())
            self.backbone = nn.Sequential(*modules[:-2])       
            self.classifiers = nn.ModuleList()
            self.lung_attention_module = MaskedAttention(1024*2)
            self.attention_modules = nn.ModuleList()
            for i in range(num_classes):
                self.attention_modules.append(MaskedAttention(1024*2))
                self.classifiers.append(nn.Conv2d(in_channels=2048*2, out_channels=1, kernel_size=1))
                nn.init.xavier_normal_(self.classifiers[i].weight)
    
    def activations_hook(self, grad):
        self.gradients = grad.detach().clone().cpu().numpy()
        np.save('./temp_grad_holder/temp_grad.npy', self.gradients)
    
    @autocast()
    def forward(self, x, abnorm_mask, lung_mask, grad_idx=None):
        grad_attention_map=None
        
        logit_maps = []
        logits = []

        # backbone
        feature_map = self.backbone(x)
        if self.model_type == 'densenet121':
            feature_map = F.relu(feature_map)  
       
        if self.mask_resize_dim is not None:
            feature_map = F.interpolate(feature_map, size=(self.mask_resize_dim, self.mask_resize_dim), mode="bilinear", align_corners=False)
       
        lung_attention_map = self.lung_attention_module(feature_map, lung_mask) # (N, 1024, H, W)
        for i in range(self.num_classes):
            attention_map = self.attention_modules[i](feature_map, abnorm_mask[:, i, :, :].unsqueeze(dim=1)) # (N, 1024, H, W)
            attention_map = torch.cat((attention_map, lung_attention_map), dim=1)  # (N, 2048, H, W)
            logit_map = self.classifiers[i](attention_map) # (N, 1, H, W)
            logit_maps.append(logit_map)
                       
            if i==grad_idx:
                attention_map.register_hook(self.activations_hook)
                grad_attention_map = attention_map
            
            out = self.pool(attention_map)
            out = F.dropout(out, p=0.2, training=self.training)
            out = self.classifiers[i](out)
            logits.append(out.squeeze(dim=-1).squeeze(dim=-1))

        logits = torch.cat(logits, dim=-1)
        logit_maps = torch.cat(logit_maps, dim=1)

        return {
            'logits': logits, 
            'logit_maps': logit_maps,
            'grad_attention_map': grad_attention_map,
            }