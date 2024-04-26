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

import cv2
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from configs import NIH_DATASET_ROOT_DIR, NIH_SPLIT_INFO_DICT_DIR, NIH_MASK_ROOT_DIR, NIH_BBOX_INFO_DICT_DIR, NIH_ABNORMALITY_MASK_DIR

IMG_LEVEL_CLASS_NAMES = [ 'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia',
                'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia', 'No Finding']

def get_train_transforms(resize, crop):
    return A.Compose([
            A.Resize(width=resize, height=resize, p=1.0),
            A.RandomCrop(width=crop, height=crop, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                max_pixel_value=255,
                p=1.0,
            ),
            ToTensorV2(always_apply=True, p=1.0),
        ], p=1.0)

def get_valid_transforms(resize, crop):
    return A.Compose([
            A.Resize(width=resize, height=resize, p=1.0),
            A.CenterCrop(width=crop, height=crop, p=1.0),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                max_pixel_value=255,
                p=1.0,
            ),
            ToTensorV2(always_apply=True, p=1.0),
        ], p=1.0)

def get_test_transforms_tta(resize):
    return A.Compose([
            A.Resize(width=resize, height=resize, p=1.0),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                max_pixel_value=255,
                p=1.0,
            ),
            ToTensorV2(always_apply=True, p=1.0),
        ], p=1.0)

def get_bbox_test_transforms(resize, crop):
    return A.Compose([
            A.Resize(width=resize, height=resize, p=1.0),
            A.CenterCrop(width=crop, height=crop, p=1.0),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                max_pixel_value=255,
                p=1.0,
            ),
            ToTensorV2(always_apply=True, p=1.0),
        ], p=1.0, bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))

class NIH_IMG_LEVEL_DS(Dataset):
    def __init__(self, data_root_dir, mask_root_dir, split_info_dict_dir, split_type, transform, eight_cls_method, abnormality_mask_dir):
        assert split_type in ['train', 'val', 'test']
        self.data_root_dir = data_root_dir
        self.mask_root_dir = mask_root_dir
        split_info_dict = np.load(split_info_dict_dir, allow_pickle=True).item()
        self.fnames = split_info_dict[f'{split_type}_fnames']
        self.labels = split_info_dict[f'{split_type}_labels']
        self.transform = transform
        self.eight_cls_method = eight_cls_method
        self.split_type = split_type
        self.abnormality_masks = np.load(abnormality_mask_dir)
        
        if not self.eight_cls_method:
            self.abnormality_masks = np.concatenate([self.abnormality_masks, np.ones([6, self.abnormality_masks.shape[-2], self.abnormality_masks.shape[-1]])])
            self.abnormality_masks = np.concatenate([self.abnormality_masks, np.ones([1, self.abnormality_masks.shape[-2], self.abnormality_masks.shape[-1]])])

        self.abnormality_masks_list = []
        for i in range(self.abnormality_masks.shape[0]):
            self.abnormality_masks_list.append(self.abnormality_masks[i].copy())
        
    def __len__(self):
        return len(self.fnames)
    
    def __getitem__(self, index):
        # read image
        fname = self.fnames[index]
        image = Image.open(self.data_root_dir+fname).convert('RGB')
        image = np.array(image)
        
        # read lung mask
        mask = Image.open(self.mask_root_dir+fname)
        mask = mask.resize((image.shape[0], image.shape[1]))
        mask = np.array(mask)
        lung_mask = mask.copy()/255
        
        # transform image and mask
        masks = [lung_mask]
        masks += self.abnormality_masks_list.copy()
        transformed = self.transform(image=image, masks=masks)
        transformed_image = transformed['image']
        transformed_masks = transformed['masks']

        # mask second round transform
        transformed_lung_mask = transformed_masks[0]
        transformed_lung_mask = transformed_lung_mask.float().unsqueeze(0)       
        transformed_abnormality_masks = torch.stack(transformed_masks[1:]).float()
        
        # read label
        if self.eight_cls_method:
            label = self.labels[index][:8]
        else:
            label = self.labels[index] 
        label = torch.FloatTensor(label)
            
        return {
            'image': transformed_image, 
            'target': label,
            'lung_mask': transformed_lung_mask,
            'abnormality_masks': transformed_abnormality_masks,
            }

class NIH_IMG_LEVEL_DS_TTA(Dataset):
    def __init__(self, data_root_dir, mask_root_dir, split_info_dict_dir, split_type, resize_size, crop_size, eight_cls_method, abnormality_mask_dir):
        assert split_type in ['train', 'val', 'test']
        self.data_root_dir = data_root_dir
        self.mask_root_dir = mask_root_dir
        split_info_dict = np.load(split_info_dict_dir, allow_pickle=True).item()
        self.fnames = split_info_dict[f'{split_type}_fnames']
        self.labels = split_info_dict[f'{split_type}_labels']
        self.transform = get_test_transforms_tta(resize_size)
        self.resize_size = resize_size
        self.crop_size = crop_size
        self.eight_cls_method = eight_cls_method
        self.split_type = split_type
        self.abnormality_masks = np.load(abnormality_mask_dir)
        
        if not self.eight_cls_method:
            self.abnormality_masks = np.concatenate([self.abnormality_masks, np.ones([6, self.abnormality_masks.shape[-2], self.abnormality_masks.shape[-1]])])
            self.abnormality_masks = np.concatenate([self.abnormality_masks, np.ones([1, self.abnormality_masks.shape[-2], self.abnormality_masks.shape[-1]])])

        self.abnormality_masks_list = []
        for i in range(self.abnormality_masks.shape[0]):
            self.abnormality_masks_list.append(self.abnormality_masks[i].copy())
        
    def __len__(self):
        return len(self.fnames)
    
    def __getitem__(self, index):
        # read image
        fname = self.fnames[index]
        image = Image.open(self.data_root_dir+fname).convert('RGB')
        image = np.array(image)
        
        # read lung mask
        mask = Image.open(self.mask_root_dir+fname)
        mask = mask.resize((image.shape[0], image.shape[1]))
        mask = np.array(mask)
        lung_mask = mask.copy()/255
        
        # transform image and mask
        masks = [lung_mask]
        masks += self.abnormality_masks_list.copy()
        transformed = self.transform(image=image, masks=masks)
        transformed_image = transformed['image']
        transformed_masks = transformed['masks']

        # mask second round transform
        transformed_lung_mask = transformed_masks[0]
        transformed_lung_mask = transformed_lung_mask.float().unsqueeze(0)       
        transformed_abnormality_masks = torch.stack(transformed_masks[1:]).float()
        
        # read label
        if self.eight_cls_method:
            label = self.labels[index][:8]
        else:
            label = self.labels[index] 
        label = torch.FloatTensor(label)
            
        
        image_1 = transformed_image[:, 0:self.crop_size, 0:self.crop_size]
        image_2 = transformed_image[:, -self.crop_size:, 0:self.crop_size]
        image_3 = transformed_image[:, 0:self.crop_size, -self.crop_size:]
        image_4 = transformed_image[:, -self.crop_size:, -self.crop_size:]
        crop_idx = int((self.resize_size-self.crop_size)/2)
        image_5 = transformed_image[:, crop_idx:-crop_idx, crop_idx:-crop_idx]
        # print(transformed_image.shape, image_1.shape, image_2.shape, image_3.shape, image_4.shape, image_5.shape)
        all_transformed_images = torch.stack([image_1, image_2, image_3, image_4, image_5])
        
        
        lung_mask_1 = transformed_lung_mask[:, 0:self.crop_size, 0:self.crop_size]
        lung_mask_2 = transformed_lung_mask[:, -self.crop_size:, 0:self.crop_size]
        lung_mask_3 = transformed_lung_mask[:, 0:self.crop_size, -self.crop_size:]
        lung_mask_4 = transformed_lung_mask[:, -self.crop_size:, -self.crop_size:]
        crop_idx = int((self.resize_size-self.crop_size)/2)
        lung_mask_5 = transformed_lung_mask[:, crop_idx:-crop_idx, crop_idx:-crop_idx]
        # print(transformed_image.shape, image_1.shape, image_2.shape, image_3.shape, image_4.shape, image_5.shape)
        all_transformed_lung_mask = torch.stack([lung_mask_1, lung_mask_2, lung_mask_3, lung_mask_4, lung_mask_5])
        
        
        abnormality_mask_1 = transformed_abnormality_masks[:, 0:self.crop_size, 0:self.crop_size]
        abnormality_mask_2 = transformed_abnormality_masks[:, -self.crop_size:, 0:self.crop_size]
        abnormality_mask_3 = transformed_abnormality_masks[:, 0:self.crop_size, -self.crop_size:]
        abnormality_mask_4 = transformed_abnormality_masks[:, -self.crop_size:, -self.crop_size:]
        crop_idx = int((self.resize_size-self.crop_size)/2)
        abnormality_mask_5 = transformed_abnormality_masks[:, crop_idx:-crop_idx, crop_idx:-crop_idx]
        # print(transformed_image.shape, image_1.shape, image_2.shape, image_3.shape, image_4.shape, image_5.shape)
        all_transformed_abnormality_masks = torch.stack([abnormality_mask_1, abnormality_mask_2, abnormality_mask_3, abnormality_mask_4, abnormality_mask_5])
        
        
        return {
            'image': all_transformed_images, 
            'target': label,
            'lung_mask': all_transformed_lung_mask,
            'abnormality_masks': all_transformed_abnormality_masks,
            }

class NIH_BBox_LEVEL_DS(Dataset):
    def __init__(self, data_root_dir, mask_root_dir, bbox_info_dict_dir, transform, eight_cls_method, abnormality_mask_dir):
        self.data_root_dir = data_root_dir
        self.mask_root_dir = mask_root_dir
        self.transform = transform
        self.eight_cls_method = eight_cls_method
        
        # read abnormality masks
        self.abnormality_masks = np.load(abnormality_mask_dir)
        if not self.eight_cls_method :
            self.abnormality_masks = np.concatenate([self.abnormality_masks, np.ones([6, self.abnormality_masks.shape[-2], self.abnormality_masks.shape[-1]])])
            self.abnormality_masks = np.concatenate([self.abnormality_masks, np.ones([1, self.abnormality_masks.shape[-2], self.abnormality_masks.shape[-1]])])
        self.abnormality_masks_list = []
        for i in range(self.abnormality_masks.shape[0]):
            self.abnormality_masks_list.append(self.abnormality_masks[i].copy())
        
        # extra bit
        self.bbox_info = np.load(bbox_info_dict_dir, allow_pickle=True).item()
        self.bbox_img_size = self.bbox_info['image_size']
        self.bbox_info = self.bbox_info['bbox_info']
        self.fnames = np.array(list(self.bbox_info.keys()))
        
    def __len__(self):
        return len(self.fnames)
    
    def __getitem__(self, index):
        # read image
        fname = self.fnames[index]
        image = Image.open(self.data_root_dir+fname).convert('RGB')
        image = np.array(image)
            
        # read lung mask
        mask = Image.open(self.mask_root_dir+fname)
        mask = mask.resize((image.shape[0], image.shape[1]))
        mask = np.array(mask)
        lung_mask = mask.copy()/255
        
        # process class labels and bbox
        class_labels = []
        for x in self.bbox_info[fname]['pathology']:
            class_labels.append(x)
            
        bboxes = []
        for bbox in self.bbox_info[fname]['bbox']:
            bbox = np.array(bbox)
            bbox = (bbox*image.shape[0])/self.bbox_img_size
            # bbox = bbox.astype(np.int64)
            bboxes.append(bbox)
        
        # transform image and mask
        masks = [lung_mask]
        masks += self.abnormality_masks_list.copy()
        transformed = self.transform(image=image, masks=masks, bboxes=bboxes, class_labels=class_labels)
        transformed_image = transformed['image']
        transformed_masks = transformed['masks']
        transformed_bboxes = transformed['bboxes']
        transformed_class_labels = transformed['class_labels']
        
        # masks second round transform
        transformed_lung_mask = transformed_masks[0]
        transformed_lung_mask = transformed_lung_mask.float().unsqueeze(0)
        transformed_abnormality_masks = torch.stack(transformed_masks[1:]).float()
        
        return {
            'image': transformed_image, 
            'lung_mask': transformed_lung_mask,
            'abnormality_masks': transformed_abnormality_masks,
            'bboxes': transformed_bboxes,
            'cls_labels': transformed_class_labels,
            'image_id': fname,
            }
    
def collate_fn_img_level_ds(batch):
    x = batch[0]
    keys = x.keys()
    out = {}
    # declare key
    for key in keys:
        out.update({key:[]})
    # append values
    for i in range(len(batch)):
        for key in keys:
            out[key].append(batch[i][key])
    # stack values
    for key in keys:
        out[key] = torch.stack(out[key])
    
    return out    