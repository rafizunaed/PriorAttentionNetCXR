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

from argparse import ArgumentParser

import torch
from torch.utils.data import DataLoader

from torch import nn
import time

import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

from datasets import NIH_IMG_LEVEL_DS, get_valid_transforms, collate_fn_img_level_ds
from models import ThoraXPriorNet
from configs import all_configs, DEVICE, NIH_DATASET_ROOT_DIR, NIH_SPLIT_INFO_DICT_DIR, NIH_MASK_ROOT_DIR, NIH_ABNORMALITY_MASK_DIR
from trainer_callbacks import set_random_state

def get_args():
    parser = ArgumentParser(description='test')
    parser.add_argument('--run_config', type=str, default='exp_512_r48')
    parser.add_argument('--gpu_ids', type=str, default='0,1,2,3')
    parser.add_argument('--n_workers', type=int, default=16)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--image_resize_dim', type=int, default=586)
    parser.add_argument('--image_crop_dim', type=int, default=512)
    args = parser.parse_args()
    return args

def main():
    args = get_args()
    
    # set gpu ids
    str_ids = args.gpu_ids.split(',')
    args.gpu_ids = []
    for str_id in str_ids:
        gpu_id = int(str_id)
        if gpu_id >= 0:
            args.gpu_ids.append(gpu_id)
            
    # get configs
    run_config = args.run_config  
    configs = all_configs[run_config]
    weight_saving_path = configs['weight_saving_path']
    
    # get dataloader
    print('Loading dataloader!')
    test_dataset = NIH_IMG_LEVEL_DS(
                        NIH_DATASET_ROOT_DIR,
                        NIH_MASK_ROOT_DIR,
                        NIH_SPLIT_INFO_DICT_DIR,
                        'test',
                        get_valid_transforms(args.image_resize_dim, args.image_crop_dim),
                        configs['use_eight_class'],
                        NIH_ABNORMALITY_MASK_DIR,
                        )
    test_loader = DataLoader(
                        test_dataset, 
                        batch_size=args.batch_size, 
                        shuffle=False, 
                        num_workers=args.n_workers,
                        drop_last=False,
                        collate_fn=collate_fn_img_level_ds,
                        )  
    
    set_random_state(args.seed)
     
    all_targets = []
    all_probabilities = []
        
    num_classes = 8 if configs['use_eight_class'] else 15
    print('Loading model!')
    model = ThoraXPriorNet(num_classes, configs['model_type'], configs['mask_resize_dim'])
        
    checkpoint = torch.load(weight_saving_path+'/checkpoint_best_auc.pth')
    print('loss score: {:.4f}'.format(checkpoint['val_loss']))
    print('auc score: {:.4f}'.format(checkpoint['val_auc']))
    model.load_state_dict(checkpoint['Model_state_dict'])
    model = model.to(DEVICE)
    if len(args.gpu_ids) > 1:
        model = nn.DataParallel(model, device_ids=args.gpu_ids)
    model.eval()                  
    del checkpoint

    torch.set_grad_enabled(False)
    with torch.no_grad():
        for itera_no, data in tqdm(enumerate(test_loader), total=len(test_loader)):
            images = data['image'].to(DEVICE) 
            targets = data['target'].to(DEVICE)
            lung_masks = data['lung_mask'].to(DEVICE)
            abnormality_masks = data['abnormality_masks'].to(DEVICE)
            
            with torch.cuda.amp.autocast():
                out = model(images, abnormality_masks, lung_masks)
                
            all_targets.append(targets.cpu().data.numpy())              
            y_prob = out['logits'].cpu().detach().clone().float().sigmoid()
            all_probabilities.append(y_prob.numpy())
            
    num_classes = 8 if configs['use_eight_class'] else 14        
    all_targets = np.concatenate(all_targets)[:,:num_classes]
    all_probabilities = np.concatenate(all_probabilities)[:,:num_classes]
    
    auc = roc_auc_score(all_targets, all_probabilities)
    print(f'auc score: {auc*100}')
    time.sleep(1)
    
    all_clswise_auc = roc_auc_score(all_targets, all_probabilities, average=None)
    all_clswise_auc = 100 * all_clswise_auc
    print(all_clswise_auc)
    
    return all_probabilities, all_targets
    
if __name__ == '__main__':
    all_probabilities, all_targets = main()