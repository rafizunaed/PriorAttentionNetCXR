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
import numpy as np

from trainer import ModelTrainer

from datasets import get_train_transforms, get_valid_transforms, collate_fn_img_level_ds, NIH_IMG_LEVEL_DS
from models import ThoraXPriorNet
from configs import all_configs, NIH_DATASET_ROOT_DIR, NIH_SPLIT_INFO_DICT_DIR, NIH_MASK_ROOT_DIR, NIH_ABNORMALITY_MASK_DIR
from trainer_callbacks import set_random_state, AverageMeter, PrintMeter

def get_args():
    """
    get command line args
    """
    parser = ArgumentParser(description='train')
    parser.add_argument('--run_configs_list', type=str, nargs="*", default=['exp_512_r48'])
    parser.add_argument('--gpu_ids', type=str, default='0,1,2,3')
    parser.add_argument('--n_workers', type=int, default=16)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--image_resize_dim', type=int, default=586)
    parser.add_argument('--image_crop_dim', type=int, default=512)
    parser.add_argument('--do_grad_accum', type=bool, default=True)
    parser.add_argument('--grad_accum_step', type=int, default=8)
    parser.add_argument('--use_ema', type=bool, default=True)
    parser.add_argument('--use_focal_loss', type=bool, default=False)
    parser.add_argument('--focal_loss_alpha', type=float, default=0.25)
    parser.add_argument('--focal_loss_gamma', type=float, default=2)
    args = parser.parse_args()
    return args

def main():
    """
    main function
    """

    args = get_args()

    # set gpu ids
    str_ids = args.gpu_ids.split(',')
    args.gpu_ids = []
    for str_id in str_ids:
        gpu_id = int(str_id)
        if gpu_id >= 0:
            args.gpu_ids.append(gpu_id)

    # check if there are duplicate weight saving paths
    unique_paths = np.unique([ x[1]['weight_saving_path'] for x in all_configs.items() ])
    assert len(all_configs.keys()) == len(unique_paths)

    for config_name in args.run_configs_list:
        configs = all_configs[config_name]   
        set_random_state(args.seed)
        
        print('Loading dataloaders!')
        train_dataset = NIH_IMG_LEVEL_DS(
                            NIH_DATASET_ROOT_DIR,
                            NIH_MASK_ROOT_DIR,
                            NIH_SPLIT_INFO_DICT_DIR,
                            'train',
                            get_train_transforms(args.image_resize_dim, args.image_crop_dim),
                            configs['use_eight_class'],
                            NIH_ABNORMALITY_MASK_DIR,
                            )
        val_dataset = NIH_IMG_LEVEL_DS(
                            NIH_DATASET_ROOT_DIR,
                            NIH_MASK_ROOT_DIR,
                            NIH_SPLIT_INFO_DICT_DIR,
                            'val',
                            get_valid_transforms(args.image_resize_dim, args.image_crop_dim),
                            configs['use_eight_class'],
                            NIH_ABNORMALITY_MASK_DIR,
                            )
        
        train_loader = DataLoader(
                            train_dataset,
                            batch_size=args.batch_size,
                            shuffle=True,
                            num_workers=args.n_workers,
                            drop_last=True,
                            collate_fn=collate_fn_img_level_ds,
                            )
        val_loader = DataLoader(
                            val_dataset,
                            batch_size=args.batch_size,
                            shuffle=False,
                            num_workers=args.n_workers,
                            drop_last=False,
                            collate_fn=collate_fn_img_level_ds,
                            )
        
        
        num_classes = 8 if configs['use_eight_class'] else 15
        print('Loading model!')
        model = ThoraXPriorNet(num_classes, configs['model_type'], configs['mask_resize_dim'])
        
        if configs['checkpoint_path'] is not None:
            checkpoint = torch.load(configs['checkpoint_path'])
            print('loss score: {:.4f}'.format(checkpoint['val_loss']))
            print('auc score: {:.4f}'.format(checkpoint['val_auc']))
            model.load_state_dict(checkpoint['Model_state_dict'], strict=False)
            del checkpoint
        
        trainer_args = {
                'model': model,
                'Loaders': [train_loader, val_loader],
                'metrics': {
                    'loss': AverageMeter,
                    'auc': PrintMeter,
                    },
                'checkpoint_saving_path': configs['weight_saving_path'],
                'lr': args.lr,
                'epochsTorun': configs['epochs'],
                'gpu_ids': args.gpu_ids,
                'do_grad_accum': args.do_grad_accum,
                'grad_accum_step': args.grad_accum_step,
                'use_ema': args.use_ema,
                'use_focal_loss': args.use_focal_loss,
                'focal_loss_alpha': args.focal_loss_alpha,
                'focal_loss_gamma': args.focal_loss_gamma,
                'num_classes': 8 if configs['use_eight_class'] else 14,
                }  
        
        trainer = ModelTrainer(**trainer_args)
        trainer.fit()
            
if __name__ == '__main__':
    main()    