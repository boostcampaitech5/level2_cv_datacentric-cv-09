import os
import os.path as osp
import time
import math
from datetime import timedelta
from argparse import ArgumentParser
import json

import torch
from torch import cuda
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from tqdm import tqdm

from east_dataset import EASTDataset
from dataset import SceneTextDataset, split_list
from model import EAST
from utils import seed_everything, get_gpu
import wandb

def parse_args():
    parser = ArgumentParser()

    # Conventional args
    parser.add_argument('--data_dir', type=str,
                        default=os.environ.get('SM_CHANNEL_TRAIN', '../data/medical'))
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR',
                                                                        'trained_models'))

    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', default='cuda' if cuda.is_available() else 'cpu')
    parser.add_argument('--num_workers', type=int, default=8)

    # for training
    parser.add_argument('--image_size', type=int, default=2048)
    parser.add_argument('--input_size', type=int, default=1024)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--max_epoch', type=int, default=50)
    parser.add_argument('--save_interval', type=int, default=5)
    parser.add_argument('--ignore_tags', type=list, default=['masked', 'excluded-region', 'maintable', 'stamp'])

    # wandb
    parser.add_argument('--wandb',action='store_true')
    parser.add_argument('--project',type=str, default='ocr_baseline')
    parser.add_argument('--name',type=str, default='base')
    args = parser.parse_args()

    if args.input_size % 32 != 0:
        raise ValueError('`input_size` must be a multiple of 32')

    return args


def do_training(args):
    
    seed_everything(args.seed)
    
    train_dataset = SceneTextDataset(
        args.data_dir,
        image_size=args.image_size,
        crop_size=args.input_size,
        ignore_tags=args.ignore_tags
    )
    
    val_dataset = SceneTextDataset(
        args.data_dir,
        split='val',
        image_size=args.image_size,
        crop_size=args.input_size,
        ignore_tags=args.ignore_tags
    )
    
    train_dataset = EASTDataset(train_dataset)
    val_dataset = EASTDataset(val_dataset)

    num_train_batches = math.ceil(len(train_dataset) / args.batch_size)
    num_val_batches = math.ceil(len(val_dataset) / args.batch_size)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = EAST()
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[args.max_epoch // 2], gamma=0.1)

    best_loss = 10000
    
    for epoch in range(args.max_epoch):
        # to train
        model.train()
        epoch_loss, epoch_start = 0, time.time()
        train_average = {
            'Cls loss':0,
            'Angle loss': 0,
            'IoU loss': 0
        }
        with tqdm(total=num_train_batches) as pbar:
            for img, gt_score_map, gt_geo_map, roi_mask in train_loader:
                pbar.set_description('[Epoch {}]'.format(epoch + 1))

                loss, extra_info = model.train_step(img, gt_score_map, gt_geo_map, roi_mask)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                loss_val = loss.item()
                epoch_loss += loss_val

                pbar.update(1)
                val_dict = {
                    'Cls loss': extra_info['cls_loss'], 'Angle loss': extra_info['angle_loss'],
                    'IoU loss': extra_info['iou_loss']
                }
                pbar.set_postfix(val_dict)
                for key in val_dict.keys():
                    train_average[key] += val_dict[key]
        
        for key in train_average.keys():
            train_average[key] = round(train_average[key]/num_train_batches,4)
        
        scheduler.step()
        epoch_loss /= num_train_batches
        print('Train Mean loss: {:.4f} | Elapsed time: {}'.format(
            epoch_loss , timedelta(seconds=time.time() - epoch_start)))
        
        # to evaluate
        model.eval()
        val_loss, val_start = 0, time.time()
        val_average = {
            'Cls loss':0,
            'Angle loss': 0,
            'IoU loss': 0
        }
        
        with torch.no_grad():
            with tqdm(total=num_val_batches) as pbar:
                for img, gt_score_map, gt_geo_map, roi_mask in val_loader:
                    # calculrate validation loss
                    pbar.set_description('[Epoch {}]'.format(epoch + 1))

                    loss, extra_info = model.train_step(img, gt_score_map, gt_geo_map, roi_mask)

                    loss_val = loss.item()
                    val_loss += loss_val

                    pbar.update(1)
                    val_dict = {
                        'Cls loss': extra_info['cls_loss'], 'Angle loss': extra_info['angle_loss'],
                        'IoU loss': extra_info['iou_loss']
                    }
                    pbar.set_postfix(val_dict)
                    
                    # calculrate validation f1 score
                    
                for key in val_dict.keys():
                    val_average[key] += val_dict[key]
        
        for key in val_average.keys():
            val_average[key] = round(val_average[key]/num_val_batches,4)

        val_loss = val_loss/num_val_batches
        print('Eval Mean loss: {:.4f} | Elapsed time: {}'.format(
            val_loss, timedelta(seconds=time.time() - val_start)))
        
        if args.wandb:
            metric_info = {
                'train/loss' : epoch_loss,
                'val/loss' : val_loss,
                'lr/lr' :optimizer.param_groups[0]['lr']
            }
            for key in train_average.keys():
                metric_info[f"train/{key}"] = train_average[key]
            for key in val_average.keys():
                metric_info[f"val/{key}"] = val_average[key]
                
            wandb.log(metric_info, step= epoch)

            
        # to save
        if not osp.exists(args.model_dir):
            os.makedirs(args.model_dir)
            
        if (epoch + 1) % args.save_interval == 0:

            ckpt_fpath = osp.join(args.model_dir, 'latest.pth')
            torch.save(model.state_dict(), ckpt_fpath)
            
        if best_loss > (val_loss/num_val_batches):
                ckpt_fpath = osp.join(args.model_dir, 'best.pth')
                torch.save(model.state_dict(), ckpt_fpath)
                best_loss = val_loss
                print(f"save best model {ckpt_fpath}")


def main(args):
    do_training(args)


if __name__ == '__main__':
    args = parse_args()
    if args.wandb:
        wandb.init(
            entity = 'boost_cv_09',
            project = args.project,
            name = args.name,
            config = args
        )
    main(args)