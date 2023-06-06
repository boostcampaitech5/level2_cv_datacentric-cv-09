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
from dataset import SceneTextDataset, CustomDataset
from model import EAST
from utils import seed_everything, get_gpu
import wandb
from deteval import calc_deteval_metrics
import numpy as np
from detect import get_bboxes

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
    parser.add_argument('--val_batch_size', type=int, default=4)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--max_epoch', type=int, default=100)
    parser.add_argument('--save_interval', type=int, default=5)
    parser.add_argument('--ignore_tags', type=list, default=['masked', 'excluded-region', 'maintable', 'stamp'])
    parser.add_argument('--weight', type=str,default='/opt/ml/input/code/trained_models/best.pth')

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

    val_dataset = CustomDataset()

    
    train_dataset = EASTDataset(train_dataset)

    num_train_batches = math.ceil(len(train_dataset) / args.batch_size)
    num_val_batches = math.ceil(len(val_dataset) / args.val_batch_size)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = EAST()
    model.load_state_dict(torch.load(args.weight, map_location='cpu'))
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    # scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[args.max_epoch // 2], gamma=0.1)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer,T_max= 10,eta_min=args.learning_rate*0.1)
    
    best_scroe = 0
    precision = 0
    recall = 0
    hmean = 0
    val_loss = 0
    val_average = {
                'Cls loss':0,
                'Angle loss': 0,
                'IoU loss': 0
            }
        
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
                for key in val_dict:
                    train_average[key] += val_dict[key]
        
        for key in train_average:
            train_average[key] = round(train_average[key]/num_train_batches,4)
        
        scheduler.step()
        epoch_loss /= num_train_batches
        print('Train Mean loss: {:.4f} | Elapsed time: {}'.format(
            epoch_loss , timedelta(seconds=time.time() - epoch_start)))
        
        # to evaluate
        if (epoch+1) % args.save_interval == 0:
            model.eval()
            val_loss, val_start = 0, time.time()
            val_average = {
                'Cls loss':0,
                'Angle loss': 0,
                'IoU loss': 0
            }
        
            predict_box = {}
            gt_box = {}
            transcriptions_dict = {}
            with torch.no_grad():
                with tqdm(total=num_val_batches) as pbar:
                    batch_data = {
                        "gt_score_maps" : [],
                        "gt_geo_maps" : [],
                        "gt_roi_masks" : [],
                    }
                    batch, orig_sizes = [], []
                    temp = 0
                    
                    for i,(input_img, word_bboxes, roi_mask, gt_score_map, gt_geo_map, ori_size, transcriptions) in enumerate(iter(val_dataset)):
                        # 배치만큼의 이미지를 넣는다
                        orig_sizes.append(ori_size)
                        batch.append(input_img)
                        
                        gt_box[i]=word_bboxes
                        transcriptions_dict[i] = transcriptions
                        
                        batch_data['gt_score_maps'].append(gt_score_map)
                        batch_data['gt_geo_maps'].append(gt_geo_map)
                        batch_data['gt_roi_masks'].append(roi_mask)
                        
                        if len(batch) == args.val_batch_size:
                            pbar.set_description('[Epoch {}]'.format(epoch + 1))

                            batch = torch.stack(batch, dim=0).to(device)
                            gt_score_maps = torch.stack(batch_data['gt_score_maps'],dim=0).to(device)
                            gt_geo_maps = torch.stack(batch_data['gt_geo_maps'],dim=0).to(device)
                            gt_roi_masks = torch.stack(batch_data['gt_roi_masks'],dim=0).to(device)
                            
                            # loss를 계산하기 위한 gt_geo_map, gt_score_map 생성
                            score_maps, geo_maps = model(batch)
                            
                            # # calculrate loss
                            loss, values_dict = model.criterion(gt_score_maps, score_maps, gt_geo_maps, geo_maps, gt_roi_masks)
                            val_loss +=loss.item()
                            extra_info = dict(**values_dict, score_map=score_maps, geo_map=geo_maps)
                            
                            # bbox output
                            score_maps, geo_maps = score_maps.cpu().numpy(), geo_maps.cpu().numpy()

                            for score_map, geo_map, orig_size in zip(score_maps, geo_maps, orig_sizes):
                                map_margin = int(abs(orig_size[0] - orig_size[1]) * 0.5 * args.image_size / max(orig_size))

                                if orig_size[0] == orig_size[1]:
                                    score_map, geo_map = score_map, geo_map
                                elif orig_size[0] > orig_size[1]:
                                    score_map, geo_map = score_map[:, :, :-map_margin], geo_map[:, :, :-map_margin]
                                else:
                                    score_map, geo_map = score_map[:, :-map_margin, :], geo_map[:, :-map_margin, :]

                                bboxes = get_bboxes(score_map, geo_map)
                                if bboxes is None:
                                    bboxes = np.zeros((0, 4, 2), dtype=np.float32)
                                else:
                                    bboxes = bboxes[:, :8].reshape(-1, 4, 2)
                                    bboxes *= max(orig_size) / args.image_size

                                predict_box[temp]=bboxes
                                temp +=1
                            
                            batch_data = {
                                "gt_score_maps" : [],
                                "gt_geo_maps" : [],
                                "gt_roi_masks" : [],
                            }
                            batch, orig_sizes = [], []
                            
                            val_dict = {
                                'Cls loss': extra_info['cls_loss'], 'Angle loss': extra_info['angle_loss'],
                                'IoU loss': extra_info['iou_loss']
                            }
                            pbar.update(1)
                            pbar.set_postfix(val_dict)
                        
                            for key in val_dict:
                                val_average[key] += val_dict[key]
        
            for key in val_average:
                val_average[key] = round(val_average[key]/num_val_batches,4)
            
            # calculrate validation f1 score
            metric = calc_deteval_metrics(predict_box,gt_box,transcriptions_dict)
            precision = metric['total']['precision']
            recall = metric['total']['recall']
            hmean = metric['total']['hmean']

            print('Eval Mean loss: {:.4f} | Elapsed time: {}'.format(
                val_loss/num_val_batches, timedelta(seconds=time.time() - val_start)))
            print(f'Eval score precision : {precision:.4f} | recall: {recall:.4f} | heman: {hmean:.4f}')
        
            # to save
            if not osp.exists(args.model_dir):
                os.makedirs(args.model_dir)
                
            ckpt_fpath = osp.join(args.model_dir, f'{epoch+1}_{hmean:.3f}.pth')
            torch.save(model.state_dict(), ckpt_fpath)
            print(f"save latest model {ckpt_fpath}")
                
            if best_scroe < hmean:
                ckpt_fpath = osp.join(args.model_dir, 'best.pth')
                torch.save(model.state_dict(), ckpt_fpath)
                best_scroe = hmean
                print(f"save best model {ckpt_fpath}")
                
        if args.wandb:
            metric_info = {

                'train/loss' : epoch_loss / num_train_batches,
                'val/loss': val_loss/num_val_batches,
                'metric/precision': precision,
                'metric/recall': recall,
                'metric/hmean': hmean,
                'lr/lr': optimizer.param_groups[0]['lr']
            }
            for key in train_average:
                metric_info[f"train/{key}"] = train_average[key]
            for key in val_average:
                metric_info[f"val/{key}"] = val_average[key]
                
            wandb.log(metric_info, step= epoch)


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

