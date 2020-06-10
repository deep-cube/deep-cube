from tqdm import trange, tqdm
from data_loader import *
from pprint import pprint
import data_utils
import torch
import numpy as np
import data_def
import train_utils
import metrics
import sys
from resnet3d import resnet, densenet
import spatial_transforms as spatial


"""
When test:
MAX_EPOCH=2
batch_size = 50
also in data_loader.py
switch comment of get_segmented_train_data and get_segmented_dev_data
"""

opt = {}
opt['model'] = 'resnet'

opt['model_depth'] = 50       # help: model_depth in [10, 18, 34, 50, 101, 152, 200]
opt['conv1_t_size'] = 3       # help: 'Kernel size in t dim of conv1.'
opt['conv1_t_stride'] = 1     # help: 'Stride in t dim of conv1.'
opt['shortcut_type'] = 'B'    # help: 'Shortcut type of resnet (A | B)'
opt['widen_factor'] = 1.0     # help: 'The number of feature maps of resnet is multiplied by this value')
opt['no_max_pool'] = False
opt['n_classes'] = data_def.NUM_CLASS
opt['n_input_channels'] = data_utils.NUM_CHANNEL

opt['batch_size'] = 200
opt['clip_length'] = 7

opt['spatial_augment'] = False
opt['color_jitter'] = True
opt['random_crop'] = True
opt['temporal_shift'] = False

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SAVE_MODEL_NAME = sys.argv[1]
MAX_EPOCH = 500



def generate_model(opt):
    model = None

    if opt['model'] == 'resnet':
        model = resnet.generate_model(opt['model_depth'],
                        n_input_channels=opt['n_input_channels'],
                        conv1_t_size=opt['conv1_t_size'],          
                        conv1_t_stride=opt['conv1_t_stride'],        
                        no_max_pool=opt['no_max_pool'],
                        shortcut_type=opt['shortcut_type'],
                        widen_factor=opt['widen_factor'],
                        n_classes=opt['n_classes'])

    return model


if __name__ == "__main__":
    B, L = opt['batch_size'], opt['clip_length']

    spatial_transform = spatial.get_transform_func(
        random_crop=opt['random_crop'], 
        color_jitter=opt['color_jitter']
    )
    if opt['spatial_augment']:
        train_dataset, train_loader = get_segmented_train_data(B, L, 
            spatial_transform=spatial_transform, temporal_shift=opt['temporal_shift'])
    else:
        train_dataset, train_loader = get_segmented_train_data(B, L, 
            spatial_transform=None, temporal_shift=opt['temporal_shift'])
    dev_dataset, dev_loader = get_segmented_dev_data(B, L)
    H, W = train_dataset.frame_size()
    # print("frame_size",H,W)

    model = generate_model(opt)
    # print(model)

    optimizer = torch.optim.Adam(model.parameters())


    def train_epoch(epoch_idx):
        total_celoss = 0
        total_item = 0
        accuracies = train_utils.AverageMeter()
        accuracies_top3 = train_utils.AverageMeter()
        accuracies_top5 = train_utils.AverageMeter()

        for i, data in enumerate(tqdm(train_loader, desc='TRAIN_CE', leave=False)):
            optimizer.zero_grad()

            x, y = data
            x = x.to(device=DEVICE, dtype=torch.float32)  # (B, C, L, H, W)
            y = y.to(device=DEVICE, dtype=torch.long) # (B,)
            
            # assert x.shape == (B, opt['n_input_channels'], L, H, W), "x shape is wrong to be {}".format(x.shape)
            # assert y.shape == (B,), "the shape of y seems to be {}".format(y.shape)

            total_item += y.shape[0]

            scores = model(x)

            celoss = metrics.cross_entropy_loss(scores, y)
            total_celoss += celoss.item() * y.shape[0]

            acc = train_utils.calculate_acc(scores, y)
            acc3, acc5 = train_utils.calculate_acc_topk(scores, y, topk=(3,5))
            accuracies.update(acc, x.size(0))
            accuracies_top3.update(acc3, x.size(0))
            accuracies_top5.update(acc5, x.size(0))

            loss = celoss
            loss.backward()
            optimizer.step()

        total_loss = total_celoss
        

        results = {
            'train_loss': total_loss / total_item,
            'train_acc': accuracies.avg,
            'train_acc_top3': accuracies_top3.avg,
            'train_acc_top5': accuracies_top5.avg,
        }

        print()
        pprint(results)

        return results


    def dev_epoch(epoch_idx):
        total_celoss = 0
        total_item = 0
        accuracies = train_utils.AverageMeter()
        accuracies_top3 = train_utils.AverageMeter()
        accuracies_top5 = train_utils.AverageMeter()

        for i, data in enumerate(tqdm(dev_loader, desc='DEV', leave=False)):
            x, y = data
            x = x.to(device=DEVICE, dtype=torch.float32)  # (B, L, C, H, W)
            y = y.to(device=DEVICE, dtype=torch.long) # (B,)
            total_item += y.shape[0]

            with torch.no_grad():
                scores = model(x)

            loss = metrics.cross_entropy_loss(scores, y)
            total_celoss += loss.item() * y.shape[0]

            acc = train_utils.calculate_acc(scores, y)
            acc3, acc5 = train_utils.calculate_acc_topk(scores, y, topk=(3,5))
            accuracies.update(acc, x.size(0))
            accuracies_top3.update(acc3, x.size(0))
            accuracies_top5.update(acc5, x.size(0))

        results = {
            'dev_loss': total_celoss / total_item,
            'dev_acc': accuracies.avg,
            'dev_acc_top3': accuracies_top3.avg,
            'dev_acc_top5': accuracies_top5.avg,
        }

        pprint(results)
        return results


    results = train_utils.train_model(
        model, 
        train_epoch_fn=train_epoch,
        dev_epoch_fn=dev_epoch,
        max_epoch=MAX_EPOCH,
        save_model_name=SAVE_MODEL_NAME,
        epoch_start=0
    )






