from model_seq_cnn import *
from model_conv import ConvPredictor
from model_lrcn import *
from model_lrcn3d import *
from tqdm import trange, tqdm
from model_trainer import train_model, load_trained
from data_loader import *
from pprint import pprint
import data_utils
import torch
import numpy as np
import unittest
import data_def
import train_utils
import metrics
import sys

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

SAVE_MODEL_NAME = sys.argv[1]

TRAIN_WITH_CE_EPOCH = 0
TRAIN_WITH_CTC_EPOCH = 70

# early terminate each train epoch (it's ok since it's random sampled), and train for more epochs
TRAIN_EPOCH_SHORTEN_FACTOR = 2
MAX_EPOCH = (TRAIN_WITH_CE_EPOCH + TRAIN_WITH_CTC_EPOCH) * \
    TRAIN_EPOCH_SHORTEN_FACTOR


if __name__ == "__main__":
    B, L, C = 20, 100, data_utils.NUM_CHANNEL
    train_dataset, train_loader = get_train_data(
        B, L, shorten_factor=TRAIN_EPOCH_SHORTEN_FACTOR)
    dev_dataset, dev_loader = get_dev_data(B, L)
    H, W = train_dataset.frame_size()

    model, existing_results = load_trained(SAVE_MODEL_NAME)
    print(model)

    optimizer = torch.optim.Adam(model.parameters())
    loss_weights = torch.FloatTensor(
        data_def.class_loss_weights(train_dataset.get_label_counts())
    ).to(DEVICE)

    def train_epoch_fn(epoch_idx):
        train_with_ce = epoch_idx < TRAIN_WITH_CE_EPOCH

        total_celoss = 0
        total_ctcloss = 0
        total_edit_distance_hardmax = 0
        total_edit_distance_beam = 0
        total_item = 0

        num_batch = len(train_loader)
        for i, data in enumerate(tqdm(train_loader, desc=f'TRAIN_{"CE" if train_with_ce else "CTC"}', leave=False)):

            optimizer.zero_grad()

            x, y, y_collapsed, y_collapsed_l = data
            x = x.to(device=DEVICE, dtype=torch.float32)  # (B, L, C, H, W)
            y = y.to(device=DEVICE, dtype=torch.long)  # (B, L)
            y_collapsed = y_collapsed.to(device=DEVICE)
            y_collapsed_l = y_collapsed_l.to(device=DEVICE, dtype=torch.long)
            total_item += y.shape[0]

            scores = model(x)

            celoss = metrics.cross_entropy_loss_mean(scores, y, loss_weights)
            total_celoss += celoss.item() * y.shape[0]

            ctcloss = metrics.ctc_loss_sum(scores, y_collapsed, y_collapsed_l)
            total_ctcloss += ctcloss.item()

            loss = celoss if train_with_ce else ctcloss
            loss.backward()
            optimizer.step()

            total_edit_distance_hardmax += metrics.sum_collapsed_edit_distance_hardmax(
                scores, y)
            total_edit_distance_beam += metrics.sum_collapsed_edit_distance_beam(
                scores, y)

        total_loss = total_celoss if train_with_ce else total_ctcloss

        results = {
            'train_loss': total_loss / total_item,
            'train_ctcloss': total_ctcloss / total_item,
            'train_ce': total_celoss / total_item,
            'train_edit_distance_hardmax': total_edit_distance_hardmax / total_item / L,
            'train_edit_distance_beam': total_edit_distance_beam / total_item / L
        }

        print()
        pprint(results)
        return results

    def dev_epoch_fn(epoch_idx):
        total_celoss = 0
        total_edit_distance_hardmax = 0
        total_edit_distance_beam = 0

        total_ctcloss = 0
        total_item = 0

        for i, data in enumerate(tqdm(dev_loader, desc='DEV', leave=False)):
            x, y, y_collapsed, y_collapsed_l = data
            x = x.to(device=DEVICE, dtype=torch.float32)  # (B, L, C, H, W)
            y = y.to(device=DEVICE, dtype=torch.long)  # (B, L)
            y_collapsed = y_collapsed.to(device=DEVICE)
            y_collapsed_l = y_collapsed_l.to(device=DEVICE, dtype=torch.long)
            total_item += y.shape[0]

            with torch.no_grad():
                scores = model(x)

            # ce loss
            loss = metrics.cross_entropy_loss_mean(scores, y, loss_weights)
            total_celoss += loss.item() * y.shape[0]

            # edit dists
            total_edit_distance_hardmax += metrics.sum_collapsed_edit_distance_hardmax(
                scores, y)
            total_edit_distance_beam += metrics.sum_collapsed_edit_distance_beam(
                scores, y)

            # ctc loss
            ctcloss = metrics.ctc_loss_sum(scores, y_collapsed, y_collapsed_l)
            total_ctcloss += ctcloss.item()

        results = {
            'dev_loss': total_ctcloss / total_item,
            'dev_ctcloss': total_ctcloss / total_item,
            'dev_ce': total_celoss / total_item,
            'dev_edit_distance_hardmax': total_edit_distance_hardmax / total_item / L,
            'dev_edit_distance_beam': total_edit_distance_beam / total_item / L
        }

        pprint(results)
        return results

    results = train_utils.train_and_save(
        model,
        train_epoch_fn=train_epoch_fn,
        dev_epoch_fn=dev_epoch_fn,
        max_epoch=MAX_EPOCH,
        results=existing_results,
        save_model_name=SAVE_MODEL_NAME,
    )
