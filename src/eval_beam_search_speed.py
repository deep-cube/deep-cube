import torch
import os
import data_loader
import metrics
import train_utils
import sys
from collections import defaultdict
from tqdm import tqdm, trange
from pprint import pprint
import time

BEAM_SIZES = [
    1,
    20,
    50,
    100
]

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
OVERLAP_LENGTH = 15
REPEAT_COUNT = 5
MODEL_NAME = 'lrcn_deep_spatial_augment'  # todo use best model


if __name__ == "__main__":
    model, results = train_utils.load_model_save(MODEL_NAME)

    total_duration = defaultdict(float)  # ns
    edt_dist = defaultdict(float)
    total_frame = 0

    _, dataloader = data_loader.get_eval_dev_data(
        data_loader.TRAIN_FILENAMES, OVERLAP_LENGTH)
    for i, data in enumerate(tqdm(dataloader)):
        x, y_expanded, y_collapsed, y_collapsed_l = data
        B, L, C, H, W = x.shape
        x = x.to(device=DEVICE, dtype=torch.float32)  # (B, L, C, H, W)
        y_expanded = y_expanded.to(
            device=DEVICE, dtype=torch.long)  # (B, L)
        y_collapsed = y_collapsed.to(device=DEVICE)
        y_collapsed_l = y_collapsed_l.to(device=DEVICE, dtype=torch.long)

        with torch.no_grad():
            scores = model(x)

        scores_trimmed = scores[:, OVERLAP_LENGTH:-OVERLAP_LENGTH, :, ]
        y_expanded_trimmed = y_expanded[:, OVERLAP_LENGTH:-OVERLAP_LENGTH]

        for beam_size in BEAM_SIZES:
            sample_duration_sum = 0
            for i in trange(REPEAT_COUNT, desc='timer', leave=False):
                tik = time.time()
                sample_edit_dist = metrics.sum_collapsed_edit_distance_beam(
                    scores, y_expanded, beam_size=beam_size
                )
                tok = time.time()
                duration = tok-tik
                sample_duration_sum += duration
            edt_dist[beam_size] += sample_edit_dist
            total_duration[beam_size] += sample_duration_sum / REPEAT_COUNT

        total_frame += L - OVERLAP_LENGTH - OVERLAP_LENGTH

    for beam_size in BEAM_SIZES:
        edt_dist[beam_size] = edt_dist[beam_size] / total_frame
        total_duration[beam_size] = total_duration[beam_size] / total_frame

    print('edpf')
    pprint(edt_dist)
    print('avg duration per frame ms')
    pprint(total_duration)
