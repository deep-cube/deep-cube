import torch
import os
import data_loader
import metrics
import train_utils
import data_def
import sys
from collections import defaultdict
from tqdm import tqdm

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
OVERLAP_LENGTH = 15
BEAM_SIZE = 50


def baseline_model_forward(x):
    B, L, C, H, W = x.shape
    scores = torch.zeros((B, L, data_def.NUM_CLASS)).to(DEVICE)
    scores[:, :, 0] = 1
    return scores


if __name__ == "__main__":
    model_name = sys.argv[1]
    if model_name == 'baseline':
        model = baseline_model_forward
        results = {}
    else:
        model, results = train_utils.load_model_save(
            model_name, load_best=True)
        print(model)

    def eval_model(video_filenames):
        edt_dist = 0
        total_frame = 0
        _, dataloader = data_loader.get_eval_dev_data(
            video_filenames, OVERLAP_LENGTH)
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

            edt_dist += metrics.sum_collapsed_edit_distance_beam(
                scores_trimmed, y_expanded_trimmed, beam_size=BEAM_SIZE
            )
            total_frame += (L - OVERLAP_LENGTH - OVERLAP_LENGTH)
        edpf = edt_dist / total_frame
        print('EDPF =', edpf)
        return edpf

    for name, filenames in {
        'train': data_loader.TRAIN_FILENAMES,
        'dev': data_loader.DEV_FILENAMES,
        'test': data_loader.TEST_FILENAMES,
    }.items():
        print(name)
        edpf = eval_model(filenames)
        results[f'{name}_eval_edpf'] = edpf

    if model_name != 'baseline':
        train_utils.save_results(
            results,
            os.path.join(
                train_utils.MODEL_SAVE_PATH,
                model_name,
                'results.json'
            )
        )
