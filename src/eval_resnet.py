from tqdm import trange, tqdm
import data_loader
from pprint import pprint
import data_utils
import torch
import numpy as np
import data_def
import train_utils
import metrics
import sys
from resnet3d import resnet, densenet

MODEL_INFO_KEYS = ['train_loss', 'dev_loss', 'train_acc', 'train_acc_top3', 'train_acc_top5', 'dev_acc', 'dev_acc_top3', 'dev_acc_top5']
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

MIN_ACC_FRAME = 7

def print_model_info(model, results):
#     print(model)
    print('\nThe current model stats:')
    
    max_dev_idx = np.argmax(results['dev_acc'])
    for key in MODEL_INFO_KEYS:
        print(f'{key}:', results[key][max_dev_idx])

def inference(model, test_loader):
    model = model.to(device=DEVICE)
    model.eval()

    total_celoss = 0
    total_item = 0
    accuracies = train_utils.AverageMeter()
    accuracies_top3 = train_utils.AverageMeter()
    accuracies_top5 = train_utils.AverageMeter()

    # prec = [0.] * len(data_def.IDX_TO_CLASS_STR)
    # rel = [0.] * len(data_def.IDX_TO_CLASS_STR)
    # all_preds = None
    # all_targets = None
    prec_result = []

    for i, data in enumerate(tqdm(test_loader, desc='INFERENCE', leave=False)):
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

        # prec, rel, all_preds, all_targets = train_utils.update_prec_rel(scores, y, all_preds, all_targets, prec, rel)
        prec_result = train_utils.update_precision_result(scores, y, prec_result)

    mAP_score = train_utils.calc_mAP(prec_result)

    results = {
        'test_loss': total_celoss / total_item,
        'test_acc': accuracies.avg,
        'test_acc_top3': accuracies_top3.avg,
        'test_acc_top5': accuracies_top5.avg,
        'test_mAP': mAP_score,
    }

    pprint(results)
    return results


if __name__ == "__main__":
    model_name = sys.argv[1]
    model, results = train_utils.load_model_save(model_name, load_best=True)

    print_model_info(model, results)
    _, test_loader = data_loader.get_segmented_eval_data(MIN_ACC_FRAME)

    test_result=inference(model, test_loader)





