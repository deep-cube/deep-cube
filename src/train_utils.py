import os
import json
import numpy as np
from tqdm import trange, tqdm
import torch
from sklearn.metrics import precision_recall_fscore_support, precision_score
import data_def

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_SAVE_PATH = os.path.realpath(
    os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        '..', 'saved_models'
    )
)

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def init_model_save(model, save_model_name):
    save_model_path = os.path.join(MODEL_SAVE_PATH, save_model_name)
    assert (not os.path.exists(save_model_path)
            ), f'{save_model_path} already exists'
    os.mkdir(save_model_path)

    newest_model_path = os.path.join(save_model_path, 'newest.pth')
    best_model_path = os.path.join(save_model_path, 'best.pth')
    results_path = os.path.join(save_model_path, 'results.json')
    model_specs_path = os.path.join(save_model_path, 'model_specs.txt')

    results = {
        'train_loss': [],
        'dev_loss': [],
        'test_loss': [],
    }

    torch.save(model, newest_model_path)
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    with open(model_specs_path, 'w') as f:
        f.write(model.__repr__())
        f.write('\n')


def load_model_save(save_model_name, device=DEVICE, load_best=False):
    save_model_path = os.path.join(MODEL_SAVE_PATH, save_model_name)
    newest_model_path = os.path.join(save_model_path, 'newest.pth')
    best_model_path = os.path.join(save_model_path, 'best.pth')
    results_path = os.path.join(save_model_path, 'results.json')

    model = None
    
    if load_best:
        model = torch.load(best_model_path, map_location=device)
    else:
        model = torch.load(newest_model_path, map_location=device)

    results = None
    if os.path.exists(results_path):
        with open(results_path) as f:
            results = json.load(f)
    return model, results


def train_and_save(
    model,
    train_epoch_fn,
    max_epoch,
    results,
    save_model_name,
    dev_epoch_fn=None,
    device=DEVICE
):
    model = model.to(device=device)

    save_model_path = os.path.join(MODEL_SAVE_PATH, save_model_name)
    newest_model_path = os.path.join(save_model_path, 'newest.pth')
    best_model_path = os.path.join(save_model_path, 'best.pth')
    results_path = os.path.join(save_model_path, 'results.json')
    print(f'Will save model checkpoints to [{save_model_path}]')

    epoch_start = len(results['train_loss'])
    print(f'starting epoch={epoch_start}, end epoch={max_epoch}')

    best_dev_loss = min(results['dev_loss'], default=float('inf'))
    print(f'Starting train, best_dev_loss=[{best_dev_loss}]')

    for e in trange(epoch_start, max_epoch, desc='EPOCH'):
        model.train()
        train_result = train_epoch_fn(e)
        merge_results(results, train_result)
        save_results(results, results_path)
        torch.save(model, newest_model_path)

        if dev_epoch_fn is not None:
            model.eval()
            dev_result = dev_epoch_fn(e)
            merge_results(results, dev_result)
            save_results(results, results_path)
            new_dev_loss = dev_result['dev_loss']
            if new_dev_loss < best_dev_loss:
                print('dev loss improved')
                best_dev_loss = new_dev_loss
                torch.save(model, best_model_path)

    return results

def train_model(
        model, 
        train_epoch_fn,
        max_epoch,
        save_model_name,
        epoch_start=0,
        dev_epoch_fn=None,
        device=DEVICE
):
    results = {
        'train_loss': [],
        'dev_loss': [],
        'test_loss': [],
        'train_acc': [],
        'dev_acc': [],
    }

    model = model.to(device=device)
    save_model_path = os.path.join(MODEL_SAVE_PATH, save_model_name)
    newest_model_path = os.path.join(save_model_path, 'newest.pth')
    best_model_path = os.path.join(save_model_path, 'best.pth')
    results_path = os.path.join(save_model_path, 'results.json')
    print(f'\nWill save model checkpoints to [{save_model_path}]')

    best_dev_loss = float('inf')
    print(f'starting epoch={epoch_start}, end epoch={max_epoch}')
    print(f'Starting train...')

    for e in trange(epoch_start,max_epoch, desc='EPOCH'):
        model.train()
        train_result = train_epoch_fn(e)
        merge_results(results, train_result)
        save_results(results, results_path)
        torch.save(model, newest_model_path)

        if dev_epoch_fn is not None:
            model.eval()
            dev_result=dev_epoch_fn(e)
            new_dev_loss = dev_result['dev_loss']
            merge_results(results, dev_result)
            save_results(results, results_path)
            if new_dev_loss < best_dev_loss:
                print('dev loss improved')
                best_dev_loss = new_dev_loss
                torch.save(model, best_model_path)

    return results

def save_results(results, path):
    with open(path, 'w') as f:
        json.dump(results, f, indent=2)


def merge_results(results, new_results):
    for k in new_results:
        if k not in results:
            results[k] = []
        results[k].append(new_results[k])

def calculate_acc(outputs, targets):
    with torch.no_grad():
        batch_size = targets.size(0)

        _, pred = outputs.topk(1, 1, largest=True, sorted=True)
        pred = pred.t()
        correct = pred.eq(targets.view(1, -1))
        n_correct_elems = correct.float().sum().item()

        return n_correct_elems * 100.0 / batch_size

def calculate_acc_topk(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        # transpose
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size).item())
        return res

def calculate_precision_and_recall(outputs, targets, pos_label=1):
    with torch.no_grad():
        _, pred = outputs.topk(1, 1, largest=True, sorted=True)
        precision, recall, _, _ = precision_recall_fscore_support(
            targets.view(-1, 1).cpu().numpy(),
            pred.cpu().numpy())

        
        return precision[pos_label], recall[pos_label]


def update_precision_result(outputs, targets, prec_result):
    """
    prec_result = [[{confidence}, {pred}, {target}], ...]
    """
    with torch.no_grad():
        batch_size = targets.size(0)

        confidence, pred = outputs.topk(1, 1, largest=True, sorted=True)
        confidence = confidence.t().flatten().cpu().numpy()
        pred = pred.t().flatten().cpu().numpy()
        targets = targets.cpu().numpy()

        for i, (p, t) in enumerate(zip(pred, targets)):
            prec_result.append([confidence[i], p, t])

    return prec_result

def calc_mAP(prec_result):

    prec_result.sort(key=lambda x:x[0], reverse=True)
        
    all_preds = []
    all_targets = []
    labels = list(data_def.IDX_TO_CLASS_STR.keys())

    prec = [0.] * len(data_def.IDX_TO_CLASS_STR)
    rel = [0.] * len(data_def.IDX_TO_CLASS_STR)

    for (_, pred, target) in prec_result:
        all_preds.append(pred)
        all_targets.append(target)
        
        is_correct = True if pred == target else False
        cur_prec = precision_score(all_targets, all_preds, labels=list(data_def.IDX_TO_CLASS_STR.keys()), average=None)

        if is_correct:
            prec[pred] += cur_prec[pred]
            rel[pred] += 1

    # print(cur_prec)
    # print(prec)
    # print(rel)

    interporlated_APs = []
    for p, r in zip(prec, rel):
        # no correct prediction for this class
        if r == 0:
            interporlated_APs.append(0)
        else:
            interporlated_APs.append( p * 1.0 / r )

#     print(interporlated_APs)
    # total AP divided by the total num of classes
    mean_AP = sum(interporlated_APs) / len(interporlated_APs)
    
    return mean_AP * 100.0




        









