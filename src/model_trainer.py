import torch
import os
import json
from tqdm import trange, tqdm
import numpy as np
import pickle
from pprint import pprint

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_SAVE_PATH = os.path.realpath(
    os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        '..', 'saved_models'
    )
)


def on_batch_placeholder(batch_idx, x, y, scores):
    pass


def load_trained(save_model_name, device=DEVICE):
    save_model_path = os.path.join(MODEL_SAVE_PATH, save_model_name)
    newest_model_path = os.path.join(save_model_path, 'newest.pth')
    results_path = os.path.join(save_model_path, 'results.json')

    model = torch.load(newest_model_path, map_location=device)

    results = None
    if os.path.exists(results_path):
        with open(results_path) as f:
            results = json.load(f)
    return model, results


def train_model(
    model,
    train_dataloader,
    optimizer,
    criterion_name='cross_entropy',
    loss_weights=None,
    dev_dataloader=None,
    test_dataloader=None,
    on_batch=on_batch_placeholder,
    num_epoch=20,
    device=DEVICE,
    existing_train_results=None,
    save_model_name=None,
    overwrite=False,
    init_only=False,
    additional_metrics={},
):
    '''
    dev in progress
    '''
    if save_model_name is not None:
        save_model_path = os.path.join(MODEL_SAVE_PATH, save_model_name)

        if os.path.exists(save_model_path) and not overwrite:
            raise ValueError(
                f'{save_model_name} already exists, and `overwrite` is false, abort')
        elif not os.path.exists(save_model_path):
            os.mkdir(save_model_path)

        print(
            f'Trainer will save model and training results to [{save_model_path}]'
        )
        newest_model_path = os.path.join(save_model_path, 'newest.pth')
        best_model_path = os.path.join(save_model_path, 'best.pth')
        results_path = os.path.join(save_model_path, 'results.json')

        model_specs_path = os.path.join(save_model_path, 'model_specs.txt')
        with open(model_specs_path, 'w') as f:
            f.write(model.__repr__())
            f.write('\n')

    if existing_train_results is not None:
        results = existing_train_results
    else:
        results = {
            'train_loss': [],
            'dev_loss': [],
            'test_loss': [],
        }

    if init_only:
        assert save_model_name is not None, '`init_only` must have `save_model_name`'
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        torch.save(model, newest_model_path)
        return results

    model = model.to(device=device)

    best_dev_loss = min(results['dev_loss'], default=float('inf'))
    print(f'Starting train, best_dev_loss=[{best_dev_loss}]')

    for e in trange(num_epoch, desc='EPOCH'):
        model.train()
        mean_train_loss = run_epoch(
            model, train_dataloader,
            optimizer, device, 'train',
            result_dict=results,
            backward=True,
            criterion_name=criterion_name,
            loss_weights=loss_weights,
            show_progress=True,
            on_batch=on_batch,
            additional_metrics=additional_metrics
        )
        print(
            f'TRAIN      | LOSS={mean_train_loss:.7f}'
        )

        if dev_dataloader is not None:
            model.eval()
            mean_dev_loss = run_epoch(
                model, dev_dataloader,
                optimizer, device, 'dev',
                result_dict=results,
                backward=False,
                criterion_name=criterion_name,
                loss_weights=loss_weights,
                additional_metrics=additional_metrics
            )
            print(
                f'DEV        | LOSS={mean_dev_loss:.7f}'
            )
            if mean_dev_loss <= best_dev_loss:
                print('Dev loss improved, saving to best.pth')
                best_dev_loss = mean_dev_loss
                if save_model_name is not None:
                    torch.save(model, best_model_path)

        if save_model_name is not None:
            with open(results_path, 'w') as f:
                json.dump(results, f, indent=2)
            torch.save(model, newest_model_path)

    if test_dataloader is not None:
        model.eval()
        mean_test_loss = run_epoch(
            model, test_dataloader,
            optimizer, device, 'test',
            result_dict=results,
            backward=False,
            criterion_name=criterion_name,
            loss_weights=loss_weights,
            additional_metrics=additional_metrics
        )
        print(
            f'TEST       | LOSS={mean_test_loss:.7f}'
        )

    if save_model_name is not None:
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)

    return results


def run_epoch(
    model, dataloader, optimizer,
    device, desc,
    result_dict,
    backward,
    show_progress=False,
    criterion_name='cross_entropy',
    loss_weights=None,
    on_batch=on_batch_placeholder,
    additional_metrics={},
):
    if loss_weights is not None:
        loss_weights = loss_weights.to(device)

    total_loss = 0
    total_item = 0
    additional_metrics_total = {k: 0 for k in additional_metrics}

    if show_progress:
        enumerator = enumerate(tqdm(dataloader, desc=desc, leave=False))
    else:
        enumerator = enumerate(dataloader)

    for i, (x, y, l) in enumerator:

        optimizer.zero_grad()
        x = x.to(device=device, dtype=torch.float32)  # (B, L, C, H, W)
        y = y.to(device=device, dtype=torch.long)  # (B, L)

        if backward:
            scores = model(x)
        else:
            with torch.no_grad():
                scores = model(x)

        on_batch(i, x, y, scores)

        if criterion_name == 'cross_entropy':
            y_reshaped = y.reshape(-1)  # (B * L)
            # (B * L, num_class)
            scores_reshaped = scores.reshape(-1, scores.shape[-1])
            criterion = torch.nn.CrossEntropyLoss(
                reduction='mean',
                weight=loss_weights
            )
            loss = criterion(scores_reshaped, y_reshaped)
        elif criterion_name == 'ctc':
            max_len = l.max()
            y_filtered = y[:, :max_len]
            B, L, num_class = scores.shape
            scores_reshaped = scores.log_softmax(dim=-1)  # (B, L, num_class)
            scores_reshaped = torch.transpose(
                scores_reshaped, 0, 1)  # (L, B, num_class)
            target = y_filtered  # (B, L)
            input_lengths = torch.full(
                size=(B,), fill_value=L, dtype=torch.long
            )
            target_lengths = l.to(device=device, dtype=torch.long)  # (B, )
            criterion = torch.nn.CTCLoss(
                blank=0, reduction='sum'
            )
            loss = criterion(scores_reshaped, target,
                             input_lengths, target_lengths)
        else:
            raise ValueError(f'no such criterion [{criterion_name}]')

        for metric_name in additional_metrics:
            metric_func = additional_metrics[metric_name]
            metric_val = metric_func(scores, y)
            additional_metrics_total[metric_name] += metric_val

        if backward:
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * y.shape[0]
        total_item += y.shape[0]

    if show_progress:
        print()

    mean_loss = total_loss / total_item
    loss_name = f'{desc}_loss'
    result_dict[loss_name].append(mean_loss)

    for metric_name in additional_metrics:
        metric_key = f'{desc}_{metric_name}'
        if metric_key not in result_dict:
            result_dict[metric_key] = []
        result_dict[metric_key].append(
            additional_metrics_total[metric_name] / total_item
        )

    return mean_loss
