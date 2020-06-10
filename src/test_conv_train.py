from model_lcrn import LRCN
from model_seq_cnn import *
from model_conv import ConvPredictor
from tqdm import trange, tqdm
from test_dummy_data_generator import *
from model_trainer import train_model
import torch
import numpy as np
import unittest
import data_def
import metrics


class TestConvTrain(unittest.TestCase):

    def _run_single_square_activated(
        self,
        square_persistence_length,
        use_background_noise=False,
        num_epochs=5
    ):
        B, L, C, H, W = 50, 30, 3, 30, 30
        num_class = 2

        conv_module = SequentialCNN3D(
            C, H, W,
            [
                (16, 13, 7, True),
                (32, 13, 5, True),
                (64, 13, 3, True),
            ],
            use_batchnorm=True
        )
        hidden_sizes = [300, 100]
        model = ConvPredictor(
            conv_module,
            hidden_sizes,
            num_class,
        )
        print(model)

        train_loader = torch.utils.data.DataLoader(
            SingleSquareActivatedDataset(
                L, C, H, W,
                use_background_noise=use_background_noise,
                dataset_length=1000,
                square_persistence_length=square_persistence_length
            ),
            batch_size=B, num_workers=1,
        )
        dev_loader = torch.utils.data.DataLoader(
            SingleSquareActivatedDataset(
                L, C, H, W,
                use_background_noise=use_background_noise,
                dataset_length=200,
                square_persistence_length=square_persistence_length
            ),
            batch_size=B, num_workers=1,
        )

        def on_batch(i, x, y, scores):
            if i % 5 == 0:
                print(y.shape)
                print(scores.shape)
                # yhat_collapsed = data_def.collapse(
                #     torch.argmax(scores[0], dim=-1).cpu().numpy())
                # y_collapsed = data_def.collapse(y[0].cpu().numpy())
                # print(yhat_collapsed)
                # print(y_collapsed)
                print(torch.argmax(scores[0], dim=-1))
                print(y[0])
                print('edit dist')
                print(metrics.sum_edit_distance(scores, y) / len(y))

        optimizer = torch.optim.Adam(model.parameters())
        results = train_model(
            model,
            train_dataloader=train_loader,
            optimizer=optimizer,
            criterion_name='cross_entropy',
            dev_dataloader=dev_loader,
            test_dataloader=dev_loader,
            num_epoch=num_epochs,
            on_batch=on_batch,
            additional_metrics={
                'edit_distance': metrics.sum_edit_distance
            }
        )

        pprint(results)

        # assert results['train_acc'][-1] > 0.92
        # assert results['dev_acc'][-1] > 0.92

    # def test_1_frame_without_noise(self):
    #     print('test_1_frame_without_noise')
    #     self._run_single_square_activated(1, num_epochs=5)

    # def test_3_frame_without_noise(self):
    #     print('test_3_frame_without_noise')
    #     self._run_single_square_activated(3, num_epochs=5)

    def test_1_frame_with_noise(self):
        print('test_1_frame_with_noise')
        self._run_single_square_activated(
            3, use_background_noise=True, num_epochs=5)


if __name__ == "__main__":
    unittest.main()
