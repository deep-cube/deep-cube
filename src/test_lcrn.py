from model_lcrn import LRCN
from model_seq_cnn import SequentialCNN
from tqdm import trange, tqdm
from test_dummy_data_generator import *
from model_trainer import train_model
import torch
import numpy as np
import unittest
import data_def


class TestLCRN(unittest.TestCase):

    # def test_ctc_data(self):
    #     B, L, H, W = 100, 30, 30, 30
    #     model = LRCN(
    #         SequentialCNN(
    #             3, H, W,
    #             conv_specs=[(10, 5), (20, 5)],
    #             use_batchnorm=False
    #         ),
    #         lstm_hidden_size=100,
    #         lstm_num_layers=1,
    #         num_classes=4,
    #     )
    #     train_loader = torch.utils.data.DataLoader(
    #         SingleSquareActivatedMultiClassCTCDataset(
    #             L, H, W,
    #             use_background_noise=False,
    #             dataset_length=3000,
    #             square_persistence_length=3
    #         ),
    #         batch_size=B, num_workers=1,
    #     )

    #     def on_batch(i, x, y, scores):
    #         if i == 0:
    #             print()
    #             yhat_collapsed = data_def.collapse(
    #                 torch.argmax(scores[0], dim=-1).cpu().numpy())
    #             y_collapsed = data_def.collapse(y[0].cpu().numpy())
    #             print(yhat_collapsed)
    #             print(y_collapsed)

    #     optimizer = torch.optim.Adam(model.parameters())
    #     results = train_model(
    #         model,
    #         optimizer=optimizer,
    #         train_dataloader=train_loader,
    #         criterion_name='ctc',
    #         num_epoch=40,
    #         on_batch=on_batch
    #     )

    #     print(results)

    def _run_single_square_activated(
        self,
        square_persistence_length,
        use_background_noise=False,
        num_epochs=5
    ):
        B, L, C, H, W = 50, 30, 3, 30, 30
        num_class = 2
        model = LRCN(
            SequentialCNN(
                C, H, W,
                conv_specs=[(10, 5), (20, 5)],
                use_batchnorm=True
            ),
            lstm_hidden_size=50,
            lstm_num_layers=2,
            num_classes=num_class,
        )
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
            if i == 0:
                print(y.shape)
                print(scores.shape)
                # yhat_collapsed = data_def.collapse(
                #     torch.argmax(scores[0], dim=-1).cpu().numpy())
                # y_collapsed = data_def.collapse(y[0].cpu().numpy())
                # print(yhat_collapsed)
                # print(y_collapsed)
                print(torch.argmax(scores[0], dim=-1))
                print(y[0])

        optimizer = torch.optim.Adam(model.parameters())
        results = train_model(
            model,
            train_dataloader=train_loader,
            optimizer=optimizer,
            criterion_name='cross_entropy',
            dev_dataloader=dev_loader,
            test_dataloader=dev_loader,
            num_epoch=num_epochs,
            on_batch=on_batch
        )

        assert results['train_acc'][-1] > 0.92
        assert results['dev_acc'][-1] > 0.92

    def test_1_frame_without_noise(self):
        print('test_1_frame_without_noise')
        self._run_single_square_activated(1, num_epochs=10)

    # def test_3_frame_interval_without_noise(self):
    #     print('test_3_frame_interval_without_noise')
    #     self._run_single_square_activated(3)

    # def test_1_frame_with_noise(self):
    #     print('test_1_frame_with_noise')
    #     self._run_single_square_activated(
    #         1, use_background_noise=True, num_epochs=10)

    # def test_3_frame_interval_with_noise(self):
    #     print('test_3_frame_interval_with_noise')
    #     self._run_single_square_activated(
    #         3, use_background_noise=True, num_epochs=10)

    # def test_forward_shape(self):
    #     for i in trange(30):
    #         B = np.random.randint(1, 10)
    #         L = np.random.randint(30, 100)
    #         C = np.random.randint(2, 4)
    #         H = np.random.randint(30, 50)
    #         W = np.random.randint(30, 50)
    #         conv_specs = [
    #             (np.random.randint(5, 10), np.random.randint(1, 3) * 2 + 1)
    #             for _ in range(np.random.randint(1, 2))
    #         ]
    #         lstm_hidden_size = np.random.randint(10, 30)
    #         lstm_num_layers = np.random.randint(1, 3)
    #         num_classes = np.random.randint(2, 30)

    #         use_lstm_state = np.random.choice([False, True])
    #         lstm_hidden_state = None
    #         if use_lstm_state:
    #             lstm_hidden_state = (
    #                 torch.zeros((lstm_num_layers, B, lstm_hidden_size)),
    #                 torch.zeros((lstm_num_layers, B, lstm_hidden_size))
    #             )

    #         model = LRCN(
    #             SequentialCNN(
    #                 C, H, W,
    #                 conv_specs=conv_specs,
    #                 use_batchnorm=True
    #             ),
    #             lstm_hidden_size,
    #             lstm_num_layers,
    #             num_classes,
    #         )
    #         x = torch.zeros((B, L, C, H, W))
    #         y = model(x, lstm_hidden_state)
    #         h, c = model.prev_lstm_state

    #         assert y.shape == (B, L, num_classes)
    #         assert h.shape == (lstm_num_layers, B, lstm_hidden_size)
    #         assert c.shape == (lstm_num_layers, B, lstm_hidden_size)


if __name__ == "__main__":
    unittest.main()
