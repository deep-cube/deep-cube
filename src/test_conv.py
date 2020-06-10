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


class TestConv(unittest.TestCase):
    def test_forward_shape(self):
        for i in trange(10):
            B = np.random.randint(1, 10)
            L = np.random.randint(30, 100)
            C = np.random.randint(2, 4)
            H = np.random.randint(30, 50)
            W = np.random.randint(30, 50)
            num_classes = np.random.randint(2, 30)
            conv_module = SequentialCNN3D(
                C, H, W,
                [
                    (
                        np.random.randint(5, 10),  # outC
                        np.random.randint(1, 3) * 2 + 1,  # time_kernel
                        np.random.randint(1, 3) * 2 + 1  # space_kernel
                    )
                    for _ in range(np.random.randint(1, 3))
                ]
            )

            hidden_sizes = [
                np.random.randint(50, 200) for _ in range(np.random.randint(1, 3))
            ]

            model = ConvPredictor(
                conv_module,
                hidden_sizes,
                num_classes,
            )
            x = torch.zeros((B, L, C, H, W))
            y = model(x)

            assert y.shape == (B, L, num_classes)


if __name__ == "__main__":
    unittest.main()
