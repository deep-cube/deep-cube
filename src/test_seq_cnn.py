from model_seq_cnn import *
from tqdm import trange
import torch
import numpy as np
import unittest


class TestSeqCNN(unittest.TestCase):

    def test_shape_2d(self):
        for i in trange(20, desc='test_shape_2d'):
            in_C = np.random.randint(2, 4)
            in_H = np.random.randint(30, 100)
            in_W = np.random.randint(30, 100)
            model = SequentialCNN(
                in_C, in_H, in_W,
                [
                    (np.random.randint(5, 10), np.random.randint(1, 3) * 2 + 1)
                    for _ in range(np.random.randint(1, 3))
                ]
            )
            x = torch.zeros((
                np.random.randint(5, 10),  # batchsize
                np.random.randint(5, 10),  # clip_length
                in_C, in_H, in_W
            ))  # (B, N, C, H, W)
            y = model(x)  # (B, N, CC, HH, WW)

            assert y.shape[-3:] == model.output_shape()
            assert y.shape[:2] == x.shape[:2]

    def test_shape_3d(self):
        for i in trange(20, desc='test_shape_3d'):
            in_C = np.random.randint(2, 4)
            in_H = np.random.randint(30, 100)
            in_W = np.random.randint(30, 100)
            B = np.random.randint(5, 10)
            L = np.random.randint(5, 10)

            model = SequentialCNN3D(
                in_C, in_H, in_W,
                [
                    (
                        np.random.randint(5, 10),  # outC
                        np.random.randint(1, 3) * 2 + 1,  # time_kernel
                        np.random.randint(1, 3) * 2 + 1,  # space_kernel
                        True
                    )
                    for _ in range(np.random.randint(1, 3))
                ]
            )
            x = torch.zeros((B, L, in_C, in_H, in_W))  # (B, L, C, H, W)
            y = model(x)  # (B, L, CC, HH, WW)

            assert y.shape[-3:] == model.output_shape()  # CC HH WW
            assert y.shape[:2] == x.shape[:2]  # B L

    def test_shape_3d_front_time_pad(self):
        for i in trange(20, desc='test_shape_3d'):
            in_C = np.random.randint(2, 4)
            in_H = np.random.randint(30, 100)
            in_W = np.random.randint(30, 100)
            B = np.random.randint(5, 10)
            L = np.random.randint(5, 10)

            model = SequentialCNN3DFrontTimePad(
                in_C, in_H, in_W,
                [
                    (
                        np.random.randint(5, 10),  # outC
                        np.random.randint(1, 3) * 2 + 1,  # time_kernel
                        np.random.randint(1, 3) * 2 + 1,  # space_kernel
                        True
                    )
                    for _ in range(np.random.randint(1, 3))
                ]
            )
            x = torch.zeros((B, L, in_C, in_H, in_W))  # (B, L, C, H, W)
            y = model(x)  # (B, L, CC, HH, WW)

            assert y.shape[-3:] == model.output_shape()  # CC HH WW
            assert y.shape[:2] == x.shape[:2]  # B L


if __name__ == "__main__":
    unittest.main()
