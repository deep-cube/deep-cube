import torch
import os
import numpy as np
from tqdm import tqdm
from pprint import pprint
from data_utils import array_to_video_view


class SingleSquareActivatedDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        L, C, H, W,
        square_persistence_length=1,
        use_background_noise=False,
        dataset_length=100
    ):
        """
        a dataset that generates dummy training data to test a conv+recurrent model
        by populating frames of data with a black background / noise background
        then a fixed number of frames with a stationary white square of size exactly
        a quarter of the frame, in a random position, the frames with the squares in 
        them depend on the label.

        intuitively, a model should be able to learn this simple pattern of "observing
        n consecutive white square should produce a prediction of 1, otherwise 0"

        Args:
        - torch ([type]): [description]
        - L (int): length of each clip 
        - C (int): num channel
        - H (int): height of frame
        - W (int): width of frame
        - square_persistence_length (int, optional): number of white squares before
            a prediction of 1. Defaults to 1.
        - use_background_noise (bool, optional): if true, background is random noise
            pixles, if false, background is black. Defaults to False.
        - dataset_length (int, optional): number of clips in the dataset. Defaults to 100.
        """
        self.C, self.H, self.W, self.L = C, H, W, L
        self.square_size = min(H, W) // 2
        self.use_background_noise = use_background_noise
        self.dataset_length = dataset_length
        self.square_persistence_length = square_persistence_length
        self.random_state = None

    def __len__(self):
        return self.dataset_length

    def __getitem__(self, idx):
        if self.random_state is None:
            self.random_state = np.random.RandomState()

        y = self.random_state.choice(
            2,
            size=(self.L,),
            p=[0.9, 0.1]
        )

        for l in range(self.L-1):
            if y[l-self.square_persistence_length:l].sum() > 0:
                y[l] = 0

        if self.use_background_noise:
            x = self.random_state.choice(
                256, size=(self.L, self.C, self.H, self.W))
        else:
            x = np.zeros((self.L, self.C, self.H, self.W))

        for l in range(self.L):
            if l < self.square_persistence_length:
                continue
            if y[l] == 1:
                up = self.random_state.choice(self.H - self.square_size)
                down = up + self.square_size
                left = self.random_state.choice(self.W - self.square_size)
                right = left + self.square_size
                x[
                    l+1-self.square_persistence_length: l+1,
                    :, up:down, left:right
                ] = 255 - self.random_state.choice(20)

        return x, y, len(y)


class SingleSquareActivatedMultiClassCTCDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        L, C, H, W,
        square_persistence_length=1,
        use_background_noise=False,
        dataset_length=100
    ):
        """
        a dataset that generates dummy training data to test a conv+recurrent model
        by populating frames of data with a black background / noise background
        then a fixed number of frames with a stationary white square of size exactly
        a quarter of the frame, in a random position, the frames with the squares in 
        them depend on the label.

        intuitively, a model should be able to learn this simple pattern of "observing
        n consecutive white square should produce a prediction of 1, otherwise 0"

        Args:
        - torch ([type]): [description]
        - L (int): length of each clip 
        - C (int): num channel
        - H (int): height of frame
        - W (int): width of frame
        - square_persistence_length (int, optional): number of white squares before
            a prediction of 1. Defaults to 1.
        - use_background_noise (bool, optional): if true, background is random noise
            pixles, if false, background is black. Defaults to False.
        - dataset_length (int, optional): number of clips in the dataset. Defaults to 100.
        """
        self.C, self.H, self.W, self.L = C, H, W, L
        self.square_size = min(H, W) // 2
        self.use_background_noise = use_background_noise
        self.dataset_length = dataset_length
        self.square_persistence_length = square_persistence_length
        self.random_state = None

    def __len__(self):
        return self.dataset_length

    def __getitem__(self, idx):
        if self.random_state is None:
            self.random_state = np.random.RandomState()

        y = self.random_state.choice(
            2,
            size=(self.L,),
            p=[0.9, 0.1]
        )

        collapsed_y = []

        for l in range(self.L-1):
            if y[l-self.square_persistence_length:l].sum() > 0:
                y[l] = 0

        if self.use_background_noise:
            x = self.random_state.choice(
                256, size=(self.L, self.C, self.H, self.W))
        else:
            x = np.zeros((self.L, self.C, self.H, self.W))

        for l in range(self.L):
            if l < self.square_persistence_length:
                continue
            if y[l] == 1:
                up = self.random_state.choice(self.H - self.square_size)
                down = up + self.square_size
                left = self.random_state.choice(self.W - self.square_size)
                right = left + self.square_size
                x[
                    l+1-self.square_persistence_length: l+1,
                    :, up:down, left:right
                ] = 0
                channel_to_activate = self.random_state.choice(self.C)
                x[
                    l+1-self.square_persistence_length: l+1,
                    channel_to_activate, up:down, left:right
                ] = 255
                collapsed_y.append(channel_to_activate + 1)

        l = len(collapsed_y)
        collapsed_y = np.pad(
            np.array(collapsed_y, dtype=np.uint8), ((0, self.L - l)))

        return x, collapsed_y, l


if __name__ == "__main__":
    ds = SingleSquareActivatedMultiClassCTCDataset(
        30, 100, 100,
        square_persistence_length=1,
        use_background_noise=True,
    )
    x, y = ds[0]
    print(y)
    print(x.shape, y.shape)
    array_to_video_view(x)

    # ds = SingleSquareActivatedDataset(50, 1, 10, 10)
    # loader = torch.utils.data.DataLoader(
    #     ds, batch_size=10, num_workers=1
    # )
    # for i, data in enumerate(tqdm(loader)):
    #     x, y = data
    #     print(x.shape, y.shape)

    # ds = SingleSquareActivatedDataset(
    #     50, 3, 100, 100,
    #     square_persistence_length=3,
    #     use_background_noise=True,
    # )
    # x, y = ds[0]
    # print(y)
    # print(x.shape, y.shape)
    # array_to_video_view(x, y)
