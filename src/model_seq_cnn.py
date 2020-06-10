import torch
from torch import nn


class SequentialCNN(nn.Module):

    def __init__(
        self,
        in_C, in_H, in_W,
        conv_specs,
        use_batchnorm=False,
        dropout=0.1
    ):
        """
        A convolutional module that takes in sequence of frames
        and returns a sequence of features

        Input of shape (B, L, C, H, W), where B is batch size, L is 
        sequence length (clip_length), C is number of channel per frame
        H and W are height and width of each frame

        Output of shape (B, L, CC, HH, WW), where B and N are preserved,
        CC, HH, WW are the resultant number of channel, height and width
        of each frame after the convolutions / pools

        The basic structure is:
        conv2d - (batchnorm2d) - relu - maxpool2d

        Args:
        - in_C (int): number of input channel
        - in_H (int): input height
        - in_W (int): input width
        - conv_specs (list): conv layer specs, list of (out_channel, kernel_size, maxpool_after)
            must be non empty, number of conv layer in module same as length of this list
        - use_batchnorm (bool, optional): use batchnorm after each conv2d. Defaults to False.
        """

        super(SequentialCNN, self).__init__()
        assert len(conv_specs) > 0, 'need at least one conv layer'

        layers = []
        H, W, C = self.HH, self.WW, self.CC = in_H, in_W, in_C

        for out_C, kernel_size, pool_after in conv_specs:
            assert (kernel_size % 2) == 1, 'kernel must be odd'
            padding = (kernel_size - 1) // 2
            layers.append(nn.Conv2d(
                in_channels=self.CC,
                out_channels=out_C,
                kernel_size=kernel_size,
                stride=1,
                padding=padding,
                bias=True,
                padding_mode='zeros'
            ))

            if use_batchnorm:
                layers.append(nn.BatchNorm2d(
                    num_features=out_C,
                    affine=True,  # learn gamma and beta
                    track_running_stats=True,  # running average at eval time
                ))

            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))

            if pool_after:
                layers.append(nn.MaxPool2d(kernel_size=2))
                self.HH = int((self.HH - 1 - 1) / 2 + 1)
                self.WW = int((self.WW - 1 - 1) / 2 + 1)

            self.CC = out_C

        self.module = torch.nn.Sequential(*layers)

    def forward(self, x):
        """
        Args:
        - x (torch.Tensor): of shape (B, L, C, H, W)

        Returns:
        - torch.Tensor: of shape (B, L, CC, HH, WW)
        """
        B, L, C, H, W = x.shape

        # (B, L, C, H, W)
        x = x.reshape(B * L, C, H, W)
        # (B * L, C, H, W)
        x = self.module(x)
        # (B * L, CC, HH, WW)
        x = x.reshape(B, L, self.CC, self.HH, self.WW)
        return x

    def output_shape(self):
        """
        Returns:
        - tuple: output size (CC, HH, WW)
        """
        return self.CC, self.HH, self.WW


class SequentialCNN3D(nn.Module):
    def __init__(
        self,
        C, H, W,
        conv_specs,
        use_batchnorm=False,
        dropout=0.1
    ):
        super(SequentialCNN3D, self).__init__()
        assert len(conv_specs) > 0, 'need at least one conv layer'

        self.conv_specs = conv_specs
        self.C, self.H, self.W = C, H, W
        self.use_batchnorm = use_batchnorm

        layers = []
        self.CC, self.HH, self.WW = C, H, W

        for out_C, time_kernel, space_kernel, pool_after in conv_specs:
            assert (space_kernel %
                    2) == 1, 'space-wise kernel size must be odd'
            assert (time_kernel %
                    2) == 1, 'time-wise kernel size must be odd'

            space_padding = (space_kernel - 1) // 2
            time_padding = (time_kernel - 1) // 2
            layers.append(nn.Conv3d(
                in_channels=self.CC,
                out_channels=out_C,
                kernel_size=(time_kernel, space_kernel, space_kernel),
                stride=1,
                padding=(time_padding, space_padding, space_padding),
                bias=True,
                padding_mode='zeros'
            ))

            if use_batchnorm:
                layers.append(nn.BatchNorm3d(
                    num_features=out_C
                ))

            layers.append(nn.LeakyReLU())
            layers.append(nn.Dropout(dropout))

            if pool_after:
                layers.append(nn.MaxPool3d(
                    kernel_size=(1, 2, 2)  # dont maxpool time-wise
                ))
                self.HH = int((self.HH - 1 - 1) / 2 + 1)
                self.WW = int((self.WW - 1 - 1) / 2 + 1)

            self.CC = out_C

        self.module = torch.nn.Sequential(*layers)

    def forward(self, x):
        B, L, C, H, W = x.shape

        # (B, L, C, H, W)
        x = torch.transpose(x, 1, 2)
        # (B, C, L, H, W)
        x = self.module(x)
        # (B, CC, L, HH, WW)
        x = torch.transpose(x, 1, 2)
        # (B, L, CC, HH, WW)

        return x

    def output_shape(self):
        """
        Returns:
        - tuple: output size (CC, HH, WW)
        """
        return self.CC, self.HH, self.WW


class SequentialCNN3DFrontTimePad(nn.Module):
    def __init__(
        self,
        C, H, W,
        conv_specs,
        use_batchnorm=False,
        dropout=0.1
    ):
        super(SequentialCNN3DFrontTimePad, self).__init__()
        assert len(conv_specs) > 0, 'need at least one conv layer'

        self.conv_specs = conv_specs
        self.C, self.H, self.W = C, H, W
        self.use_batchnorm = use_batchnorm

        layers = []
        self.CC, self.HH, self.WW = C, H, W

        for out_C, time_kernel, space_kernel, pool_after in conv_specs:
            assert (space_kernel %
                    2) == 1, 'space-wise kernel size must be odd'
            assert (time_kernel %
                    2) == 1, 'time-wise kernel size must be odd'

            time_padding = time_kernel - 1
            layers.append(nn.ConstantPad3d(
                padding=(
                    0, 0,
                    0, 0,
                    time_padding - 1, 1  # time dim
                ),
                value=0
            ))

            space_padding = (space_kernel - 1) // 2
            layers.append(nn.Conv3d(
                in_channels=self.CC,
                out_channels=out_C,
                kernel_size=(time_kernel, space_kernel, space_kernel),
                stride=1,
                padding=(0, space_padding, space_padding),
                bias=True,
                padding_mode='zeros'
            ))

            if use_batchnorm:
                layers.append(nn.BatchNorm3d(
                    num_features=out_C
                ))

            layers.append(nn.LeakyReLU())
            layers.append(nn.Dropout(dropout))

            if pool_after:
                layers.append(nn.MaxPool3d(
                    kernel_size=(1, 2, 2)  # dont maxpool time-wise
                ))
                self.HH = int((self.HH - 1 - 1) / 2 + 1)
                self.WW = int((self.WW - 1 - 1) / 2 + 1)

            self.CC = out_C

        self.module = torch.nn.Sequential(*layers)

    def forward(self, x):
        B, L, C, H, W = x.shape

        # (B, L, C, H, W)
        x = torch.transpose(x, 1, 2)
        # (B, C, L, H, W)
        x = self.module(x)
        # (B, CC, L, HH, WW)
        x = torch.transpose(x, 1, 2)
        # (B, L, CC, HH, WW)

        return x

    def output_shape(self):
        """
        Returns:
        - tuple: output size (CC, HH, WW)
        """
        return self.CC, self.HH, self.WW
