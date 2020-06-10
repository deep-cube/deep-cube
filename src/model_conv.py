import torch
import sys
import data_def
from model_seq_cnn import *
import train_utils
import data_utils


class ConvPredictor(torch.nn.Module):

    def __init__(
        self,
        conv_module,
        hidden_sizes,
        num_classes,
        dropout=0.1
    ):
        super(ConvPredictor, self).__init__()

        assert len(hidden_sizes) > 0, 'need at least 1 hddenlayer'
        self.num_classes = num_classes
        self.hidden_sizes = hidden_sizes

        self.conv_module = conv_module
        self.CC, self.HH, self.WW = self.conv_module.output_shape()

        layers = []
        feat_in = self.CC * self.HH * self.WW
        for l in range(len(hidden_sizes)):
            feat_out = hidden_sizes[l]
            layers.append(torch.nn.Linear(
                in_features=feat_in,
                out_features=feat_out,
                bias=True
            ))
            layers.append(torch.nn.LeakyReLU())
            layers.append(torch.nn.Dropout(dropout))
            feat_in = feat_out

        layers.append(torch.nn.Linear(
            in_features=feat_in,
            out_features=num_classes,
            bias=True
        ))

        self.fc_module = torch.nn.Sequential(*layers)

    def forward(self, x):
        B, L, C, H, W = x.shape

        # (B, L, C, H, W)
        x = self.conv_module(x)
        # (B, L, CC, HH, WW)
        x = x.reshape(B*L, self.CC*self.HH*self.WW)
        # (BL, CCHHWW)
        x = self.fc_module(x)
        # (BL, num_classes)
        x = x.reshape(B, L, self.num_classes)

        return x


DEFAULT_CONV_CONFIG = [
    (32, 5, 5, True),
    (64, 5, 5, True),
    (128, 5, 5, True),
    (256, 5, 5, True),
    (512, 3, 3, True),
    (1024, 3, 3, False),
]
DEFAULT_FC_CONFIG = [256, 64]

if __name__ == "__main__":

    if len(sys.argv) != 2:
        print('Use this script to initialize a pre-defined model,')
        print('and save their initial weight to disk for training later')
        print('USAGE: python model_conv_config.py save_model_name')

    save_model_name = sys.argv[1]
    print(f'model name [{save_model_name}]')

    model = ConvPredictor(
        conv_module=SequentialCNN3DFrontTimePad(
            data_utils.NUM_CHANNEL, 90, 68,
            DEFAULT_CONV_CONFIG,
            use_batchnorm=True
        ),
        hidden_sizes=DEFAULT_FC_CONFIG,
        num_classes=data_def.NUM_CLASS
    )
    print(model)

    train_utils.init_model_save(
        model,
        save_model_name
    )
