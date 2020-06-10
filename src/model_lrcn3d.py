import torch
import data_def
import sys
from model_seq_cnn import *
from model_trainer import train_model, load_trained
import model_conv
import train_utils
import data_utils


class LRCN3DPredictor(torch.nn.Module):

    def __init__(
        self,
        conv_module,
        lstm_hidden_size,
        fc_hidden_size,
        num_classes,
        dropout=0.1
    ):
        super(LRCN3DPredictor, self).__init__()
        self.num_classes = num_classes
        self.lstm_hidden_size = lstm_hidden_size

        self.conv_module = conv_module
        self.CC, self.HH, self.WW = self.conv_module.output_shape()

        lstm_input_dim = self.CC * self.HH * self.WW
        # (B, L, input_dim)
        self.lstm = torch.nn.LSTM(
            input_size=lstm_input_dim,
            hidden_size=lstm_hidden_size,
            num_layers=1,
            bias=True,
            batch_first=True,  # so that input is (B, L, input_dim)
            dropout=0,  # dont use dropout between lstm layers,
            bidirectional=False
        )
        self.dropout = nn.Dropout(dropout)

        self.fc = torch.nn.Linear(
            in_features=lstm_hidden_size,
            out_features=num_classes,
            bias=True,
        )

    def forward(self, x):

        B, L, C, H, W = x.shape
        # x=(B, L, C, H, W)
        x = self.conv_module(x)
        # x=(B, L, CC, HH, WW)
        x = x.reshape(B, L, self.CC * self.HH * self.WW)
        # x=(B, L, CC * HH * WW)
        x, _ = self.lstm(x)
        # x=(B, L, lstm_hidden_size)
        x = x.reshape(B * L, self.lstm_hidden_size)
        # x=(B * L, lstm_hidden_size)
        x = self.dropout(x)
        x = self.fc(x)
        # x=(B * L, num_classes)
        x = x.reshape(B, L, self.num_classes)
        # x=(B, L, num_classes)

        return x


DEFAULT_LSTM_HIDDEN_SIZE = 1000
DEFAULT_FC_SIZE = 100

if __name__ == "__main__":
    conv_module = SequentialCNN3DFrontTimePad(
        data_utils.NUM_CHANNEL, 90, 68,
        model_conv.DEFAULT_CONV_CONFIG,
        use_batchnorm=True
    )
    model_lrcn = LRCN3DPredictor(
        conv_module,
        DEFAULT_LSTM_HIDDEN_SIZE,
        DEFAULT_FC_SIZE,
        data_def.NUM_CLASS
    )
    print(model_lrcn)
    train_utils.init_model_save(model_lrcn, sys.argv[1])
