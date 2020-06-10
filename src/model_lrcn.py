import torch
import data_def
import sys
from model_seq_cnn import *
from model_trainer import train_model, load_trained
# import model_conv
import train_utils
import data_utils


class LRCNPredictor(torch.nn.Module):

    def __init__(
        self,
        conv_module,
        lstm_hidden_size,
        lstm_num_layer,
        num_classes,
        dropout=0.1
    ):
        """
        High level viz, where time goes from top to bottom:

        ```
        [x0] -> <CNN> -> [f0] -> <LSTM> -> [h0] -> <FC> -> [yhat0]
                  |                 =               |
        [x1] -> <CNN> -> [f1] -> <LSTM> -> [h1] -> <FC> -> [yhat1]
                  |                 =               |
                 ...               ...             ...
                  |                 =               |
        [xn] -> <CNN> -> [fn] -> <LSTM> -> [hn] -> <FC> -> [yhatn]
        ```

        - where <CNN> is a single conv module
        - where <LSTM> is a single recurrent module unrolled over time
        - where <FC> is a single fully connected layer

        Args:
        - conv_module (torch.nn.Module): a convolutional module acting as 
            feature extractor, must take in input of size (B, L, C, H, W) 
            and output of size (B, L, CC, HH, WW), must implement method
            get_output_shzpe() that returns (CC ,HH, WW)
        - lstm_hidden_size (int): hidden dimension of lstm layers
        - lstm_num_layers (int): number of lstm layers stacked together
        - num_classes (int): number of classes, output size
        """

        super(LRCNPredictor, self).__init__()
        self.lstm_hidden_size = lstm_hidden_size
        self.num_classes = num_classes

        self.conv_module = conv_module
        self.CC, self.HH, self.WW = self.conv_module.output_shape()
        # (B, L, CC, HH, WW)
        # flatten to (B, L, CC*HH*WW)

        lstm_input_dim = self.CC * self.HH * self.WW
        # (B, L, input_dim)
        self.lstm = torch.nn.LSTM(
            input_size=lstm_input_dim,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layer,
            bias=True,
            batch_first=True,  # so that input is (B, L, input_dim)
            dropout=0,  # dont use dropout between lstm layers,
            bidirectional=False
        )
        # output=(B, L, lstm_hidden_size)
        # h_n=(lstm_num_layers, B, lstm_hidden_size)
        # c_n=(lstm_num_layers, B, lstm_hidden_size)
        self.dropout = nn.Dropout(dropout)

        self.fc = torch.nn.Linear(
            in_features=lstm_hidden_size,
            out_features=num_classes,
            bias=True
        )

    def forward(self, x):
        """
        Args:
        - x (torch.Tensor): of shape (B, L, C, H, W)
        - lstm_state (tuple, optional): (h_0, c_0) of tensors, initial state
            for lstm forward. Defaults to None.

        Returns:
        - torch.Tensor: output of shape (B, L, num_classes), a tuple
            of lstm hidden state
        """

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


DEFAULT_CONV_CONFIG = [
    (32, 5, True),
    (64, 5, True),
    (128, 5, True),
    (256, 5, True),
    (512, 3, True),
    (1024, 3, False),
]
DEFAULT_LSTM_HIDDEN_SIZE = 500
DEFAULT_LSTM_NUM_LAYER = 2

if __name__ == "__main__":
    conv_module = SequentialCNN(
        data_utils.NUM_CHANNEL, 90, 68,
        DEFAULT_CONV_CONFIG,
        use_batchnorm=True
    )
    model_lrcn = LRCNPredictor(
        conv_module,
        DEFAULT_LSTM_HIDDEN_SIZE,
        DEFAULT_LSTM_NUM_LAYER,
        data_def.NUM_CLASS
    )
    print(model_lrcn)
    train_utils.init_model_save(model_lrcn, sys.argv[1])
