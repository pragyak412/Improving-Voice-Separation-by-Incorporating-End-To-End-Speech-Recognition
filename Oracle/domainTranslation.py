from torch import nn
from torch.nn import functional as Func

import config


class Block(nn.Module):

    def __init__(self, filter_size, dilation, num_filters, input_filters, padding):
        super(Block, self).__init__()

        self.conv = nn.Conv1d(
            in_channels=input_filters,
            out_channels=num_filters,
            kernel_size=filter_size,
            dilation=dilation,
            padding=padding,
        )

        self.bn = nn.BatchNorm1d(num_filters)
        self.act = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)

        return x


class DomainTranslation(nn.Module):

    def __init__(self):
        super(DomainTranslation, self).__init__()
        self.params = nn.Sequential(
            Block(filter_size=7, dilation=1, num_filters=256, input_filters=config.num_features,
                  padding=3),
            Block(filter_size=5, dilation=1, num_filters=256, input_filters=256, padding=2),
            Block(filter_size=5, dilation=2, num_filters=256, input_filters=256, padding=4),
            Block(filter_size=5, dilation=4, num_filters=256, input_filters=256, padding=8),
            Block(filter_size=5, dilation=8, num_filters=256, input_filters=256, padding=16),
            Block(filter_size=5, dilation=16, num_filters=256, input_filters=256, padding=32),
        )

    def forward(self, x):
        batch_size, num_speakers, num_frames, num_features = x.shape  # [M, C, X, N]
        x = x.view([batch_size * num_speakers, num_frames, num_features]).transpose(1, 2)  # [ M*C, N, X]
        x = self.params(x)  # [M*C, N, X]
        x = Func.interpolate(x, 2399).view([batch_size, num_speakers * 256, 2399])  # [M, N, 2399]

        return x
