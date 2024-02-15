import torch
from torch import nn
import speechbrain as sb

from src.module import (
    FTDNN,
    OutputLayer
)

class FTDNNAcoustic(nn.Module):
    def __init__(self, num_senones=6112, device_name='cpu'):
        super(FTDNNAcoustic, self).__init__()
        self.ftdnn        = FTDNN(device_name=device_name)
        self.output_layer = OutputLayer(256, 1536, 256, num_senones)

    def forward(self, x):
        x = self.ftdnn(x)
        x = self.output_layer(x)
        return x