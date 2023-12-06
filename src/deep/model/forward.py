import torch.nn as nn
import torch
import torchaudio
from .sudormrf import SuDORMRF
def Artifact_remove_network():
    model = SuDORMRF(out_channels=128,
                 in_channels=512,
                 num_blocks=4, # tiny-model
                 upsampling_depth=4,
                 enc_kernel_size=21,
                 enc_num_basis=512,
                 num_sources=1)
    return model
class Forward_Model(nn.Module):
    '''
    Special deep learning model, combined with non-deep learning model (frequency response etc)
    It can be either deep learning free or hybrid deep learning or Pure deep learning
    Normally, it will be smaller than reverse model
    '''
    def __init__(self, transfer_function=None):
        super().__init__()
        self.transfer_function = transfer_function
        self.model = Artifact_remove_network()
    def forward(self, x):
        if self.transfer_function is not None:
            b, a = self.transfer_function
            x = torchaudio.functional.filtfilt(x, b, a)
        x = self.model(x) + x
        return x