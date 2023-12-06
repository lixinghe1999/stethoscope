from sudormrf import SuDORMRF
from original_convtasnet import TasNet
from sepformer import Sepformer
from dcc_tf import Net
from onedim_unet import UNet
from twodim_unet import Two_Stage_Unet

import torch
import time
from tqdm import tqdm
def latency_measure(inp, model, device):
    model.eval()
    inp = inp.to(device)
    model = model.to(device)
    t_start = time.time()
    for i in tqdm(range(50)):
        out = model(inp)
    return 100 / (time.time() - t_start) 
inp = torch.randn(1, 1, 4000)
device = torch.device('cuda')
# print('SuDORMRF', latency_measure(inp, SuDORMRF(), device))
# print('TasNet', latency_measure(inp, TasNet(), device))
# print('Sepformer', latency_measure(inp, Sepformer(), device))
# print('DCC-TF', latency_measure(inp, Net(), device))
# print('UNet', latency_measure(inp, UNet(), device))
print('Two_Stage_Unet', latency_measure(inp, Two_Stage_Unet(), device))
