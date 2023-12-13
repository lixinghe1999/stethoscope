import torch
from tqdm import tqdm
from .loss import sisnr, snr, lsd, rmse, get_loss
from .forward import Forward_Model
import itertools
class Forward_Reverse(torch.nn.Module):
    def __init__(self, model, LR=0.0001):
        super(Forward_Reverse, self).__init__() 
        self.forward_model = Forward_Model()
        self.reverse_model = model()
        self.optimizer_1 = torch.optim.Adam(itertools.chain(self.forward_model.parameters(), self.reverse_model.parameters()), lr=LR)
        self.optimizer_2 = torch.optim.Adam(self.reverse_model.parameters(), lr=LR)
    def run_epoch_train(self, small_train, large_train, device):
        loss_forward = []
        pbar = tqdm(small_train)
        for sample in pbar: # with reference
            z = sample['audio'].to(device).unsqueeze(1); x = sample['reference'].to(device).unsqueeze(1)
            self.optimizer_1.zero_grad()
            est_z = self.forward_model(x)
            est_x = self.reverse_model(z)
            loss = sisnr(est_z, z) + sisnr(est_x, x)
            loss.backward() 
            self.optimizer_1.step()
            pbar.set_description("loss:{:.4f}".format(loss.item()))
            loss_forward.append(loss.item())

        pbar = tqdm(large_train)
        for sample in pbar:
            x = sample['reference'].to(device).unsqueeze(1)
            self.optimizer_2.zero_grad()
            with torch.no_grad():
                est_z = self.forward_model(x)
            est_x = self.reverse_model(est_z)
            loss = sisnr(est_x, x)   
            loss.backward() 
            self.optimizer_2.step()
            pbar.set_description("loss:{:.4f}".format(loss.item()))
        return loss_forward
    def run_epoch_test(self, test_loader, device):
        loss_list = []
        for sample in (test_loader):
            z = sample['audio'].to(device).unsqueeze(1); x = sample['reference'].to(device).unsqueeze(1)
            est_x = self.reverse_model(z)
            est_x = est_x.squeeze(1)
            x = x.squeeze(1)
            metric = [-sisnr(est_x, x).item(), -snr(est_x, x).item(), 
                      rmse(est_x, x).item(), lsd(est_x, x).item()]
            loss_list.append(metric)
        return loss_list