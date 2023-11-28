import torch
from tqdm import tqdm
from .loss import sisnr
from .forward import Forward_Model
import itertools
class Forward_Reverse:
    def __init__(self, model, LR=0.0001):
        super(Forward_Reverse, self).__init__() 
        self.forward_model = Forward_Model()
        self.reverse_model = model()
        self.optimizer_1 = torch.optim.Adam(itertools.chain(self.forward_model.parameters(), self.reverse_model.parameters()), lr=LR)
        self.optimizer_2 = torch.optim.Adam(self.reverse_model.parameters(), lr=LR)
    def run_epoch_train(self, small_train, large_train, device):
        loss_forward = []
        for sample in tqdm(small_train): # with reference
            z = sample['audio'].to(device).unsqueeze(1); x = sample['reference'].to(device).unsqueeze(1)
            self.optimizer_1.zero_grad()
            est_z = self.forward_model(x)
            est_x = self.reverse_model(z)
            loss = sisnr(est_z, z) + sisnr(est_x, x)
            loss.backward() 
            self.optimizer_1.step()
            loss_forward.append(loss.item())

        loss_reverse = []
        for sample in tqdm(large_train):
            x = sample['reference'].to(device)
            self.optimizer_2.zero_grad()
            with torch.no_grad():
                est_z = self.forward_model(x)
            est_x = self.reverse_model(est_z)
            loss = sisnr(est_x, x)   
            loss.backward() 
            self.optimizer_2.step()
            loss_reverse.append(loss.item())
        
        return loss_forward, loss_reverse
    def run_epoch_test(self, test_loader, device):
        loss_list = []
        for sample in tqdm(test_loader):
            z = sample['audio'].to(device).unsqueeze(1); x = sample['reference'].to(device).unsqueeze(1)
            est_reference = self.Reverse(z)
            loss = - sisnr(est_reference.squeeze(1), x.squeeze(1))
            loss_list.append(loss.item())
        return loss_list