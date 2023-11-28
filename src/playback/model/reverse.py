from .loss import sisnr
import torch
from tqdm import tqdm
class Reverse(torch.nn.Module):
    def __init__(self, model):
        super(Reverse, self).__init__()
        self.reverse_model = model()
        self.optimizer = torch.optim.Adam(params=self.reverse_model, lr=0.0001)

    def Reverse(self, z):
        x = self.reverse_model(z)
        return x
    def run_epoch_train(self, train_loader, device):
        loss_list = []
        for sample in tqdm(train_loader):
            audio = sample['audio'].to(device); reference = sample['reference'].to(device)
            self.optimizer.zero_grad()
            est_reference = self.Reverse(audio.unsqueeze(1))
            loss = sisnr(est_reference.squeeze(1), reference.squeeze(1))   
            loss.backward() 
            self.optimizer.step()
            loss_list.append(loss.item())
        return loss_list
    def run_epoch_test(self, test_loader, device):
        loss_list = []
        for sample in tqdm(test_loader):
            audio = sample['audio'].to(device); reference = sample['reference'].to(device)
            est_reference = self.Reverse(audio.unsqueeze(1))
            loss = - sisnr(est_reference.squeeze(1), reference.squeeze(1))
            loss_list.append([loss.item()])
        return loss_list