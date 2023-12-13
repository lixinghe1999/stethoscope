from .loss import snr, sisnr, rmse, lsd, psnr, get_loss
import torch
from tqdm import tqdm
class Reverse(torch.nn.Module):
    def __init__(self, model):
        super(Reverse, self).__init__()
        self.reverse_model = model()
        self.optimizer = torch.optim.Adam(params=self.reverse_model.parameters(), lr=0.0001)

    def Reverse(self, z):
        x = self.reverse_model(z)
        return x
    def run_epoch_train(self, small_train, large_train, device):
        loss_list = []
        pbar = tqdm(small_train)
        for sample in pbar:
            audio = sample['audio'].to(device); reference = sample['reference'].to(device);
            mix = sample['mix'].to(device) if 'mix' in sample else None
            audio = audio.unsqueeze(1); reference = reference.unsqueeze(1)

            self.optimizer.zero_grad()
            est_reference = self.Reverse(audio)
            loss = get_loss(est_reference, reference, mix)   
            loss.backward() 
            self.optimizer.step()
            loss_list.append(loss.item())
            pbar.set_description("loss:{:.4f}".format(loss.item()))
        return loss_list
    def run_epoch_test(self, test_loader, device):
        loss_list = []
        for sample in (test_loader):
            audio = sample['audio'].to(device); reference = sample['reference'].to(device)
            audio = audio.unsqueeze(1); reference = reference.unsqueeze(1)
            est_reference = self.Reverse(audio)
            # est_reference = audio
            if est_reference.shape[1] == 1:
                est_reference = est_reference.squeeze(1)
                reference = reference.squeeze(1)
               
            else:
                snr1 = -sisnr(est_reference[:,0,:], reference[:,0,:]).item()
                snr2 = -sisnr(est_reference[:,1,:], reference[:,0,:]).item()
                if snr1 > snr2:
                    est_reference = est_reference[:,0,:]
                else:   
                    est_reference = est_reference[:,1,:]
            metric = [-sisnr(est_reference, reference).item(), -snr(est_reference, reference).item(), rmse(est_reference, reference).item(),
                       psnr(est_reference, reference).item(), lsd(est_reference, reference).item()]
            
            loss_list.append(metric)
        return loss_list