'''
unpaired dataset only
1. paired dataset: x_p, z_p
2. unpaired dataset: x_u
train: x_u, z_p
test: x_p, z_p
'''
import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm
from .loss import sisnr
import random
from tqdm.auto import tqdm
import itertools
class SNConv2DLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.nn = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)),
            nn.LeakyReLU()
        )

    def forward(self, x):
        return self.nn(x)
class SNLinearLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.nn = nn.Sequential(
            spectral_norm(nn.Linear(in_channels, out_channels)),
            nn.LeakyReLU()
        )

    def forward(self, x):
        return self.nn(x)
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv_layers = nn.Sequential(
            SNConv2DLayer(2, 15, 5),
            SNConv2DLayer(15, 25, 7),
            SNConv2DLayer(25, 40, 9),
            SNConv2DLayer(40, 50, 11)
        )

        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.linear_layers = nn.Sequential(
            SNLinearLayer(50, 50),
            SNLinearLayer(50, 10),
            SNLinearLayer(10, 1),
        )

    def forward(self, x):
        # TODO: multiple-resolution
        x = torch.stft(x.squeeze(1), n_fft=512, hop_length=120, win_length=512, window=torch.hann_window(512).to(x.device),
                        return_complex=False).permute(0, 3, 1, 2)  # [batch_size, 2, 257, X]
        o = self.conv_layers(x)
        o = self.gap(o)  # [batch_size, channels, 1, 1]
        o = o.reshape((o.shape[0], o.shape[1]))  # [batch_size, channels]
        o = self.linear_layers(o)  # [batch_size, 1]
        return o
class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0):
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(input)

    def __call__(self, input, target_is_real):
        target_tensor = self.get_target_tensor(input, target_is_real)
        return self.loss(input, target_tensor)

class CycleGAN(torch.nn.Module):
    def __init__(self, model, LR=0.0001):
        super(CycleGAN, self).__init__()
        self.forward_model = model()
        self.reverse_model = model()

        self.forward_discriminator = Discriminator()
        self.reverse_discriminator = Discriminator()

        # define loss functions
        self.criterionGAN = GANLoss()
        self.criterionCycle = torch.nn.L1Loss()
        self.criterionIdt = torch.nn.L1Loss()
        # initialize optimizers
        self.optimizer_G = torch.optim.Adam(itertools.chain(self.forward_model.parameters(), self.reverse_model.parameters()),
                                            lr=LR)
        self.optimizer_D = torch.optim.Adam(itertools.chain(self.forward_discriminator.parameters(), self.reverse_discriminator.parameters()),
                                            lr=LR)
        self.optimizers = []
        self.optimizers.append(self.optimizer_G)
        self.optimizers.append(self.optimizer_D)    
    def run_epoch_train(self, train_loader, device):
        loss_list = []
        pbar = tqdm(train_loader)
        for sample in pbar:
            z = sample['audio'].to(device).unsqueeze(1); x = sample['reference'].to(device).unsqueeze(1)
            # forward
            fake_z, rec_x, fake_x, rec_z = self.forward_cycle(x, z)
            # G_A and G_B
            self.set_requires_grad([self.forward_discriminator, self.reverse_discriminator], False)
            self.optimizer_G.zero_grad()
            loss_g = self.backward_G(x, z, fake_x, fake_z, rec_x, rec_z)
            self.optimizer_G.step()
            # D_A and D_B
            self.set_requires_grad([self.forward_discriminator, self.reverse_discriminator], True)
            self.optimizer_D.zero_grad()
            loss_d = self.backward_D(x, z, fake_x, fake_z)
            self.optimizer_D.step()
            loss_list.append(loss_g)
            pbar.set_description("loss_g:{:.4f}, loss_d:{:.4f}".format(loss_g, loss_d))

        return loss_list
    def run_epoch_test(self, test_loader, device):
        loss_list = []
        for sample in tqdm(test_loader):
            z = sample['audio'].to(device).unsqueeze(1); x = sample['reference'].to(device).unsqueeze(1)
            est_x = self.reverse_model(z)
            loss = - sisnr(est_x.squeeze(1), x.squeeze(1))
            loss_list.append([loss.item()])
        return loss_list
    def forward_cycle(self, x, z):
        # forward: x -> z, reverse: z -> x
        fake_z = self.forward_model(x)
        rec_x = self.reverse_model(fake_z)

        fake_x = self.reverse_model(z)
        rec_z = self.forward_model(fake_x)
        return fake_z, rec_x, fake_x, rec_z

    def backward_D_basic(self, netD, real, fake):
        # Real
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Combined loss
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        # backward
        loss_D.backward()
        return loss_D.item()
    def backward_D(self, x, z, fake_x, fake_z):
        p = random.uniform(0, 1)
        fake_z = fake_z if p < 0.5 else x
        loss_D_forward = self.backward_D_basic(self.forward_discriminator, z, fake_z)

        p = random.uniform(0, 1)
        fake_x = fake_x if p < 0.5 else z
        loss_D_reverse = self.backward_D_basic(self.reverse_discriminator, x, fake_x)
        return loss_D_forward + loss_D_reverse
    def backward_G(self, x, z, fake_x, fake_z, rec_x, rec_z):
        lambda_idt = 1; lambda_A = 0.5; lambda_B = 0.5
        # Identity loss
        if lambda_idt > 0:
            # G_A should be identity if real_B is fed.
            idt_z = self.forward_model(z)
            loss_idt_forward = self.criterionIdt(idt_z, z) * lambda_B * lambda_idt
            # G_B should be identity if real_A is fed.
            idt_x = self.reverse_model(x)
            loss_idt_reverse = self.criterionIdt(idt_x, x) * lambda_A * lambda_idt
        else:
            loss_idt_forward = 0
            loss_idt_reverse = 0

        # GAN loss D_A(G_A(A))
        loss_G_forward = self.criterionGAN(self.forward_discriminator(fake_z), True)
        # GAN loss D_B(G_B(B))
        loss_G_reverse = self.criterionGAN(self.reverse_discriminator(fake_x), True)
        # Forward cycle loss
        loss_cycle_forward = self.criterionCycle(x, rec_x) * lambda_A
        # Backward cycle loss
        loss_cycle_reverse = self.criterionCycle(z, rec_z) * lambda_B
        # combined loss
        loss_G = loss_G_forward + loss_G_reverse + loss_cycle_forward + loss_cycle_reverse + loss_idt_forward + loss_idt_reverse
        loss_G.backward()
        return loss_G.item()
    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad
       