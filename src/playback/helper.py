import torch

def Spectral_Loss(x_mag, y_mag, vad=1):
    """Calculate forward propagation.
          Args:
              x_mag (Tensor): Magnitude spectrogram of predicted signal (B, #frames, #freq_bins).
              y_mag (Tensor): Magnitude spectrogram of groundtruth signal (B, #frames, #freq_bins).
              vad (Tensor): VAD of groundtruth signal (B, #frames, #freq_bins).
          Returns:
              Tensor: Spectral convergence loss value.
          """
    x_mag = torch.clamp(x_mag, min=1e-7)
    y_mag = torch.clamp(y_mag, min=1e-7)
    spectral_convergenge_loss =  torch.norm(vad * (y_mag - x_mag), p="fro") / torch.norm(y_mag, p="fro")
    log_stft_magnitude = (vad * (torch.log(y_mag) - torch.log(x_mag))).abs().mean()
    return 0.5 * spectral_convergenge_loss + 0.5 * log_stft_magnitude
def sisnr(x, s, eps=1e-8, vad=1):
    """
    calculate training loss
    input:
          x: separated signal, N x S tensor
          s: reference signal, N x S tensor
    Return:
          sisnr: N tensor
    """

    def l2norm(mat, keepdim=False):
        return torch.norm(mat, dim=-1, keepdim=keepdim)

    if x.shape != s.shape:
        raise RuntimeError(
            "Dimention mismatch when calculate si-snr, {} vs {}".format(
                x.shape, s.shape))
    x_zm = x - torch.mean(x, dim=-1, keepdim=True)
    s_zm = s - torch.mean(s, dim=-1, keepdim=True)
    t = torch.sum(
        x_zm * s_zm, dim=-1,
        keepdim=True) * s_zm / (l2norm(s_zm, keepdim=True)**2 + eps)
    return -20 * torch.log10(eps + l2norm(t * vad) / (l2norm((x_zm - t)*vad) + eps)).mean()
def train(model, sample, optimizer, device='cuda',):
    audio = sample['audio'].to(device); reference = sample['reference'].to(device)
    optimizer.zero_grad()
    est_audio = model(audio)
    loss = sisnr(est_audio, reference.squeeze(1))   
    loss.backward() 
    optimizer.step()
    return loss.item()

def test(model, sample, device='cuda'):
    audio = sample['audio'].to(device); reference = sample['reference'].to(device)
    est_audio = model(audio.unsqueeze(1))
    est_audio = est_audio.cpu().numpy().squeeze(1)
    reference = reference.cpu().numpy()
    print(est_audio.shape, reference.shape)
    return reference, est_audio