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
def MultiResolutionSTFTLoss(x, y):
    fft_size = [1024, 2048, 512]
    hop_size = [120, 240, 50]
    win_length = [600, 1200, 240]
    loss = 0
    for fft, hop, win in zip(fft_size, hop_size, win_length):
        window = torch.hann_window(win).to(x.device)
        x_stft = torch.stft(x,  fft, hop, win, window, return_complex=True)
        x_mag = torch.sqrt(
            torch.clamp((x_stft.real**2) + (x_stft.imag**2), min=1e-8)
        )
        x_phs = torch.angle(x_stft)

        y_stft = torch.stft(y,  fft, hop, win, window, return_complex=True)
        y_mag = torch.sqrt(
            torch.clamp((y_stft.real**2) + (y_stft.imag**2), min=1e-8)
        )
        y_phs = torch.angle(y_stft)
        loss += Spectral_Loss(x_mag, y_mag) + torch.nn.functional.mse_loss(x_phs, y_phs)
    return loss
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
def snr(x, s, eps=1e-8, vad=1):
    """
    calculate training loss
    input:
          x: separated signal, N x S tensor
          s: reference signal, N x S tensor
    Return:
          snr: N tensor
    """

    def l2norm(mat, keepdim=False):
        return torch.norm(mat, dim=-1, keepdim=keepdim)

    if x.shape != s.shape:
        raise RuntimeError(
            "Dimention mismatch when calculate si-snr, {} vs {}".format(
                x.shape, s.shape))
    x_zm = x - torch.mean(x, dim=-1, keepdim=True)
    s_zm = s - torch.mean(s, dim=-1, keepdim=True)
    return -20 * torch.log10(eps + l2norm(s_zm * vad) / (l2norm((x_zm - s_zm)*vad) + eps)).mean()
def lsd(x, s, eps=1e-8, vad=1):
    window = torch.hann_window(256).to(x.device)
    x_stft = torch.stft(x, 256, 120, 120, window, return_complex=True)
    x_mag = torch.sqrt(
        torch.clamp((x_stft.real**2) + (x_stft.imag**2), min=1e-8)
    )
    s_stft = torch.stft(s, 256, 120, 120, window, return_complex=True)
    s_mag = torch.sqrt(
        torch.clamp((s_stft.real**2) + (s_stft.imag**2), min=1e-8)
    )
    lsd = torch.log10(x_mag **2 / ((s_mag + eps) ** 2) + eps) ** 2 * vad
    lsd = torch.mean(torch.mean(lsd, axis=-1) ** 0.5, axis=-1)
    return lsd

def get_loss(est_audio, reference):
    loss = 0
    loss += sisnr(est_audio, reference) * 0.9
    loss += snr(est_audio, reference) * 0.1
    loss += MultiResolutionSTFTLoss(est_audio, reference) 
    return loss


