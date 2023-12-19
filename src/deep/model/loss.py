import torch
import itertools
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
    fft_size = [128, 256, 512]
    hop_size = [64, 128, 256]
    win_length = [128, 256, 512]
    loss = 0
    for fft, hop, win in zip(fft_size, hop_size, win_length):
        window = torch.hann_window(win).to(x.device)
        x_stft = torch.stft(x,  fft, hop, win, window, return_complex=True)
        x_mag = torch.sqrt(
            torch.clamp((x_stft.real**2) + (x_stft.imag**2), min=1e-8))
        y_stft = torch.stft(y,  fft, hop, win, window, return_complex=True)
        y_mag = torch.sqrt(
            torch.clamp((y_stft.real**2) + (y_stft.imag**2), min=1e-8))
        loss += Spectral_Loss(x_mag, y_mag)
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
    window_size, hop_size = 512, 256
    window = torch.hann_window(window_size).to(x.device)
    x_stft = torch.stft(x, window_size, hop_size, window_size, window, return_complex=True)
    x_mag = torch.sqrt(torch.clamp((x_stft.real**2) + (x_stft.imag**2), min=1e-8))
    s_stft = torch.stft(s, window_size, hop_size, window_size, window, return_complex=True)
    s_mag = torch.sqrt(torch.clamp((s_stft.real**2) + (s_stft.imag**2), min=1e-8))

    lsd = torch.log10(x_mag **2 / ((s_mag + eps) ** 2) + eps) ** 2 * vad
    lsd = torch.mean(torch.mean(lsd, axis=-1) ** 0.5)
    return lsd
def rmse(x, s, eps=1e-8, vad=1):
    return torch.mean(torch.mean(((x - s) * vad) ** 2, axis=-1)**0.5)
def psnr(x, s, eps=1e-8, vad=1):
    window_size, hop_size = 512, 256
    window = torch.hann_window(window_size).to(x.device)
    x_stft = torch.stft(x, window_size, hop_size, window_size, window, return_complex=True)
    x_mag = torch.abs(x_stft)
    s_stft = torch.stft(s, window_size, hop_size, window_size, window, return_complex=True)
    s_mag = torch.abs(s_stft)

    mse = torch.mean((x_mag - s_mag) ** 2, dim=(1, 2))
    max_mag = torch.max(s_mag.reshape(s_mag.shape[0], -1), dim=-1)[0]
    psnr = torch.mean(10 * torch.log10(max_mag/mse))
    return psnr
def get_loss(est_audio, reference, mix):
    if est_audio.shape[1] == 1:
        est_audio = est_audio.squeeze(1)
        reference = reference.squeeze(1)
        loss = 0
        # loss += sisnr(est_audio, reference) * 0.9
        # loss += snr(est_audio, reference) * 0.1
        loss += MultiResolutionSTFTLoss(est_audio, reference) 
    else:
        bs, n, _ = est_audio.shape
        func = PermInvariantSISDR(bs, n_sources=n)
        loss = func(est_audio, mix)
    return loss
class PermInvariantSISDR(torch.nn.Module):
    """!
    Class for SISDR computation between reconstructed signals and
    target wavs by also regulating it with learned target masks."""

    def __init__(self,
                 batch_size=None,
                 zero_mean=False,
                 n_sources=None,
                 backward_loss=True,
                 improvement=False,
                 return_individual_results=False):
        """
        Initialization for the results and torch tensors that might
        be used afterwards

        :param batch_size: The number of the samples in each batch
        :param zero_mean: If you want to perform zero-mean across
        last dimension (time dim) of the signals before SDR computation
        """
        super().__init__()
        self.bs = batch_size
        self.perform_zero_mean = zero_mean
        self.backward_loss = backward_loss
        self.permutations = list(itertools.permutations(
            torch.arange(n_sources)))
        self.permutations_tensor = torch.LongTensor(self.permutations)
        self.improvement = improvement
        self.n_sources = n_sources
        self.return_individual_results = return_individual_results

    def normalize_input(self, pr_batch, t_batch, initial_mixtures=None):
        min_len = min(pr_batch.shape[-1],
                      t_batch.shape[-1])
        if initial_mixtures is not None:
            min_len = min(min_len, initial_mixtures.shape[-1])
            initial_mixtures = initial_mixtures[:, :, :min_len]
        pr_batch = pr_batch[:, :, :min_len]
        t_batch = t_batch[:, :, :min_len]

        if self.perform_zero_mean:
            pr_batch = pr_batch - torch.mean(
                pr_batch, dim=-1, keepdim=True)
            t_batch = t_batch - torch.mean(
                t_batch, dim=-1, keepdim=True)
            if initial_mixtures is not None:
                initial_mixtures = initial_mixtures - torch.mean(
                    initial_mixtures, dim=-1, keepdim=True)
        return pr_batch, t_batch, initial_mixtures

    @staticmethod
    def dot(x, y):
        return torch.sum(x * y, dim=-1, keepdim=True)

    def compute_permuted_sisnrs(self,
                                permuted_pr_batch,
                                t_batch,
                                t_t_diag, eps=10e-8):
        s_t = (self.dot(permuted_pr_batch, t_batch) /
               (t_t_diag + eps) * t_batch)
        e_t = permuted_pr_batch - s_t
        sisnrs = 10 * torch.log10(self.dot(s_t, s_t) /
                                  (self.dot(e_t, e_t) + eps))
        return sisnrs

    def compute_sisnr(self,
                      pr_batch,
                      t_batch,
                      initial_mixtures=None,
                      eps=10e-8):

        t_t_diag = self.dot(t_batch, t_batch)

        sisnr_l = []
        for perm in self.permutations:
            permuted_pr_batch = pr_batch[:, perm, :]
            sisnr = self.compute_permuted_sisnrs(permuted_pr_batch,
                                                 t_batch,
                                                 t_t_diag, eps=eps)
            sisnr_l.append(sisnr)
        all_sisnrs = torch.cat(sisnr_l, -1)
        best_sisdr, best_perm_ind = torch.max(all_sisnrs.mean(-2), -1)

        if self.improvement:
            initial_mix = initial_mixtures.repeat(1, self.n_sources, 1)
            base_sisdr = self.compute_permuted_sisnrs(initial_mix,
                                                      t_batch,
                                                      t_t_diag, eps=eps)
            best_sisdr -= base_sisdr.mean()

        if not self.return_individual_results:
            best_sisdr = best_sisdr.mean()

        if self.backward_loss:
            return -best_sisdr, best_perm_ind
        return best_sisdr, best_perm_ind

    def forward(self,
                pr_batch,
                t_batch,
                eps=1e-9,
                initial_mixtures=None,
                return_best_permutation=False):
        """!
        :param pr_batch: Reconstructed wavs: Torch Tensors of size:
                         batch_size x self.n_sources x length_of_wavs
        :param t_batch: Target wavs: Torch Tensors of size:
                        batch_size x self.n_sources x length_of_wavs
        :param eps: Numerical stability constant.
        :param initial_mixtures: Initial Mixtures for SISDRi: Torch Tensor
                                 of size: batch_size x 1 x length_of_wavs

        :returns results_buffer Buffer for loading the results directly
                 to gpu and not having to reconstruct the results matrix: Torch
                 Tensor of size: batch_size x 1
        """
        pr_batch, t_batch, initial_mixtures = self.normalize_input(
            pr_batch, t_batch, initial_mixtures=initial_mixtures)

        sisnr_l, best_perm_ind = self.compute_sisnr(
            pr_batch, t_batch, eps=eps,
            initial_mixtures=initial_mixtures)

        if return_best_permutation:
            best_permutations = self.permutations_tensor[best_perm_ind]
            return sisnr_l, best_permutations
        else:
            return sisnr_l


