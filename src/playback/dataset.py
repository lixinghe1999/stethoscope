import math
import torchaudio
import librosa
import os
import warnings
import numpy as np
warnings.filterwarnings("ignore")
def tailor_dB_FS(y, target_dB_FS=-25, eps=1e-6):
    rms = (y ** 2).mean() ** 0.5
    scaler = 10 ** (target_dB_FS / 20) / (rms + eps)
    y *= scaler
    return y, rms, scaler
def snr_mix(noise_y, clean_y, snr, target_dB_FS, rir=None, eps=1e-6):
        """
        Args:
            noise_y: 噪声
            clean_y: 纯净语音
            snr (int): 信噪比
            target_dB_FS (int):
            target_dB_FS_floating_value (int):
            rir: room impulse response, None 或 np.array
            eps: eps
        Returns:
            (noisy_y，clean_y)
        """
        if rir is not None:
            clean_y = torchaudio.functional.fftconvolve(clean_y, rir)[:len(clean_y)]
        clean_rms = (clean_y ** 2).mean() ** 0.5
        noise_rms = (noise_y ** 2).mean() ** 0.5
        snr_scalar = clean_rms / (10 ** (snr / 20)) / (noise_rms + eps)
        noise_y *= snr_scalar
        noisy_y = clean_y + noise_y
        # Randomly select RMS value of dBFS between -15 dBFS and -35 dBFS and normalize noisy speech with that value
        noisy_target_dB_FS = target_dB_FS
        # 使用 noisy 的 rms 放缩音频
        noisy_y, _, noisy_scalar = tailor_dB_FS(noisy_y, noisy_target_dB_FS)
        clean_y *= noisy_scalar
        noise_y *= noisy_scalar
        # 合成带噪语音的时候可能会 clipping，虽然极少
        # 对 noisy, clean_y, noise_y 稍微进行调整
        if (noise_y.max() > 0.999).any():
            noisy_y_scalar = (noisy_y).abs().max() / (0.99 - eps)  # 相当于除以 1
            noisy_y = noisy_y / noisy_y_scalar
            clean_y = clean_y / noisy_y_scalar
            noise_y = noise_y / noisy_y_scalar
        return noisy_y, clean_y, noise_y, noisy_scalar
class BaseDataset:
    def __init__(self, files=None, length_info=None, pad=True, sample_rate=16000, length=5, stride=3):
        """
        files should be a list [(file, length)]
        """
        self.files = files
        self.length_info = length_info
        self.num_examples = []
        self.sample_rate = sample_rate
        self.length = length
        self.stride = stride
        self.pad = True
        for file, file_length in zip(files, length_info):
            if self.length is None:
                examples = 1
            elif file_length < self.length:
                examples = 1 if pad else 0
            elif pad:
                examples = int(math.ceil((file_length - self.length) / self.stride) + 1)
            else:
                examples = (file_length - self.length) // (self.stride) + 1
            self.num_examples.append(examples)
            self.dataset_len = sum(self.num_examples)
    def __len__(self):
        return self.dataset_len
    def __getitem__(self, index):
        for file, samples, examples in zip(self.files, self.length_info, self.num_examples):
            if index >= examples:
                index -= examples
                continue
            # data, _  = torchaudio.load(file, frame_offset =self.stride * index*self.sample_rate, 
            # num_frames=self.length * self.sample_rate, backend='ffmpeg')
            data, _  = librosa.load(file, sr=self.sample_rate, offset =self.stride * index, duration=self.length, mono=True)
            if data.shape[-1] < (self.sample_rate * self.length):
                pad_before = np.random.randint((self.sample_rate * self.length) - data.shape[-1])
                pad_after = (self.sample_rate *self.length) - data.shape[-1] - pad_before
                # data = torch.nn.functional.pad(data, (pad_before, pad_after, 0, 0 ), 'constant')
                data = np.pad(data, (pad_before, pad_after), 'constant')
            return data
class StethDataset:
    def __init__(self, directory = 'normalized_dataset', dataset='CHSC'):
        heart_datasets = {
            'CHSC': ['set_a', 'set_b'],
            'PhysioNet': ['training-a', 'training-b', 'training-c', 'training-d', 'training-e', 'training-f', 'validation'],
            # 'Thinklabs':['wav'],
            'CirCor': ['training_data'],
            # 'ephongram': ['WAV']
        }
        lung_datasets = {
            'Respiratory': ['audio_and_txt_files'],
            'lungsound': ['Audio Files']
        }
        audio_files = []
        length_info = []
        for dataset in heart_datasets:
            for choose_set in heart_datasets[dataset]:
                sub_dir = os.path.join(directory, dataset, choose_set)
                if os.path.exists(sub_dir) is False:
                    continue
                files = os.listdir(sub_dir)
                files.remove('reference.txt')
                audio_files += [os.path.join(sub_dir, f) for f in files]
                reference = np.loadtxt(os.path.join(sub_dir, 'reference.txt'), dtype=str).tolist()
                length_info += [float(r[-1]) for r in reference]
        self.audio_dataset = BaseDataset(audio_files, length_info, sample_rate=4000, length=3, stride=2)
        print('heart dataset size:', len(self.audio_dataset))

        audio_files = []
        length_info = []
        for dataset in lung_datasets:
            for choose_set in lung_datasets[dataset]:
                sub_dir = os.path.join(directory, dataset, choose_set)
                if os.path.exists(sub_dir) is False:
                    continue
                files = os.listdir(sub_dir)
                files.remove('reference.txt')
                audio_files += [os.path.join(sub_dir, f) for f in files]
                reference = np.loadtxt(os.path.join(sub_dir, 'reference.txt'), dtype=str).tolist()
                length_info += [float(r[-1]) for r in reference]
        self.noise_dataset = BaseDataset(audio_files, length_info, sample_rate=4000, length=3, stride=2)
        print('lung dataset size:', len(self.noise_dataset))
    def __len__(self):
        return len(self.audio_dataset)
    def __getitem__(self, index):
        reference = self.audio_dataset[index]
        rms_reference = np.sqrt(np.mean(reference ** 2))
        
        noise = np.random.randn(len(reference)).astype(np.float32)
        rms_noise = np.sqrt(np.mean(noise ** 2))
        noise = noise * rms_reference / rms_noise * 0.5
        # noise = self.noise_dataset[np.random.randint(len(self.noise_dataset))]
        # rms_noise = np.sqrt(np.mean(noise ** 2))
        # noise = noise * rms_reference / rms_noise * 0.5
        # if len(reference) > len(noise):
        #     noise = np.pad(noise, (0, len(reference) - len(noise)), 'constant')
        # else:
        #     noise = noise[:len(reference)]
        audio = reference + noise
        return {'audio': audio, 'reference': reference}
    
class PairedDataset:
    def __init__(self, directory = 'normalized_dataset', dataset='CHSC'):
        datasets = {
        }
        audio_files = []
        length_info = []
        for dataset in datasets:
            for choose_set in datasets[dataset]:
                sub_dir = os.path.join(directory, dataset, choose_set)
                if os.path.exists(sub_dir) is False:
                    continue
                files = os.listdir(sub_dir)
                files.remove('reference.txt')
                audio_files += [os.path.join(sub_dir, f) for f in files]
                reference = np.loadtxt(os.path.join(sub_dir, 'reference.txt'), dtype=str).tolist()
                length_info += [float(r[-1]) for r in reference]
        self.audio_dataset = BaseDataset(audio_files, length_info, sample_rate=4000, length=3, stride=2)
        print('paired dataset size:', len(self.audio_dataset))

    def __len__(self):
        return len(self.audio_dataset)
    def __getitem__(self, index):
        audio = self.audio_dataset[index]
        mic = audio[0]
        steth = audio[1]
        mic_rms = (mic ** 2).mean() ** 0.5
        steth_rms = (steth ** 2).mean() ** 0.5
        snr_scalar = steth_rms / (mic_rms + 1e-8)
        mic *= snr_scalar
        return {'audio': mic, 'reference': steth}
if __name__ == "__main__":
    dataset = StethDataset()
    print(len(dataset))