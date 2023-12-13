import math
import torchaudio
import torch
import librosa
import os
import warnings
import numpy as np
from mutagen.wave import WAVE 
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
        noisy_target_dB_FS = np.random.randint(target_dB_FS - 10, target_dB_FS + 10)
        # 使用 noisy 的 rms 放缩音频
        noisy_y, _, noisy_scalar = tailor_dB_FS(noisy_y, noisy_target_dB_FS)
        clean_y *= noisy_scalar
        noise_y *= noisy_scalar
        # 合成带噪语音的时候可能会 clipping，虽然极少
        # 对 noisy, clean_y, noise_y 稍微进行调整
        if (noise_y.max() > 0.999).any():
            noisy_y_scalar = np.abs(noisy_y).max() / (0.99 - eps)  # 相当于除以 1
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
    def __len__(self):
        return sum(self.num_examples)
    def __getitem__(self, index):
        for file, samples, examples in zip(self.files, self.length_info, self.num_examples):
            if index >= examples:
                index -= examples
                continue
            # data, _  = torchaudio.load(file, frame_offset =self.stride * index*self.sample_rate, 
            # num_frames=self.length * self.sample_rate, backend='ffmpeg')
            data, _  = librosa.load(file, sr=self.sample_rate, offset =self.stride * index, duration=self.length, mono=False)
            if data.ndim == 1:
                data = data.reshape(1, -1)

            if data.shape[-1] < (self.sample_rate * self.length):
                pad_before = np.random.randint((self.sample_rate * self.length) - data.shape[-1])
                pad_after = (self.sample_rate *self.length) - data.shape[-1] - pad_before
                # data = torch.nn.functional.pad(data, (pad_before, pad_after, 0, 0 ), 'constant')
                data = np.pad(data, ((0, 0), (pad_before, pad_after)), 'constant')
            return data
class PublicDataset:
    def __init__(self, directory = 'normalized_dataset', dataset='CHSC'):
        heart_datasets = {
            'CHSC': ['set_a', 'set_b'],
            'PhysioNet': ['training-a', 'training-b', 'training-c', 'training-d', 'training-e', 'training-f', 'validation'],
            # 'Thinklabs':['wav'],
            # 'CirCor': ['training_data'],
            # 'ephongram': ['WAV']
            # 'radarheart': ['WAV']
        }
        lung_datasets = {
            'Respiratory': ['audio_and_txt_files'],
            'lungsound': ['Audio Files']
        }
        audio_files = []
        length_info = []
        for dataset in heart_datasets:
            for choose_set in heart_datasets[dataset]:
                sub_dir = os.path.join('dataset', directory, dataset, choose_set)
                if os.path.exists(sub_dir) is False:
                    continue
                files = os.listdir(sub_dir)
                files.sort()
                files.remove('reference.txt')
                audio_files += [os.path.join(sub_dir, f) for f in files]
                reference = np.loadtxt(os.path.join(sub_dir, 'reference.txt'), dtype=str).tolist()
                length_info += [float(r[-1]) for r in reference]
        self.audio_dataset = BaseDataset(audio_files, length_info, sample_rate=4000, length=3, stride=2)
        print('heart dataset size:', len(self.audio_dataset))

        self.noise_files = []
        for section in os.listdir(os.path.join('dataset', 'audioset')):
            sub_dir = os.path.join('dataset', 'audioset', section)
            files = os.listdir(sub_dir)
            files.sort()
            self.noise_files += [os.path.join(sub_dir, f) for f in files]
        print('noise dataset size:', len(self.noise_files))
    def __len__(self):
        return len(self.audio_dataset)
    def __getitem__(self, index):
        reference = self.audio_dataset[index][0]

        noise_file = self.noise_files[np.random.randint(len(self.noise_files))]
        noise, sr = librosa.load(noise_file, duration=9, sr=4000)
        if reference.shape[-1] > noise.shape[-1]:
            noise = np.pad(noise, (0, reference.shape[-1] - noise.shape[-1]), 'constant')
        else:
            random_start = np.random.randint(noise.shape[-1] - reference.shape[-1])
            noise = noise[random_start:random_start + reference.shape[-1]]
        # noise = np.random.randn(reference.shape[-1]).astype(np.float32)

        noisy_y, clean_y, noise_y, noisy_scalar = snr_mix(noise, reference, np.random.randint(-5, 5), -10)
        mix = np.stack([noise_y, clean_y])
        return {'audio': noisy_y, 'reference': clean_y, 'mix': mix}
    
class PairedDataset:
    def __init__(self, directory = 'dataset/thinklabs_processed', people = [], phone = [], textile = [], train=True):
        self.audio_dataset = []
        people = os.listdir(directory) if people == [] else people
        for p in people:
            if phone == [] and textile == []:
                sessions = os.listdir(os.path.join(directory, p))
            else:
                sessions = []
                for s in os.listdir(os.path.join(directory, p)):
                    ph, te = s.split('_')
                    if ph in phone or te in textile:
                        sessions.append(s)
            for s in sessions:
                sub_dir = os.path.join(directory, p, s)
                files = os.listdir(sub_dir)
                if 'reference.txt' in files: # if not, the subsec is excluded (or not synchronized)
                    files.remove('reference.txt')
                    files.sort()
                    audio_files = [os.path.join(sub_dir, f) for f in files if f.startswith('Stereo')]
                    reference = np.loadtxt(os.path.join(sub_dir, 'reference.txt'), dtype=str).tolist()
                    length_info = [float(r) for r in reference]
                    audio_dataset = BaseDataset(audio_files, length_info, sample_rate=4000, length=3, stride=2)
                    train_datset, test_dataset = torch.utils.data.random_split(audio_dataset, [int(len(audio_dataset) * 0.8), len(audio_dataset) - int(len(audio_dataset) * 0.8)],
                                                                generator=torch.Generator().manual_seed(42))
                    if train:
                        self.audio_dataset.append(train_datset)
                    else:
                        self.audio_dataset.append(test_dataset)
        self.audio_dataset = torch.utils.data.ConcatDataset(self.audio_dataset)
        # if train:
        #     print('train dataset size:', people, phone, textile, len(self.audio_dataset))
        # else:   
        #     print('test dataset size:', people, phone, textile, len(self.audio_dataset))

    def __len__(self):
        return len(self.audio_dataset)
    def __getitem__(self, index):
        audio = self.audio_dataset[index]
        mic = audio[0]
        steth = audio[1]
        # mic_rms = (mic ** 2).mean() ** 0.5
        # steth_rms = (steth ** 2).mean() ** 0.5
        # snr_scalar = steth_rms / (mic_rms + 1e-8)
        # mic *= snr_scalar
        return {'audio': mic, 'reference': steth}
if __name__ == "__main__":
    dataset = PublicDataset()
    print(len(dataset))
    dataset = PairedDataset(train=True)
    print(len(dataset))
    dataset = PairedDataset(train=False)
    print(len(dataset))