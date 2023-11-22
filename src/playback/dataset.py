from playback_prepare import *
import math
import torch
import librosa
import warnings
warnings.filterwarnings("ignore")

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
            if data.shape[-1] < (self.sample_rate * self.length):
                pad_before = np.random.randint((self.sample_rate * self.length) - data.shape[-1])
                pad_after = (self.sample_rate *self.length) - data.shape[-1] - pad_before
                # data = torch.nn.functional.pad(data, (pad_before, pad_after, 0, 0 ), 'constant')
                data = np.pad(data, ((0, 0), (pad_before, pad_after)), 'constant')
            return data
class PlaybackDataset:
    def __init__(self, directory = 'smartphone', dataset='CHSC'):
        datasets = {
            # 'CHSC': ['set_a', 'set_b'],
            'CHSC': ['set_b'],
            'Cardiology_Challenge_2016': ['training-a', 'training-b', 'training-c', 'training-d', 'training-e', 'training-f', 'validation'],
            'Thinklabs':['wav'],
            'CirCor': ['training_data']
        }
        audio_files = []
        imu_files = []
        reference_files = []
        length_info = []
        for choose_set in datasets[dataset]:
            sub_dir = os.path.join(directory, dataset, choose_set + '_processed')
            if os.path.exists(sub_dir) is False:
                continue
            reference = np.loadtxt(os.path.join(sub_dir, 'reference.txt'), dtype=str).tolist()
            imu_files += [os.path.join(sub_dir, r[0]) for r in reference]
            audio_files += [os.path.join(sub_dir, r[1]) for r in reference]
            length_info += [float(r[2]) for r in reference]
        self.audio_dataset = BaseDataset(audio_files, length_info, sample_rate=4000, length=3, stride=2)
        self.imu_dataset = BaseDataset(imu_files, length_info, sample_rate=400, length=3, stride=2)
    def __len__(self):
        return len(self.audio_dataset)
    def __getitem__(self, index):
        audio = self.audio_dataset[index]
        mic_recording = audio[0]
        max_scale = np.max(np.abs(mic_recording))
        mic_recording = mic_recording / max_scale
        reference = audio[1]
        return {'audio': mic_recording, 'reference': reference}
if __name__ == "__main__":
    dataset = PlaybackDataset()
    print(len(dataset))