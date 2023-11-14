from playback_prepare import *
import librosa
import math
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
            if self.length is None:
                data, _ = librosa.load(file)
                return data, file
            else:
                data, _ = librosa.load(file, sr=self.sample_rate, offset=self.stride * index,
                                        duration=self.length)
                if data.shape[-1] < (self.sample_rate * self.length):
                    pad_before = np.random.randint((self.sample_rate * self.length) - data.shape[-1])
                    pad_after = (self.sample_rate *self.length) - data.shape[-1] - pad_before
                    data = np.pad(data, (pad_before, pad_after, 0, 0), 'constant')
                return data, file
class PlaybackDataset:
    def __init__(self, directory = 'smartphone'):
        datasets = {
            'CHSC': ['set_a', 'set_b'],
            'Cardiology_Challenge_2016': ['training-a', 'training-b', 'training-c', 'training-d', 'training-e', 'training-f', 'validation'],
            'Thinklabs':['wav']
        }
        audio_files = []
        imu_files = []
        reference_files = []
        length_info = []
        for dataset in datasets:
            for choose_set in datasets[dataset]:
                sub_dir = os.path.join(directory, dataset, choose_set)
                if os.path.exists(sub_dir) is False:
                    continue
                file = os.listdir(sub_dir)
                audio_files += [os.path.join(sub_dir, f) for f in file if f.split('_')[0] == 'MIC']
                imu_files += [os.path.join(sub_dir, f) for f in file if f.split('_')[0] == 'IMU']
                reference = np.loadtxt(os.path.join(sub_dir, 'reference.txt'), dtype=str).tolist()
                reference_files += [r[0] for r in reference]
                length_info += [float(r[-1]) for r in reference] 
        self.audio_dataset = BaseDataset(audio_files, length_info, sample_rate=44100, length=3, stride=3)
        self.reference_dataset = BaseDataset(reference_files, length_info, sample_rate=44100, length=3, stride=3)
        self.imu_dataset = BaseDataset(imu_files, length_info, sample_rate=400, length=3, stride=3)
    def __len__(self):
        return len(self.audio_dataset)
    def __getitem__(self, index):
        audio, _ = self.audio_dataset[index]
        reference, _ = self.reference_dataset[index]
        return {'audio': audio, 'reference': reference}
if __name__ == "__main__":
    dataset = PlaybackDataset()
    print(len(dataset))