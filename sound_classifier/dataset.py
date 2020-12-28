import os
import numpy as np
from torch.utils.data import Dataset

class SpecDataset(Dataset):
    def __init__(self, root='../Data/umsun_dataset_44k/spec', mean=None, std=None, time_dim_size=431, mode=None):
        self.classes = {'alarm clock': 0, 'clock': 1, 'bee': 2, 'bird': 3, 'owl': 4,
                        'phone': 5, 'church': 6, 'cow': 7, 'duck': 8, 'dog': 9,
                        'frog': 10, 'horse': 11, 'keyboard': 12, 'pencil': 13, 'guitar': 14,
                        'piano': 15, 'train': 16, 'violin': 17, 'clarinet': 18, 'pig': 19}
        self.classes_list = ['alarm clock', 'clock', 'bee', 'bird', 'owl',
                        'phone', 'church', 'cow', 'duck', 'dog',
                        'frog', 'horse', 'keyboard', 'pencil', 'guitar',
                        'piano', 'train', 'violin', 'clarinet', 'pig']
        self.paths = self.path_finder(root, mode)
        if mean is None or std is None or time_dim_size is None:
            self.mean, self.std, self.time_dim_size, self.paths = self.initialize(self.paths)
            print(self.mean, self.std, self.time_dim_size)
        else:
            self.mean = mean
            self.std = std
            self.time_dim_size = time_dim_size


    def __getitem__(self, i):
        # Get i-th path.
        path, label = self.paths[i]
        # Load the mel-spectrogram.
        spec = np.load(path)
        if self.time_dim_size is not None:
            # Slice the temporal dimension with a fixed length so that they have
            # the same temporal dimensionality in mini-batches.
            spec = spec[:, :self.time_dim_size]
        # Perform standard normalization using pre-computed mean and std.
        spec = (spec - self.mean) / self.std

        return np.expand_dims(spec, axis=0), label

    def path_finder(self, root, mode):
        paths = []
        assert mode is not None
        cls_list = os.listdir(root)
        id = 0
        for cls in cls_list:
            cls_path = os.path.join(root, cls)
            specs = os.listdir(cls_path)
            for file in specs:
                if mode == 'val' and id%8 == 0:
                    paths.append((os.path.join(cls_path, file), self.classes[cls]))
                if mode == 'train' and id%8 != 0:
                    paths.append((os.path.join(cls_path, file), self.classes[cls]))
                id += 1
        return paths

    def initialize(self, paths, limit=800):
        specs = []
        for s, l in paths:
            sp = np.load(s)
            specs.append(sp)
        time_dims = [s.shape[1] for s in specs]
        min_time_dim_size = min(time_dims)
        specs = [s[:, :min_time_dim_size] for s in specs]
        specs = np.stack(specs)
        mean = specs.mean()
        std = specs.std()
        return mean, std, min_time_dim_size, paths


    def __len__(self):
        return len(self.paths)


