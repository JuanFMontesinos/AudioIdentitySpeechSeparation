from random import randint
import numpy as np
from scipy.io.wavfile import read
from flerken.audio import np_int2float
from torch.utils.data import DataLoader, Dataset
from torchtree import Directory_Tree

__all__ = ['dataloader', 'AudioLoader']


def dataloader(path, batch_size, debug=False, duplicate=False):
    from multiprocessing import cpu_count
    n_workers = 0 if debug else cpu_count()
    return DataLoader(AudioLoader(path, duplicate), batch_size, num_workers=n_workers, shuffle=True, pin_memory=True, )


def normalize_max(waveform):
    if np.abs(waveform).max() != 0:
        waveform_out = waveform / np.abs(waveform).max()
    else:
        waveform_out = waveform

    return waveform_out


class AudioLoader(Dataset):
    def __init__(self, path, duplicate):
        self.path = path
        tree = Directory_Tree(self.path)
        self.files = list(tree.paths(root=self.path))
        if duplicate:
            self.files = self.files + self.files + self.files

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        sampled = False
        while not sampled:
            try:
                inputs = self.getitem(randint(0, len(self) - 1))
                sampled = True
            except ValueError as ex:
                print(ex)
        return inputs

    def getitem(self, idx):
        file = self.files[idx]
        emb_path = file.replace('.wav', '.npy').replace('audio', 'identity_emb')
        embedding = np.load(emb_path)
        sr, waveform = read(file, mmap=True)
        i = randint(0, len(waveform) - 4 * 16384 - 10)
        waveform = waveform[i:i + 4 * 16384 - 10].copy()
        waveform = np_int2float(waveform, raise_error=True)
        waveform = normalize_max(waveform)

        return {'audio': waveform, 'embedding': embedding}
