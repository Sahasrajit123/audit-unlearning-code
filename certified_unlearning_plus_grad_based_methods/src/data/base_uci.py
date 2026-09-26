from torch.utils.data import Dataset
import numpy as np

class UCIDatasetBase(Dataset):
    def __init__(self, data, target, transform=None):
        self.X = data.astype(np.float32)
        self.y = target
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.y[idx]
        if self.transform:
            x = self.transform(x)
        return x, y