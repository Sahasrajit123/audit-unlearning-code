from torchvision import transforms
from sklearn.datasets import load_iris, load_breast_cancer, load_wine
from sklearn.model_selection import train_test_split
from src.data.base_datamodule import BaseDataModule
from src.data.base_uci import UCIDatasetBase
from torch.utils.data import Dataset
from sklearn.datasets import fetch_rcv1
import numpy as np
from sklearn.preprocessing import StandardScaler


class IRISDataModule(BaseDataModule):
    def __init__(self, config, generator):
        transform = None
        super().__init__(config, generator, self.IRISDataset, transform, transform)

    class IRISDataset(UCIDatasetBase):
        def __init__(self, root, train=True, transform=None, download=True):
            data = load_iris()
            X_train, X_test, y_train, y_test = train_test_split(
                data.data, data.target, test_size=0.2, random_state=42
            )
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
            super().__init__(
                X_train if train else X_test, y_train if train else y_test, transform
            )


class BreastCancerDataModule(BaseDataModule):
    def __init__(self, config, generator):
        transform = transforms.Compose([transforms.ToTensor()])
        super().__init__(config, generator, self.BCDataset, transform, transform)

    class BCDataset(UCIDatasetBase):
        def __init__(self, root, train=True, transform=None, download=True):
            data = load_breast_cancer()
            X_train, X_test, y_train, y_test = train_test_split(
                data.data, data.target, test_size=0.2, random_state=42
            )
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
            super().__init__(
                X_train if train else X_test, y_train if train else y_test, transform
            )


class WineDataModule(BaseDataModule):
    def __init__(self, config, generator):
        transform = None
        super().__init__(config, generator, self.WineDataset, transform, transform)

    class WineDataset(UCIDatasetBase):
        def __init__(self, root, train=True, transform=None, download=True):
            from ucimlrepo import fetch_ucirepo
            from sklearn.model_selection import train_test_split
            from sklearn.preprocessing import StandardScaler

            wine_quality = fetch_ucirepo(id=186)

            X = wine_quality.data.features.to_numpy()
            y = wine_quality.data.targets.to_numpy()
            # set 0-5 to 0 and 6 to 1 and 7-9 to 2
            y=np.where(y<=5,0,y)
            y=np.where(y==6,1,y)
            y=np.where(y>=7,2,y)

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )

            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

            if train:
                super().__init__(X_train, y_train, transform)
            else:
                super().__init__(X_test, y_test, transform)


class RCV1DataModule(BaseDataModule):
    def __init__(self, config, generator):
        # transform = transforms.Compose([transforms.ToTensor()])
        transform = None
        super().__init__(config, generator, self.RCV1Dataset, transform, transform)

    class RCV1Dataset(Dataset):
        def __init__(self, root, train=True, transform=None, download=True):
            self.rcv1 = fetch_rcv1(
                data_home=root,
                subset="train" if train else "test",
                download_if_missing=download,
            )
            self.X = self.rcv1.data.toarray().astype(np.float32)
            self.y = self.rcv1.target.toarray().astype(np.float32)
            self.transform = transform

        def __len__(self):
            return len(self.X)

        def __getitem__(self, idx):
            x = self.X[idx]
            y = self.y[idx]
            if self.transform:
                x = self.transform(x)
            return x, y
