import numpy as np
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from implem.utils import device

class SimpleDataset(torch.utils.data.Dataset):
    def __init__(self, data, offset=1, start=None, end=None):

        super(SimpleDataset, self).__init__()
        assert len(data.shape) >= 2 #[T,*D], where D can be [C,W,H] etc.
        self.T  = len(data)
        
        self.data = data

        self.offset = offset
        self.start = 0 if start is None else start        
        self.end = self.T-np.asarray(self.offset).max() if end is None else end
        assert self.end > self.start

        self.idx = torch.arange(self.start, self.end, requires_grad=False, device='cpu')

    def __getitem__(self, index):
        """ Generate one batch of data """
        x = self.data[self.idx[index]].reshape(*self.data.shape[1:])
        y = self.data[self.idx[index]+self.offset].reshape(len(self.offset), *self.data.shape[1:])
        return x,y
    
    def __len__(self):
        return len(self.idx)

class MultiTrialDataset(torch.utils.data.Dataset):
    def __init__(self, data, offset=1, start=None, end=None):

        super(MultiTrialDataset, self).__init__()
        assert len(data.shape) >= 3 #[N,T,*D], where D can be [C,W,H] etc.
        self.N, self.T  = data.shape[:2]

        self.data = data.reshape(-1, *data.shape[2:]) #[NT,*D]

        self.offset = offset
        self.start = 0 if start is None else start        
        self.end = self.T-np.asarray(self.offset).max() if end is None else end
        assert self.end > self.start

        idx = torch.arange(self.start, self.end, requires_grad=False, device='cpu')
        idx = [idx for j in range(self.N)]
        self.idx = torch.cat([j*self.T + idx[j] for j in range(len(idx))])

    def __getitem__(self, index):
        """ Generate one batch of data """
        x = self.data[self.idx[index]].reshape(*self.data.shape[1:])
        y = self.data[self.idx[index]+self.offset].reshape(*self.data.shape[1:])
        return x,y
    
    def __len__(self):
        return len(self.idx)

class MultiStepMultiTrialDataset(MultiTrialDataset):
    def __init__(self, data, offset=1, start=None, end=None):

        super(MultiStepMultiTrialDataset, self).__init__(data=data, offset=offset, start=start, end=end)
        self.offset = torch.as_tensor(np.asarray(offset, dtype=np.int).reshape(1,-1), device='cpu')

    def __getitem__(self, index):
        """ Generate one batch of data """
        io = (self.idx[index].reshape(-1,1) + self.offset.reshape(1,-1)).flatten()
        x = self.data[self.idx[index]].reshape(*self.data.shape[1:])
        y = self.data[io].reshape(np.prod(self.offset.shape), *self.data.shape[1:])
        return x,y        

class DataModule(pl.LightningDataModule):

    def __init__(self, data, train_valid_split: int = 0.9,
                 batch_size: int = 2, offset: int = 1, Dataset=SimpleDataset,
                 **kwargs):
        super().__init__()

        self.data = data
        self.Dataset = Dataset
        self.batch_size = batch_size
        self.offset = offset if isinstance(offset, np.ndarray) else np.arange(offset)
        self.num_workers = 0
        assert 0. < train_valid_split and train_valid_split <= 1.
        self.train_valid_split = train_valid_split

    def setup(self, stage=None):

        if stage == 'fit' or stage is None:
            split_index = int(len(self.data) * self.train_valid_split)

            self.train_data = self.Dataset(data = self.data[:split_index], offset = self.offset)
            self.valid_data = self.Dataset(data = self.data[split_index:], offset = self.offset)

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size, num_workers=self.num_workers,
                          shuffle=True, generator=torch.Generator(device=device))

    def val_dataloader(self):
        return DataLoader(self.valid_data, batch_size=self.batch_size, num_workers=self.num_workers,
                          shuffle=False, generator=torch.Generator(device=device))