import h5py
import torch
import torch.utils.data as Data


class MyDataSet(Data.Dataset):
    def __init__(self, h5py_path):
        data_file = h5py.File(h5py_path, 'r')
        self.data = torch.from_numpy(data_file['data'].value)
        self.nSamples = self.data.size(0)
        self.label = torch.ones((self.nSamples,1))

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        assert index < len(self), 'index range error'
        data = self.data[index]
        label = self.label[index]
        return (data, label)
