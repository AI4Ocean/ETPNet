import h5py
from Moudle import *
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset


class _Loader:
    def __init__(self, batch_size, pic_days, flag, file_name, test_file_name='Northern.h5'):
        self.train_path = Path('dataset/train/')
        self.verification_path = Path('dataset/verification')
        self.test_path = Path('dataset/test')
        self.file_name = file_name
        self.test_file_name = test_file_name
        self.batch_size = batch_size
        self.pic_days = pic_days
        self.train_mean = None
        self.train_std = None
        self.flag = flag

    def tensor_loader(self, data, shuffle, batch_size):
        dataset = TensorDataset(torch.tensor(data[:, :-self.pic_days, :].astype('float64'), dtype=torch.float),
                              torch.tensor(data[:, -self.pic_days:, :2].astype('float64'), dtype=torch.float))
        loader = DataLoader(dataset, shuffle=shuffle, batch_size=batch_size)

        return loader

    def mean_std(self, data):
        train_mean = []
        train_std = []
        m, n, z = data.shape
        for i in range(2, 4):
            temp_mean = np.mean(data[:, :, i])
            train_mean.append(temp_mean)

            temp_std = np.std(data[:, :, i])
            train_std.append(temp_std)

        self.train_mean = train_mean
        self.train_std = train_std

    def standard(self, data):
        m, n, z = data.shape
        for i in range(2, 4):
            temp_mean = self.train_mean[i - 2]
            temp_std = self.train_std[i - 2]
            temp_stand = np.divide(np.subtract(data[:, :, i], temp_mean), temp_std)
            data[:, :, i] = temp_stand


        return data

    def data_loader(self):
        train_content = h5py.File(self.train_path / self.file_name, 'r')
        train_data = train_content['train'][()]
        train = np.delete(train_data, 0, axis=2)
        self.mean_std(train)
        train = self.standard(train)
        train_loader = self.tensor_loader(train, False, self.batch_size)

        if self.flag == 'train':
            return train_loader

        if self.flag == 'test':
            data_content = h5py.File(self.test_path / self.test_file_name, 'r')
            test_data = data_content['test'][()]
            test = np.delete(test_data, 0, axis=2)
            test = self.standard(test)
            test_loader = self.tensor_loader(test, False, self.batch_size)

            return test_loader
