import gzip
import os
import pickle
import tarfile

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from dezero.transforms import Compose, Flatten, Normalize, ToFloat
from dezero.utils import get_file


class Dataset:
    def __init__(self, train=True, transform=None, target_transform=None):
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        if self.transform is None:
            self.transform = lambda x: x
        if self.target_transform is None:
            self.target_transform = lambda x: x

        self.data = None
        self.label = None
        self.prepare()

    def __getitem__(self, index):
        assert np.isscalar(index)
        if self.label is None:
            return self.transform(self.data[index]), None
        else:
            return self.transform(self.data[index]), self.target_transform(self.label[index])

    def __len__(self):
        return len(self.data)

    def prepare(self):
        pass

# ==============================================================================================
# MNIST dataset: MNIST / Fashion-MNIST / CIFAR-10 / CIFAR-100
# ==============================================================================================
class MNIST(Dataset):
    def __init__(self, train=True, 
                 transform=Compose([Flatten(), ToFloat(), Normalize(0., 255.)]), 
                 target_transform=None):
        super().__init__(train, transform, target_transform)

    def prepare(self):
        url = 'https://ossci-datasets.s3.amazonaws.com/mnist/'
        train_files = {'target': 'train-images-idx3-ubyte.gz',
                       'label': 'train-labels-idx1-ubyte.gz'}
        test_files = {'target': 't10k-images-idx3-ubyte.gz',
                      'label': 't10k-labels-idx1-ubyte.gz'}

        files = train_files if self.train else test_files
        data_path = get_file(url + files['target'], 'mnist')
        label_path = get_file(url + files['label'], 'mnist')

        self.data = self._load_data(data_path)
        self.label = self._load_label(label_path)

    def _load_label(self, filepath):
        with gzip.open(filepath, 'rb') as f:
            labels = np.frombuffer(f.read(), np.uint8, offset=8)
        return labels

    def _load_data(self, filepath):
        with gzip.open(filepath, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=16)
        data = data.reshape(-1, 1, 28, 28)
        return data

    def show(self, row=10, col=10, file_name='mnist'):
        H, W = 28, 28
        img = np.zeros((H * row, W * col))
        for r in range(row):
            for c in range(col):
                img[r * H:(r + 1) * H, c * W:(c + 1) * W] = self.data[
                        np.random.randint(0, len(self.data) - 1)].reshape(H, W)
        plt.imshow(img, cmap='gray', interpolation='nearest')
        plt.axis('off')
        plt.savefig('{}.png'.format(file_name))

    @staticmethod
    def labels():
        return {0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9'}

class FashionMNIST(Dataset):
    def __init__(self, train=True, 
                 transform=Compose([Flatten(), ToFloat(), Normalize(0., 255.)]), 
                 target_transform=None):
        super().__init__(train, transform, target_transform)

    def prepare(self):
        url = 'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/'
        train_files = {'target': 'train-images-idx3-ubyte.gz',
                       'label': 'train-labels-idx1-ubyte.gz'}
        test_files = {'target': 't10k-images-idx3-ubyte.gz',
                      'label': 't10k-labels-idx1-ubyte.gz'}

        files = train_files if self.train else test_files
        data_path = get_file(url + files['target'], 'fashion-mnist')
        label_path = get_file(url + files['label'], 'fashion-mnist')

        self.data = self._load_data(data_path)
        self.label = self._load_label(label_path)

    def _load_label(self, filepath):
        with gzip.open(filepath, 'rb') as f:
            labels = np.frombuffer(f.read(), np.uint8, offset=8)
        return labels

    def _load_data(self, filepath):
        with gzip.open(filepath, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=16)
        data = data.reshape(-1, 1, 28, 28)
        return data

    def show(self, row=10, col=10, file_name='mnist'):
        H, W = 28, 28
        img = np.zeros((H * row, W * col))
        for r in range(row):
            for c in range(col):
                img[r * H:(r + 1) * H, c * W:(c + 1) * W] = self.data[
                        np.random.randint(0, len(self.data) - 1)].reshape(H, W)
        plt.imshow(img, cmap='gray', interpolation='nearest')
        plt.axis('off')
        plt.savefig('{}.png'.format(file_name))

    @staticmethod
    def labels():
        return {0: 'T-shirt/top', 1: 'Trouser', 2: 'Pullover', 3: 'Dress', 4: 'Coat', 5: 'Sandal', 6: 'Shirt', 7: 'Sneaker', 8: 'Bag', 9: 'Ankle boot'}

class CIFAR10(Dataset):
    def __init__(self, train=True,
                 transform=Compose([ToFloat(), Normalize(mean=0.5, std=0.5)]),
                 target_transform=None):
        super().__init__(train, transform, target_transform)

    def prepare(self):
        url = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
        self.data, self.label = load_cache_npz(url, 'cifar-10', self.train)
        if self.data is not None:
            return
        filepath = get_file(url, 'cifar-10')
        if self.train:
            self.data = np.empty((50000, 3 * 32 * 32))
            self.label = np.empty((50000), dtype=int)
            for i in range(5):
                self.data[i * 10000:(i + 1) * 10000] = self._load_data(filepath, i + 1, 'train')
                self.label[i * 10000:(i + 1) * 10000] = self._load_label(filepath, i + 1, 'train')
        else:
            self.data = self._load_data(filepath, 5, 'test')
            self.label = self._load_label(filepath, 5, 'test')
        self.data = self.data.reshape(-1, 3, 32, 32)
        save_cache_npz(self.data, self.label, url, "cifar-10", self.train)

    def _load_data(self, filepath, idx, data_type='train'):
        assert data_type in ['train', 'test']
        with tarfile.open(filepath, 'r:gz') as file:
            for item in file.getmembers():
                if ('data_batch_{}'.format(idx) in item.name and data_type == 'train') or ('test_batch' in item.name and data_type == 'test'):
                    data_dict = pickle.load(file.extractfile(item), encoding='bytes')
                    data = data_dict[b'data']
                    return data

    def _load_label(self, filepath, idx, data_type='train'):
        assert data_type in ['train', 'test']
        with tarfile.open(filepath, 'r:gz') as file:
            for item in file.getmembers():
                if ('data_batch_{}'.format(idx) in item.name and data_type == 'train') or ('test_batch' in item.name and data_type == 'test'):
                    data_dict = pickle.load(file.extractfile(item), encoding='bytes')
                    labels = data_dict[b'labels']
                    return np.array(labels, dtype=int)

    def show(self, row=10, col=10, file_name='cifar10'):
        H, W = 32, 32
        img = np.zeros((H * row, W * col, 3), dtype=np.uint8)
        for r in range(row):
            for c in range(col):
                idx = np.random.randint(0, len(self.data) - 1)
                img[r * H:(r + 1) * H, c * W:(c + 1) * W] = self.data[idx].reshape(3, H, W).transpose(1, 2, 0)/255
        plt.imshow(img, interpolation='nearest')
        plt.axis('off')
        plt.savefig('{}.png'.format(file_name))

    @staticmethod
    def labels():
        return {0: 'airplane', 1: 'automobile', 2: 'bird', 3: 'cat', 4: 'deer',
                5: 'dog', 6: 'frog', 7: 'horse', 8: 'ship', 9: 'truck'}

class CIFAR100(CIFAR10):
    def __init__(self, train=True,
                 transform=Compose([ToFloat(), Normalize(mean=0.5, std=0.5)]),
                 target_transform=None,
                 label_type='fine'):
        assert label_type in ['fine', 'coarse']
        self.label_type = label_type
        super().__init__(train, transform, target_transform)

    def prepare(self):
        url = 'https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz'
        self.data, self.label = load_cache_npz(url, 'cifar-100', self.train)
        if self.data is not None:
            return

        filepath = get_file(url, 'cifar-100')
        if self.train:
            self.data = self._load_data(filepath, 5, 'test')
            self.label = self._load_label(filepath, 5, 'test')
        else:
            self.data = self._load_data(filepath, 5, 'test')
            self.label = self._load_label(filepath, 5, 'test')
        self.data = self.data.reshape(-1, 3, 32, 32)
        save_cache_npz(self.data, self.label, url, "cifar-100", self.train)

    def _load_data(self, filepath, data_type='train'):
        with tarfile.open(filepath, 'r:gz') as file:
            for item in file.getmembers():
                if data_type in item.name:
                    data_dict = pickle.load(file.extractfile(item), encoding='bytes')
                    data = data_dict[b'data']
                    return data

    def _load_label(self, filepath, data_type='train'):
        assert data_type in ['train', 'test']
        with tarfile.open(filepath, 'r:gz') as file:
            for item in file.getmembers():
                if data_type in item.name:
                    data_dict = pickle.load(file.extractfile(item), encoding='bytes')
                    if self.label_type == 'fine':
                        labels = data_dict[b'fine_labels']
                    elif self.label_type == 'coarse':
                        labels = data_dict[b'coarse_labels']
                    return np.array(labels, dtype=int)

    @staticmethod
    def labels(label_type='fine'):
        coarse_labels = dict(enumerate([
            'aquatic mammals', 'fish', 'flowers', 'food containers',
            'fruit and vegetables', 'household electrical devices',
            'household furniture', 'insects', 'large carnivores',
            'large man-made outdoor things', 'large natural outdoor scenes',
            'large omnivores and herbivores', 'medium-sized mammals',
            'non-insect invertebrates', 'people', 'reptiles',
            'small mammals', 'trees', 'vehicles 1', 'vehicles 2'
        ]))
        fine_labels = []
        return fine_labels if label_type is 'fine' else coarse_labels 

class BigData(Dataset):
    def __getitem__(index):
        x = np.load('data/{}.npy'.format(index))
        t = np.load('label/{}.npy'.format(index))
        return x, t

    def __len__(self):
        return 1000000

# ==============================================================================================
# Utils
# ==============================================================================================
def load_cache_npz(filename, dir_name=None, train=True):
    cache_dir = os.path.join(os.path.expanduser('~'), '.dezero', 'datasets')
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)

    if not os.path.exists(os.path.join(cache_dir, dir_name)):
        os.makedirs(os.path.join(cache_dir, dir_name))

    file_name = filename[filename.rfind('/') + 1:]
    prefix = '.train.npz' if train else '.test.npz'
    file_path = os.path.join(cache_dir, dir_name + file_name + prefix)
    if not os.path.exists(file_path):
        return None, None

    loaded = np.load(filepath)
    return loaded['data'], loaded['label']

def save_cache_npz(data, label, url, dir_name=None, train=True):
    cache_dir = os.path.join(os.path.expanduser('~'), '.dezero', 'datasets')
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    if not os.path.exists(os.path.join(cache_dir, dir_name)):
        os.makedirs(os.path.join(cache_dir, dir_name))

    file_name = url[url.rfind('/') + 1:]
    prefix = '.train.npz' if train else '.test.npz'
    file_path = os.path.join(cache_dir, dir_name + file_name + prefix)

    if os.path.exists(file_path):
        return

    print("Saving: " + file_name + prefix)
    try:
        np.savez_compressed(file_path, data=data, label=label)
    except (Exception, KeyboardInterrupt) as e:
        if os.path.exists(file_path):
            os.remove(file_path)
        raise
    print("Done")
    return file_path

