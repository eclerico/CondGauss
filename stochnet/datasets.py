import torch
import torchvision.transforms as transforms

from torch.utils.data import DataLoader
from torchvision.datasets import MNIST, CIFAR10

torch.manual_seed(0)

class MyData():

  def make_loaders(self):
    self.TrainLoader = DataLoader(self.TrainData, batch_size=self.train_batch_size, shuffle=True)
    self.TrainLoader.info = dict(self.info)
    self.TrainLoader.info.update({'Loader':'Train', 'Total size':self.train_size, 'Batch size':self.train_batch_size})
    self.TestLoader = DataLoader(self.TestData, batch_size=self.test_batch_size, shuffle=False)
    self.TestLoader.info = dict(self.info)
    self.TestLoader.info.update({'Loader':'Test', 'Total size':self.test_size, 'Batch size':self.test_batch_size})
    self.SingleLoader = DataLoader(self.TrainData, batch_size=1, shuffle=False)
    self.SingleLoader.info = dict(self.info)
    self.SingleLoader.info.update({'Loader':'Train', 'Total size':self.train_size, 'Batch size':1})
    self._SingleIterLoader = iter(self.SingleLoader)

  def Next(self):
    try:
      return next(self._SingleIterLoader)
    except StopIteration:
      self._SingleIterLoader = iter(self.SingleLoader)
      return next(self._SingleIterLoader)

  def SplittedTrainLoader(self, alpha):
    assert alpha < 1 and alpha >= 0
    size = int(alpha*self.train_size)
    data1, data2 = torch.utils.data.random_split(self.TrainData, (size, self.train_size-size), generator=torch.Generator().manual_seed(42))
    tot_size1 = len(data1)
    batch_size1 = min(self.train_batch_size, tot_size1)
    loader1 = DataLoader(data1, batch_size=batch_size1, shuffle=True)
    loader1.info = dict(self.info)
    loader1.info.update({'Loader':'Splitted Train (1)', 'Total size': tot_size1, 'Batch_size': batch_size1})
    tot_size2 = len(data2)
    batch_size2 = min(self.train_batch_size, tot_size2)
    loader2 = DataLoader(data2, batch_size=batch_size2, shuffle=True)
    loader2.info = dict(self.info)
    loader2.info.update({'Loader':'Splitted Train (2)', 'Total size': tot_size2, 'Batch_size': batch_size2})
    return loader1, loader2

  def _info(self, name):
    self.info = {
      'Dataset' : name,
      'Classes' : self.classes,
      'Features' : self.n_features,
      'Channels' : self.channels
      }

#Load MNIST already on {device}
class FastMNIST(MNIST):

    def __init__(self, *args, flatten = True, binary = False, device = torch.device("cuda" if torch.cuda.is_available() else "cpu"), **kwargs):
        super().__init__(*args, **kwargs)
        #Scale data to [0,1]
        self.data = self.data.float().div(255)
        self.data = self.data.sub_(0.1307).div_(0.3081)
        if flatten: self.data = self.data.view(self.data.shape[0], -1)
        else: self.data = self.data.unsqueeze(-3)
        if binary:
          self.targets = torch.floor_divide(self.targets, 5)
        self.data, self.targets = self.data.to(device), self.targets.to(device)

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        return img, target


class MNISTData(MyData):

  def __init__(self, batch_size = None, train_batch_size = None, test_batch_size = None, binary = False, flatten = False, device = torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    self.classes = 2 if binary else 10 #number of categories for classification
    self.n_features = 28*28 #dimension of the input space
    self.channels = 1
    self.binary = binary
    if batch_size is None:
      if train_batch_size is None: train_batch_size = 128
      if test_batch_size is None: test_batch_size = 128
    else:
      assert train_batch_size is None and test_batch_size is None, "Too many batch sizes given"
      train_batch_size = batch_size
      test_batch_size = batch_size
    self.train_batch_size = train_batch_size
    self.test_batch_size = test_batch_size
    self.device = device
    self.TrainData = FastMNIST('./MNIST/', train=True, binary=binary, flatten=flatten, device=device, download=True)
    self.TestData = FastMNIST('./MNIST/', train=False, binary=binary, flatten=flatten, device=device, download=True)
    self.train_size = len(self.TrainData)
    self.test_size = len(self.TestData)
    self.tot_size = self.train_size + self.test_size
    self._info('MNIST')
    self.make_loaders()


class CIFAR10Data(MyData):

  def __init__(self, batch_size = None, train_batch_size = None, test_batch_size = None, device = "cpu"):
    self.classes = 10
    self.n_features = 32*32*3
    self.channels = 3
    if batch_size is None:
      if train_batch_size is None: train_batch_size = 128
      if test_batch_size is None: test_batch_size = 128
    else:
      assert train_batch_size is None and test_batch_size is None, "Too many batch sizes given"
      train_batch_size = batch_size
      test_batch_size = batch_size
    self.train_batch_size = train_batch_size
    self.test_batch_size = test_batch_size
    self.device = device
    transform = transforms.Compose(
      [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    self.TrainData = CIFAR10(root='./CIFAR10', train=True,
                                        download=True, transform=transform)
    self.TestData = CIFAR10(root='./CIFAR10', train=False,
                                       download=True, transform=transform)
    self.train_size = len(self.TrainData)
    self.test_size = len(self.TestData)
    self.tot_size = self.train_size + self.test_size
    self._info('CIFAR10')
    self.make_loaders()
