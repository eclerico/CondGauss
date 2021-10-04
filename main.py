import torch
import torch.nn as nn
import torch.nn.functional as F

from stochnet.network import GhostNet, StochNet
from stochnet.datasets import MNISTData, CIFAR10Data
import stochnet.tools as tools
from stochnet.tools import Print

import os, sys
from time import strftime


outpath = f'./StochNet_{strftime("%Y%m%d-%H%M%S")}'

try: assert os.path.exists(outpath)
except AssertionError: os.makedirs(outpath)
assert os.path.isdir(outpath)

tools.__out_dir__ = outpath
tools.__out_file__ = 'output' #Name of the log file, if None the output will not be printed to a file
tools.__term__ = True #If False does not print output on the terminal

Print(strftime("%Y%m%d-%H%M%S"))

torch.manual_seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Print(f'Running on {device}')
Print()

Print('Running examples')

"""MNIST"""

DATA = MNISTData()
classes = DATA.classes
n_features = DATA.n_features
channels = DATA.channels
TL = DATA.TrainLoader

Print()

gn1 = GhostNet()
gn1.add_block(nn.Conv2d(in_channels=channels, out_channels=10, kernel_size=5, bias=True), name='conv0', post=lambda x: F.max_pool2d(x, 2), post_name='pool')
gn1.add_block(nn.Conv2d(in_channels=10, out_channels=20, kernel_size=5, bias=True), name='conv1', act=F.relu, post=lambda x: torch.flatten(F.max_pool2d(x, 2), 1), post_name='pool|flatten')
gn1.add_block(nn.Linear(in_features=320, out_features=50, bias=True), name='lin0', act=F.relu)
gn1.add_block(nn.Linear(in_features=50, out_features=classes, bias=True), name='lin1', act=F.relu)

sn1 = gn1.StochSon(name='SN1')

EPOCH = [1, 1]
LR = [0.05, 0.001]

sn1.TrainingSchedule(TL, EPOCH, LR, procedure='cond', method='invKL', track=True)
sn1b = StochNet.Load('Best_SN1', name='SN1 Best')

sn1.GuessBound(TL, repeat=10)
sn1.PrintBound(TL, N_nets=10)



"""CIFAR10"""

DATA = CIFAR10Data()
classes = DATA.classes
n_features = DATA.n_features
channels = DATA.channels
TL1, TL2 = DATA.SplittedTrainLoader(alpha=.5)
Print()

gn2 = GhostNet()

gn2.add_block(nn.Conv2d(in_channels=channels, out_channels=6, kernel_size=5, padding=1, bias=True), name='conv0', post=lambda x: F.max_pool2d(x, kernel_size=2, stride=2), post_name='pool')
gn2.add_block(nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, bias=True), name='conv1', act=F.relu, post=lambda x: torch.flatten(F.max_pool2d(x, kernel_size=2, stride=2), 1), post_name='pool|flatten')
gn2.add_block(nn.Linear(in_features=400, out_features=120, bias=True), name='lin0', act=F.relu)
gn2.add_block(nn.Linear(in_features=120, out_features=84, bias=True), name='lin1', act=F.relu)
gn2.add_block(nn.Linear(in_features=84, out_features=classes, bias=True), name='lin2', act=F.relu)

sn2 = gn2.StochSon(name='SN2')

EPOCH = [1, 1]
LR = [0.05, 0.001]

#Training the prior
sn2.set_dropout(.1)
sn2.TrainingSchedule(TL1, EPOCH, LR, procedure='cond', method='ERM')
sn2.ResetPrior()

EPOCH = [1, 1]
LR = [0.001, 0.0001]
sn2.set_dropout(0)
sn2.TrainingSchedule(TL2, EPOCH, LR, procedure='cond', method='invKL', track=True)
sn2b = StochNet.Load('Best_SN2', name='SN2 Best')
sn2b.PrintBound(TL2, N_nets=10)

