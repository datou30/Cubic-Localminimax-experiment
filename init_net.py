import torch
import torchvision
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import time
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torchvision.utils import make_grid
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import random_split
from math import sqrt
from net import Net
from device import DeviceDataLoader

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

torch.manual_seed(0)
net = Net()
torch.save(net.state_dict(), "path/to/ini.pth")
net.to(device)

transform = torchvision.transforms.Compose([
        # transforms.Grayscale(3),
        torchvision.transforms.ToTensor()
])
dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)

train_set, test_set = random_split(dataset=dataset, lengths=[50000, 10000], generator=torch.Generator().manual_seed(0))
test_loader = torch.utils.data.DataLoader(test_set, batch_size=2000, shuffle=False, num_workers=0)
test_loader = DeviceDataLoader(test_loader, device)

correct = 0
total = 0
with torch.no_grad():
    for data in test_loader:
        images, labels = data
        images = images.float().to(device)
        labels.float().to(device)
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
print("initial acc = ", correct/total)