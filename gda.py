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

def gda(gamma=1.0, batch_size=128, num_epochs=10, num_ascent_epochs=20, lr_1=0.1, lr_2=0.01):
    ## lr_1 is used for ascent, lr_2 is used for the cubic step, lr_3 and lr_4 are used for the gda step to solve the cubic problem
    ## here we set B1=B2=B11=B12=B21=B22=batch_size, and in each epoch we only solve one minibatch

    # Step 0. Initialization

    transform = torchvision.transforms.Compose([
        # transforms.Grayscale(3),
        torchvision.transforms.ToTensor()
    ])
    dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)

    train_set, test_set = random_split(dataset=dataset, lengths=[50000, 10000], generator=torch.Generator().manual_seed(0))
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=2000, shuffle=False, num_workers=0)
    test_loader = DeviceDataLoader(test_loader, device)

    net = Net()
    net.load_state_dict(torch.load("path/to/ini.pth"))
    torch.save(net.state_dict(), "tmp.pth")
    net.to(device)

    train_loss = []
    acc = []
    robust_acc = []
    sample_size = []
    time_result = []

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
    print("initialize acc =", correct / total)
    print("==========> Start train by classical GDA")

    for i in range(num_epochs):
        sample = 0
        print("The %d-th epoch starts." % i)
        epoch_start_time = time.time()
        est_loss = 0.0

        ## step 1, num_ascent_epochs steps of gradient ascend on kxi

        train_loader = DataLoader(train_set, batch_size, shuffle=True, num_workers=4, pin_memory=True)
        train_loader = DeviceDataLoader(train_loader, device)

        net = Net()
        net.load_state_dict(torch.load("tmp.pth"))
        net.to(device)

        for images, labels in train_loader:
            kxi = torch.clone(images)  ## first set kxi=x
            x = torch.clone(images)
            targets = labels
            break

        start_time = time.time()

        kxi.requires_grad = True
        x.requires_grad = False
        targets.requires_grad = False
        for param in net.parameters():
            param.requires_grad = False

        for j in range(num_ascent_epochs):
            output = net(kxi)
            loss1 = F.cross_entropy(output, targets)
            loss2 = torch.sum((kxi - x) ** 2) / batch_size
            loss = loss1 - gamma * loss2
            loss.backward()
            kxi.data = kxi.data + lr_1 * kxi.grad.data
            kxi.grad.data.zero_()
            sample += batch_size

        ## step 2, use gradient to calculate the new parameters of the neural network

        for j in range(1):
            net.load_state_dict(torch.load("tmp.pth"))

            kxi.requires_grad = False
            for param in net.parameters():
                param.requires_grad = True
            output = net(kxi)
            loss1 = F.cross_entropy(output, labels)
            loss2 = torch.sum((x - kxi) ** 2) / batch_size
            loss = loss1 - gamma * loss2

            f_xgrad = torch.autograd.grad(loss, net.parameters())  ## use to store grad_x f(x,y), here x means the net parameters

            num_layer = 0
            with torch.no_grad():
                for name, param in net.named_parameters():
                    param.add_(-lr_2 * f_xgrad[num_layer])
                    num_layer += 1

            torch.save(net.state_dict(), "tmp.pth")

            sample += batch_size
        
        end_time = time.time()
        time_result.append(end_time-start_time)

        ## step 3 variation step

        sample_size.append(sample)

        ## compute the phi(x)
        # use 100-step gradient ascent as estimate loss
        est_net = Net()
        est_net.load_state_dict(torch.load("tmp.pth"))
        est_net.to(device)

        test_loader = torch.utils.data.DataLoader(train_set, batch_size=2000, shuffle=False, num_workers=0)
        test_loader = DeviceDataLoader(test_loader, device)

        est_loss = 0.0
        loss_tmp = []
        count = 0
        for param in est_net.parameters():
            param.requires_grad = False
        for data in test_loader:
            count += 1
            images, labels = data
            kxi = torch.clone(images)
            kxi.requires_grad = True
            for j in range(40):
                output = est_net(kxi)
                loss1 = F.cross_entropy(output, labels)
                loss2 = torch.sum((kxi - images) ** 2) / 2000
                loss = loss1 - gamma * loss2
                loss.backward()
                kxi.data = kxi.data + 0.1 * kxi.grad.data
                kxi.grad.data.zero_()
            loss_tmp.append(loss.item())
            if count==1:
              break
        est_loss = np.mean(loss_tmp)
        train_loss.append(est_loss)

        ## compute accuracy
        testloader = torch.utils.data.DataLoader(test_set, batch_size=2000, shuffle=False, num_workers=0)

        test_net = Net()
        test_net.load_state_dict(torch.load("tmp.pth"))
        correct = 0
        total = 0
        with torch.no_grad():
            for data in testloader:
                images, labels = data
                images = images.float()
                labels.float()
                outputs = test_net(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        acc.append(correct / total)

        ## test robust accuracy

        test_loader = torch.utils.data.DataLoader(test_set, batch_size=2000, shuffle=False, num_workers=0)
        test_loader = DeviceDataLoader(test_loader, device)

        test_net = Net()
        test_net.to(device)
        test_net.load_state_dict(torch.load("tmp.pth"))

        correct_robust = 0
        total_robust = 0
        for data in test_loader:
            images, labels = data
            kxi = torch.clone(images)
            kxi.requires_grad = True
            for j in range(10):
                output = test_net(kxi)
                loss1 = F.cross_entropy(output, labels)
                loss2 = torch.sum((kxi - images) ** 2) / 50
                loss = loss1 - gamma * loss2
                loss.backward()
                kxi.data = kxi.data + lr_1 * kxi.grad.data
                kxi.grad.data.zero_()

            with torch.no_grad():
                outputs = test_net(kxi)
                _, predicted = torch.max(outputs.data, 1)
                total_robust += labels.size(0)
                correct_robust += (predicted == labels).sum().item()
            break

        robust_acc.append(correct_robust / total_robust)

        print("train loss:", est_loss, "accuracy:", correct / total, "robust accuracy ", correct_robust / total_robust, "train time ", end_time-start_time)
        print("Epoch ends. Time spent:", time.time() - epoch_start_time)
        
    return train_loss, acc, robust_acc, sample_size, time_result