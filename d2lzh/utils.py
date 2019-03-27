import collections
import math
import os
import random
import sys
import tarfile
import time
import zipfile
import torch
import numpy as np

from IPython import display
from matplotlib import pyplot as plt
from torchvision import transforms
from torchvision.datasets import FashionMNIST
from torch.utils import data

def data_iter(batch_size, features, labels):
    """Iterate through a data set."""
    num_examples = len(features)
    indices = list(range(num_examples))
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        j = torch.tensor(indices[i: min(i + batch_size, num_examples)])
        yield features[j], labels[j]
        
def linreg(X, w, b):
    """Linear regression."""
    return X @ w + b

def sgd(params, lr, batch_size):
    """Mini-batch stochastic gradient descent."""
    for param in params:
        param.data = param.data - lr * param.grad / batch_size
        
def use_svg_display():
    """Use svg format to display plot in jupyter"""
    display.set_matplotlib_formats('svg')

def set_figsize(figsize=(3.5, 2.5)):
    use_svg_display()
    plt.rcParams['figure.figsize'] = figsize
    
def show_fashion_mnist(images, labels):
    use_svg_display()
    # 这里的_表示我们忽略（不使用）的变量
    _, figs = plt.subplots(1, len(images), figsize=(12, 12))
    for f, img, lbl in zip(figs, images, labels):
        f.imshow(img.reshape((28, 28)).numpy())
        f.set_title(lbl)
        f.axes.get_xaxis().set_visible(False)
        f.axes.get_yaxis().set_visible(False)
        
def load_data_fashion_mnist(batch_size, resize=None, root=os.path.join('~', 'datasets')):
    """Download the fashion mnist dataset and then load into memory."""
    root = os.path.expanduser(root)
    if resize is None:
        transformer = transforms.ToTensor()
    else:
        transformer = transforms.Compose([
            transforms.Resize(resize),
            transforms.ToTensor()
        ])
    mnist_train = FashionMNIST(root, train=True, transform=transformer)
    mnist_test = FashionMNIST(root, train=False, transform=transformer)
    num_workers = 0 if sys.platform.startswith('win32') else 4
    train_iter = data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_iter = data.DataLoader(mnist_test, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_iter, test_iter

def evaluate_accuracy(data_iter, net):
    acc_sum, n = 0.0, 0
    for X, y in data_iter:
        acc_sum += (net(X).argmax(dim=1) == y).float().sum().item()
        n += y.shape[0]
    return acc_sum / n

def get_fashion_mnist_labels(labels):
    """Get text label for fashion mnist."""
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]

def train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size, params=None, lr=None, optimizer=None):
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n = 0.0, 0.0, 0
        for X, y in train_iter:
            y_hat = net(X)
            l = loss(y_hat, y).sum()
            if optimizer is not None:
                optimizer.zero_grad()
            elif params is not None and params[0].grad is not None:
                for param in params:
                    param.grad.zero_()
            l.backward()
            if optimizer is None:
                sgd(params, lr, batch_size)
            else:
                optimizer.step()
            train_l_sum += l.item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).float().sum().item()
            n += y.shape[0]
        test_acc = evaluate_accuracy(test_iter, net)
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f' % (epoch + 1, train_l_sum / n, train_acc_sum / n, test_acc))