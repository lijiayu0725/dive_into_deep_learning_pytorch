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
from torch import nn
from torch.nn import init
from torch.nn import functional as F
from torch import optim

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
        
def load_data_fashion_mnist(batch_size, resize=None, root=os.path.join('~', 'datasets/FashionMNIST')):
    """Download the fashion mnist dataset and then load into memory."""
    root = os.path.expanduser(root)
    if resize is None:
        transformer = transforms.Compose([
            transforms.ToTensor()
        ])
    else:
        transformer = transforms.Compose([
            transforms.Resize(resize),
            transforms.ToTensor()
        ])
    mnist_train = FashionMNIST(root, train=True, transform=transformer, download=False)
    mnist_test = FashionMNIST(root, train=False, transform=transformer, download=False)
    num_workers = 0 if sys.platform.startswith('win32') else 4
    train_iter = data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_iter = data.DataLoader(mnist_test, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_iter, test_iter

def get_fashion_mnist_labels(labels):
    """Get text label for fashion mnist."""
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]

class FlattenLayer(nn.Module):
    def __init__(self):
        super(FlattenLayer, self).__init__()
    def forward(self, x): # x shape: (batch, *, *, ...)
        return x.view(x.shape[0], -1)

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
        
def semilogy(x_vals, y_vals, x_label, y_label, x2_vals=None, y2_vals=None, legend=None, figsize=(3.5, 2.5)):
    set_figsize(figsize)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.semilogy(x_vals, y_vals)
    if x2_vals and y2_vals:
        plt.semilogy(x2_vals, y2_vals, linestyle=':')
        plt.legend(legend)
        
def fit_and_plot(train_features, test_features, train_labels, test_labels):
    net = nn.Linear(train_features.shape[1], 1)
    init.normal_(net.weight)
    batch_size = min(10, train_labels.shape[0])
    train_iter = data.DataLoader(data.TensorDataset(train_features, train_labels), batch_size, shuffle=True)
    optimizer = optim.SGD(net.parameters(), lr=0.01)
    train_ls, test_ls = [], []
    for _ in range(num_epochs):
        for X, y in train_iter:
            l = loss(net(X), y.view(-1, 1))
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
        train_labels = train_labels.view(-1, 1)
        test_labels = test_labels.view(-1, 1)
        train_ls.append(loss(net(train_features), train_labels).item())
        test_ls.append(loss(net(test_features), test_labels).item())
    print('final epoch: train loss', train_ls[-1], 'test loss', test_ls[-1])
    semilogy(range(1, num_epochs + 1), train_ls, 'epochs', 'loss',
             range(1, num_epochs + 1), test_ls, ['train', 'test'])
    print('weight:', net.weight.data, '\nbias:', net.bias.data)
    
def squared_loss(y_hat, y): 
    return (y_hat - y.view(y_hat.size())) ** 2 / 2

def corr2d(X, K):
    h, w = K.shape
    Y = torch.zeros(X.shape[0] - h + 1, X.shape[1] - w + 1)
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j] = (X[i: i + h, j: j + w] * K).sum()
    return Y

def train_ch5(net, train_iter, test_iter, batch_size, optimizer, device, num_epochs):
    print('training on', device)
    net = net.to(device)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n, start = 0.0, 0.0, 0, time.time()
        for X, y in train_iter:
            X, y = X.to(device), y.to(device)
            
            y_hat = net(X)
            l = criterion(y_hat, y)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            train_l_sum += l.item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).data.sum().item()
            n += y.shape[0]
        test_acc = evaluate_accuracy(test_iter, net, device)
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f, '
              'time %.1f sec'
              % (epoch + 1, train_l_sum / n, train_acc_sum / n, test_acc,
                 time.time() - start))
        
def evaluate_accuracy(data_iter, net, device):
    acc_sum, n = 0.0, 0
    with torch.no_grad():
        for X, y in data_iter:
            X = X.to(device)
            y = y.to(device)
            acc_sum += (torch.argmax(net(X), dim=1) == y).float().sum().item()
            n += y.shape[0]
    return acc_sum / n

class GlobalAvgPool2d(nn.Module):
    def __init__(self):
        super(GlobalAvgPool2d, self).__init__()
    def forward(self, x):
        return F.avg_pool2d(x, kernel_size=x.shape[2:])
    
def show_trace_2d(f, results):  # 本函数将保存在d2lzh包中方便以后使用
    plt.plot(*zip(*results), '-o', color='#ff7f0e')
    x1, x2 = np.meshgrid(np.arange(-5.5, 1.0, 0.1), np.arange(-3.0, 1.0, 0.1))
    plt.contour(x1, x2, f(x1, x2), colors='#1f77b4')
    plt.xlabel('x1')
    plt.ylabel('x2')
    
def train_2d(optimizer):
    x1, x2, s1, s2 = -5, -2, 0, 0
    result = [(x1, x2)]
    for i in range(20):
        x1, x2, s1, s2 = optimizer(x1, x2, s1, s2)
        result.append((x1, x2))
    print('epoch %d, x1 %f, x2 %f' % (i + 1, x1, x2))
    return result

def get_data_ch7():
    data = np.genfromtxt('data/airfoil_self_noise.dat', delimiter='\t')
    data = (data - data.mean(axis=0)) / data.std(axis=0)
    return torch.tensor(data[:1500, :-1], dtype=torch.float), torch.tensor(data[:1500, -1], dtype=torch.float)

def train_ch7(trainer_fn, states, hyperparams, features, labels, batch_size=10, num_epochs=2):
    net, loss = linreg, squared_loss
    w = torch.randn((features.shape[1], 1)) * 0.01
    b = torch.zeros(1)
    w.requires_grad=True
    b.requires_grad=True
    
    def eval_loss():
        return loss(net(features, w, b), labels).mean().item()
    
    ls = [eval_loss()]
    data_iter =  data.DataLoader(data.TensorDataset(features, labels), batch_size, shuffle=True)
    for _ in range(num_epochs):
        start = time.time()
        for batch_i, (X, y) in enumerate(data_iter):
            if w.grad is not None:
                w.grad.zero_()
            if b.grad is not None:
                b.grad.zero_()
            l = loss(net(X, w, b), y).mean() # 使⽤平均损失
            l.backward()
            trainer_fn([w, b], states, hyperparams) # 迭代模型参数
            if (batch_i + 1) * batch_size % 100 == 0:
                ls.append(eval_loss()) # 每100个样本记录下当前训练误差
    # 打印结果和作图
    print('loss: %f, %f sec per epoch' % (ls[-1], time.time() - start))
    set_figsize()
    plt.plot(np.linspace(0, num_epochs, len(ls)), ls)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    
def train_pytorch_ch7(lr, features, labels, batch_size=10, op='SGD', num_epochs=2, momentum=0, alpha=1):
    # 初始化模型
    
    net = nn.Sequential(
        nn.Linear(features.shape[1], 1)
    )
    
    init.normal_(net[0].weight, std=0.01)
    loss = torch.nn.MSELoss()

    def eval_loss():
        with torch.no_grad():
            l = loss(net(features).squeeze(), labels).item()
        return l
      
    ls = [eval_loss() / 2]

    data_iter = data.DataLoader(data.TensorDataset(features, labels), batch_size=batch_size, shuffle=True)
    # 创建Trainer实例来迭代模型参数
    if op == 'sgd':
        optimizer = optim.SGD(net.parameters(), lr=lr, momentum=momentum)
    elif op == 'adagrad':
        optimizer = optim.Adagrad(net.parameters(), lr=lr)
    elif op == 'rmsprop':
        optimizer = optim.RMSprop(net.parameters(), lr=lr, alpha=alpha)
    elif op == 'adadelta':
        optimizer = optim.Adadelta(net.parameters(), rho=lr)
    elif op == 'adam':
        optimizer = optim.Adam(net.parameters(), lr=lr)
    for _ in range(num_epochs):
        start = time.time()
        for batch_i, (X, y) in enumerate(data_iter):
            net.zero_grad()
            y_hat = net(X)
            l = loss(y_hat.squeeze(), y)
            l.backward()
            optimizer.step() # 在Trainer实例里做梯度平均
            if (batch_i + 1) * batch_size % 100 == 0:
                ls.append(eval_loss() / 2)
               
    # 打印结果和作图
    print('loss: %f, %f sec per epoch' % (ls[-1], time.time() - start))
    set_figsize()
    plt.plot(np.linspace(0, num_epochs, len(ls)), ls)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    
def show_images(imgs, num_rows, num_cols, scale=2):
    figsize = (num_cols * scale, num_rows * scale)
    _, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
    for i in range(num_rows):
        for j in range(num_cols):
            axes[i][j].imshow(imgs[i * num_cols + j])
            axes[i][j].axes.get_xaxis().set_visible(False)
            axes[i][j].axes.get_yaxis().set_visible(False)
    return axes

def train(train_iter, test_iter, net, criterion, optimizer, device, num_epochs):
    """Train and evaluate a model."""
    print('training on', device)
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n, m, start = 0.0, 0.0, 0, 0, time.time()
        net.train()
        for i, (imgs, labels) in enumerate(train_iter):
            
            imgs = imgs.to(device)
            labels = labels.to(device)
            
            y_hat = net(imgs)
            l = criterion(y_hat, labels)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            train_l_sum += l.item()
            
            train_acc_sum += (torch.argmax(y_hat, dim=1) == labels).float().sum().item()
            n += imgs.shape[0]
        net.eval()
        test_acc = evaluate_accuracy(test_iter, net, device)
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f, '
              'time %.1f sec'
              % (epoch + 1, train_l_sum / n, train_acc_sum / n, test_acc,
                 time.time() - start))
        
def bbox_to_rect(bbox, color):  # 本函数已保存在d2lzh包中方便以后使用
    # 将边界框(左上x, 左上y, 右下x, 右下y)格式转换成matplotlib格式：
    # ((左上x, 左上y), 宽, 高)
    return plt.Rectangle(
        xy=(bbox[0], bbox[1]), width=bbox[2]-bbox[0], height=bbox[3]-bbox[1],
        fill=False, edgecolor=color, linewidth=2)

def MultiBoxPrior(X, sizes, ratios):
    
    h, w = X.shape[-2:]
    
    n = len(sizes)
    m = len(ratios)
    count = 0
    res = np.zeros((1, h*w*(n + m - 1), 4))
    if w <= h:
        t = w
    else:
        t = h
    for i in range(h):
        for j in range(w):
            for l in range(len(sizes)):
                s = sizes[l]
                r = ratios[0]
                w_a = t * s * math.sqrt(r)
                h_a = t * s / math.sqrt(r)
                left_x = j - w_a / 2
                left_y = i - h_a / 2
                right_x = j + w_a / 2
                right_y = i + h_a / 2
                res[:, count, 0] = left_x / w
                res[:, count, 1] = left_y / h
                res[:, count, 2] = right_x / w
                res[:, count, 3] = right_y / h
                count += 1
            for k in range(len(ratios)):
                if k == 0:
                    continue
                s = sizes[0]
                r = ratios[k]
                w_a = t * s * math.sqrt(r)
                h_a = t * s / math.sqrt(r)
                left_x = j - w_a / 2
                left_y = i - h_a / 2
                right_x = j + w_a / 2
                right_y = i + h_a / 2
                res[:, count, 0] = left_x / w
                res[:, count, 1] = left_y / h
                res[:, count, 2] = right_x / w
                res[:, count, 3] = right_y / h
                count += 1
    print(count)            
    return res

def show_bboxes(axes, bboxes, labels=None, colors=None):
    def _make_list(obj, default_values=None):
        if obj is None:
            obj = default_values
        elif not isinstance(obj, (list, tuple)):
            obj = [obj]
        return obj
    
    labels = _make_list(labels)
    colors = _make_list(colors, ['b', 'g', 'r', 'm', 'c'])
    for i, bbox in enumerate(bboxes):
        color = colors[i % len(colors)]
        rect = bbox_to_rect(bbox, color)
        axes.add_patch(rect)
        if labels and len(labels) > i:
            text_color = 'k' if color == 'w' else 'w'
            axes.text(rect.xy[0], rect.xy[1], labels[i],
                      va='center', ha='center', fontsize=9, color=text_color,
                      bbox=dict(facecolor=color, lw=0))