import os

import torch
from torch import nn, optim
from torch.nn import functional as F


class LeNet(nn.Module):
    def __init__(self, k: int, args):
        super(LeNet, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.args = args
        self.conv1 = nn.Conv2d(1, 6, 5, padding=2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, k)
        self.criterion = nn.CrossEntropyLoss(reduction='sum')
        self.optimizer = optim.Adam(self.parameters(), lr=1e-3, betas=(0.9, 0.99))
        self.apply(self.weight_init)
        self.test_acc = []
        self.best_accuracy = 0.0
        self.soft_targets = []

    def weight_init(self, m):
        if isinstance(m, nn.Conv2d):
            import math
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))

    def forward(self, x):
        # Max pooling over a (2, 2) window
        # print(x.shape)
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.softmax(x, dim=1)

    def forward_with_temperature(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.softmax(x / self.args.temperature, dim=1)

    def loss(self, data, target):
        output = self(data)
        loss = self.criterion(output, target)
        return loss, output

    def test_(self, test_loader, epoch):
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                if self.args.cuda:
                    data, target = data.cuda(), target.cuda()
                output = self(data)
                # sum up batch loss
                # get the index of the max log-probability
                predict = output.data.max(1, keepdim=True)[1]
                correct += predict.eq(target.data.view_as(predict)).sum().item()

        accuracy = 100.0 * correct / len(test_loader.dataset)
        print('\nTest Accuracy: {:.2f}%\n'.format(
            accuracy))
        self.test_acc.append(accuracy)

        if accuracy > self.best_accuracy:
            if self.args.save_model:
                self.save_best('./result')
            self.best_accuracy = accuracy

    def make_soft_labels(self, train_loader):
        correct = 0
        with torch.no_grad():
            for data, target in train_loader:
                if self.args.cuda:
                    data, target = data.cuda(), target.cuda()
                output = self(data)

                # sum up batch loss
                # get the index of the max log-probability
                predict = output.data.max(1, keepdim=True)[1]
                correct += predict.eq(target.data.view_as(predict)).sum().item()

        accuracy = 100.0 * correct / len(train_loader.dataset)
        print('\nTest Accuracy: {:.2f}%\n'.format(
            accuracy))

        if accuracy > self.best_accuracy:
            self.best_accuracy = accuracy

    def save_best(self, path):
        try:
            os.makedirs('./result')
        except:
            print('directory ./result already exists')

        with open(os.path.join(path, 'lenet5_best.json'), 'wb') as output_file:
            torch.save(self.state_dict(), output_file)
