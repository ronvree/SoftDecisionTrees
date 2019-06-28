import numpy as np
import torch
import torch.optim
import torch.utils.data
import torchvision
import torchvision.transforms as transforms

from data_soft_targets import SoftMNIST


def get_mnist(soft=False, input_dimensions=1):
    shape = w, h = (28, 28)
    classes = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9)

    def flatten(x):
        return np.resize(x, w * h)

    def to_one_hot(y):
        return torch.zeros(len(classes)).scatter_(0, y, 1)

    if input_dimensions == 2:
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.1307,), (0.3081,)),
                                        ])
        target_transform = None
    else:
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.1307,), (0.3081,)),
                                        transforms.Lambda(flatten),
                                        ])

        target_transform = transforms.Lambda(to_one_hot)

    if soft:
        trainset = SoftMNIST(root='./data',
                             train=True,
                             download=True,
                             transform=transform
                             )

    else:
        trainset = torchvision.datasets.MNIST(root='./data',
                                              train=True,
                                              download=True,
                                              transform=transform,
                                              target_transform=target_transform
                                              )

    testset = torchvision.datasets.MNIST(root='./data',
                                         train=False,
                                         download=True,
                                         transform=transform,
                                         # target_transform=target_transform
                                         )

    return trainset, testset, classes, shape


if __name__ == '__main__':
    from PIL import Image

    _trainset, _testset, _classes, _shape = get_mnist()

    _trainloader = torch.utils.data.DataLoader(_trainset,
                                               batch_size=64,
                                               shuffle=True,
                                               )

    _testloader = torch.utils.data.DataLoader(_testset,
                                              batch_size=64,
                                              shuffle=False,
                                              )

    for _x_batch, _y_batch in _trainloader:
        for _x, _y in zip(_x_batch, _y_batch):
            _x = torch.reshape(_x, _shape)
            _x = _x.numpy()
            _x *= 255
            img = Image.new('F', _shape)

            pixels = img.load()

            for i in range(_shape[0]):  # TODO -- atm the images are rotated!
                for j in range(_shape[1]):
                    pixels[i, j] = _x[i][j]

            img.show()
            print(_y)
            input()
        break
