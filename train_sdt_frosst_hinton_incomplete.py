import torch
import torch.optim
import torch.utils.data
import torch.cuda

import argparse

from sdt_frosst_hinton_incomplete import SoftDecisionTree
from data import get_mnist

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='SoftDecisionTree on MNIST')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--depth', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--momentum', type=float, default=0.5)
    parser.add_argument('--disable_cuda', type=bool, default=True)
    parser.add_argument('--log_interval', type=int, default=10)  # TODO -- not supported yet

    args = parser.parse_args()
    cuda = not args.disable_cuda and torch.cuda.is_available()

    trainset, testset, classes, shape = get_mnist()
    w, h = shape

    trainloader = torch.utils.data.DataLoader(trainset,
                                              batch_size=args.batch_size,
                                              shuffle=True,
                                              pin_memory=cuda
                                              )

    testloader = torch.utils.data.DataLoader(testset,
                                             batch_size=args.batch_size,
                                             shuffle=False,
                                             pin_memory=cuda
                                             )

    tree = SoftDecisionTree(k=len(classes),
                            in_features=w * h,
                            args=args
                            )

    if cuda:
        tree.cuda()

    optimizer = torch.optim.SGD(tree.parameters(), lr=args.lr, momentum=args.momentum)

    for epoch in range(1, args.epochs + 1):
        tree.train()

        for i, (x_batch, y_batch) in enumerate(trainloader):
            if cuda:
                x_batch, y_batch = x_batch.cuda(), y_batch.cuda()

            loss, out = tree.loss(x_batch, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            ys_true = torch.argmax(y_batch, dim=1)
            ys_pred = torch.argmax(out, dim=1)
            correct = torch.sum(torch.eq(ys_pred, ys_true))

            print('Epoch: {:3d}, Batch {:3d}/{}, Loss: {:.5f}, Accuracy: {:.5f}'.format(
                epoch,
                i,
                len(trainloader),
                loss.item(),
                correct.item() / len(x_batch))
            )

        tree.eval()
        correct, total = 0, 0
        for i, (x_batch, y_batch_true) in enumerate(testloader):
            if cuda:
                x_batch, y_batch_true = x_batch.cuda(), y_batch_true.cuda()

            y_batch_pred = torch.argmax(tree.forward(x_batch), dim=1)

            correct += torch.sum(torch.eq(y_batch_pred, y_batch_true)).item()
            total += x_batch.shape[0]

        print('Accuracy: {}\n\n\n\n'.format(correct / total))

