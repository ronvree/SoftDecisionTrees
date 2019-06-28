import torch
import torch.cuda
import torch.optim
import torch.utils.data

import argparse
import pickle

from data import get_mnist
from sdt_alternative import SoftDecisionTree

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='SoftDecisionTree on MNIST')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=10)  # Does not influence anything here
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--momentum', type=float, default=0.5)
    parser.add_argument('--lamb', type=float, default=0.1)
    parser.add_argument('--gamma', type=float, default=0.001)
    parser.add_argument('--disable_cuda', type=bool, default=True)
    parser.add_argument('--log_interval', type=int, default=20)
    parser.add_argument('--test_interval', type=int, default=1)

    args = parser.parse_args()
    cuda = not args.disable_cuda and torch.cuda.is_available()
    device = torch.device('cuda' if cuda else 'cpu')

    trainset, testset, classes, shape = get_mnist(soft=True)
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
                            args=args)

    proceed = True
    while proceed:  # TODO -- termination criterion

        if cuda:
            tree.cuda()

        optimizer = torch.optim.SGD(tree.parameters(), lr=args.lr, momentum=args.momentum)

        epoch = 1
        previous_loss_sum = float('inf')
        converged = False

        tree.train()
        while not converged:  # TODO -- proper convergence criterion
            loss_sum = torch.zeros(1, device=device)
            for i, (xs, ys) in enumerate(trainloader):
                if cuda:
                    xs, ys = xs.cuda(), ys.cuda()

                loss, out, info = tree.loss(xs, ys)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                ys_true = torch.argmax(ys, dim=1)
                ys_pred = torch.argmax(out, dim=1)
                correct = torch.sum(torch.eq(ys_pred, ys_true))

                if i % args.log_interval == 0:
                    print('Epoch: {:3d}, Batch {:3d}/{}, Loss: {:.5f}, Reg: {:.5f}, Accuracy: {:.5f}'.format(
                        epoch,
                        i,
                        len(trainloader),
                        loss.item(),
                        info['C'].item() if 'C' in info.keys() else 0,
                        correct.item() / len(xs))
                    )
                loss_sum += loss

            converged = abs(loss_sum - previous_loss_sum) < args.gamma * previous_loss_sum
            previous_loss_sum = loss_sum
            epoch += 1

        if tree.size() % args.test_interval == 0:
            tree.eval()
            correct, total = 0, 0
            for i, (x_batch, y_batch_true) in enumerate(testloader):
                if cuda:
                    x_batch, y_batch_true = x_batch.cuda(), y_batch_true.cuda()

                y_batch_pred = torch.argmax(tree.forward(x_batch), dim=1)

                correct += torch.sum(torch.eq(y_batch_pred, y_batch_true)).item()
                total += x_batch.shape[0]

            print('Accuracy: {}\n\n\n\n'.format(correct / total))

        proceed = True  # TODO

        with open('tree{}_acc{:.5f}.pickle'.format(tree.size(), correct / total), 'wb') as f:
            pickle.dump(tree, f)

        tree.expand(args)
