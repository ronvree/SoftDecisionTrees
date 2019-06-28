import argparse

import torch.utils.data

from data import get_mnist
from lenet import LeNet

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='SoftDecisionTree on MNIST')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--depth', type=int, default=7)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--disable-cuda', action='store_true',
                        help='Disable CUDA')
    parser.add_argument('--log_interval', type=int, default=20)

    args = parser.parse_args()
    cuda = not args.disable_cuda and torch.cuda.is_available()

    trainset, testset, classes, shape = get_mnist(input_dimensions=2)
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

    model = LeNet(k=len(classes), args=args)

    if cuda:
        model.cuda()

    for epoch in range(1, args.epochs + 1):
        model.train()

        for i, (x_batch, y_batch) in enumerate(trainloader):
            if cuda:
                x_batch, y_batch = x_batch.cuda(), y_batch.cuda()

            model.optimizer.zero_grad()

            # forward + backward + optimize
            output = model(x_batch)
            loss = model.criterion(output, y_batch)
            loss.backward()
            model.optimizer.step()

            if i % args.log_interval == 0:
                print('Epoch: {:3d}, Batch {:3d}/{}, Loss: {:.5f}'.format(
                    epoch,
                    i,
                    len(trainloader),
                    loss.item()
                ))

        model.eval()
        correct, total = 0, 0
        for i, (x_batch, y_batch_true) in enumerate(testloader):
            if cuda:
                x_batch, y_batch_true = x_batch.cuda(), y_batch_true.cuda()

            y_batch_pred = torch.argmax(model.forward(x_batch), dim=1)

            correct += torch.sum(torch.eq(y_batch_pred, y_batch_true)).item()
            total += x_batch.shape[0]

        print('Accuracy: {}\n\n\n\n'.format(correct / total))
