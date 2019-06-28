import argparse
import random

import torch

import sdt_frosst_hinton as sdt


class SDTDropout(sdt.SoftDecisionTree):

    def __init__(self, k: int, in_features: int, args: argparse.Namespace):
        self.dropout = args.dropout
        super().__init__(k, in_features, args)

    def _init_tree(self, i: int, d: int, args: argparse.Namespace):
        """
        Construct the Soft Decision Tree
        :param d: The depth of the tree
        :return: a complete Soft Decision Tree of depth d
        """
        if d == 1:
            return Leaf(i, self.k, args)
        else:
            left = self._init_tree(i + 1, d - 1, args)
            return Branch(i,
                          left,
                          self._init_tree(i + left.size() + 1, d - 1, args),
                          self.in_features,
                          self.lamb * 2 ** -(self.depth - d),  # Lambda is proportional to 2 ** -d
                          self.dropout * 2 ** -(d - 2),  # Scale the dropout with the depth of tree
                          args
                          )

    def loss(self, xs, ys, **kwargs):
        """
        Compute the mean loss for all data/label pairs in the train data
        :param xs: Train data batch. shape: (bs, w * h)
        :param ys: Train label batch. shape: (bs, k)
        :return: a two-tuple consisting of
                    - a Tensor containing the computed loss
                    - a Tensor containing the output distributions for all x in xs
        """
        loss, out = self.root.loss(xs, ys)

        return -loss.mean(), out

    def print_tree(self):
        self.root.pretty_print(0)


class Branch(sdt.Branch):
    def __init__(self, i: int, l: sdt.Node, r: sdt.Node, in_features: int, lamb: float, dropout: float,
                 args: argparse.Namespace):
        super().__init__(i, l, r, in_features, lamb, args)
        # Set the dropout parameter
        self.dropout = dropout

    def loss(self, xs, ys, **kwargs):
        """
        Compute the loss based on the train data batch xs and train labels batch ys
        :param xs: Batch of data points to compute the loss on. shape: (bs, w * h)
        :param ys: Batch of true labels to compute the loss on. shape: (bs, k)
        :param kwargs: Dictionary containing optional named arguments. Used to store the following:
        :return: a three-tuple containing
            - a tensor with the loss values for each data/label pair
            - a tensor with the output distributions
            """
        z = self.linear(xs)  # shape: (bs, 1)
        # Remove redundant dimension
        z = z.view(-1)  # shape: (bs,)

        # Generate a random variable
        dm = random.random()

        if dm > self.dropout:
            # Apply the sigmoid function to obtain the probability of choosing the right subtree for all x in xs
            ps = torch.sigmoid(sdt.SoftDecisionTree.BETA * z)  # shape: (bs,)
            # Obtain the unweighted loss/output values from the child nodes
            l_loss, l_out = self.l.loss(xs, ys)  # loss shape: (bs,), out shape: (bs, k)
            r_loss, r_out = self.r.loss(xs, ys)  # loss shape: (bs,), out shape: (bs, k)
            # Weight the loss values by their path probability (by element wise multiplication)

            w_loss = (1 - ps) * l_loss + ps * r_loss  # shape: (bs,)
            # Weight the output values by their path probability
            ps = ps.view(xs.shape[0], 1)
            w_out = (1 - ps) * l_out + ps * r_out  # shape: (bs,)
            return w_loss, w_out
        else:
            return self.r.loss(xs, ys)  # loss shape: (bs,), out shape: (bs, k)

    def pretty_print(self, depth):
        print(depth * '-' + "Branch({}, {})".format(self.index, self.dropout))
        self.l.pretty_print(depth + 1)
        self.r.pretty_print(depth + 1)


class Leaf(sdt.Leaf):
    def pretty_print(self, depth):
        print(depth * '-' + "Leaf({})".format(self.index))

    def loss(self, xs, ys, **kwargs):
        loss, out, _ = super().loss(xs, ys)
        return loss, out
