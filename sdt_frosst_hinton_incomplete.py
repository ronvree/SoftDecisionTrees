import torch
import torch.nn
import torch.autograd
import torch.optim
import torch.cuda
import torch.nn.functional as func

import argparse

from treevis import VisNode, VisLeaf, VisTree


class SoftDecisionTree(torch.nn.Module, VisTree):
    """
        Soft Decision Tree implementation as described in 'Distilling a Neural Network Into a Soft Decision Tree' by
        Nicholas Frosst and Geoffrey Hinton of the Google Brain Team

        This version does NOT support:
          - the regularization term proposed in the paper
          - classification by picking the maximum path probability (only weighted version)

        However, this version was kept as it is a simpler variant of SDTs which makes it more readable. The complete
        version is pretty much the same, but some properties need to be kept track of during the forward pass of the
        model, thus adding a layer of complexity to the code.
    """

    # Inverse temperature parameter that is multiplied with all decision node filters
    BETA = 1.0

    # Different modes in which the tree can generate its output
    MODES = ['prob',  # Node output is weighted with the probability of arriving at that node
             'max',   # Node output is taken from the node with highest probability (NOT SUPPORTED)
             ]
    MODE = MODES[0]

    def __init__(self,
                 k: int,
                 in_features: int,
                 args: argparse.Namespace,
                 ):
        """
        Create a new Soft Decision Tree
        :param k: The number of output labels
        :param in_features: The size of the decision node input data
        :param args: Parser arguments containing hyperparameters
        """
        super(SoftDecisionTree, self).__init__()
        assert args.depth > 0
        assert k > 0

        self.depth = args.depth
        self.k = k

        # self.cuda = not args.disable_cuda and torch.cuda.is_available()

        self.num_nodes = 2 ** self.depth - 1
        self.num_leaves = 2 ** (self.depth - 1)

        self.root = self._init_tree(self.depth, in_features)

    def _init_tree(self, d: int, in_features: int):
        """
        Construct the Soft Decision Tree
        :param d: The depth of the tree
        :param in_features: The input size of the decision nodes
        :return: a complete Soft Decision Tree of depth d
        """
        if d == 1:
            return Leaf(self.k)
        else:
            return Branch(self._init_tree(d - 1, in_features),
                          self._init_tree(d - 1, in_features),
                          in_features)

    def forward(self, xs, **kwargs):
        """
        Perform a forward pass for all data samples in the batch
        :param xs: The batch of data samples. shape: (bs, w * h)
        :return: a Tensor with an output distribution over all output classes for each data point
        """
        return self.root.forward(xs)  # shape: (bs, k)

    def loss(self, xs, ys):
        """
        Compute the mean loss for all data/label pairs in the train data
        :param xs: Train data batch. shape: (bs, w * h)
        :param ys: Train label batch. shape: (bs, k)
        :return: a two-tuple consisting of
                    - a Tensor containing the computed loss
                    - a Tensor containing the output distributions for all x in xs
        """
        loss, out = self.root.loss(xs, ys)
        return torch.neg(loss).mean(), out  # loss shape: (1,), output shape: (bs, k)


class Node(torch.nn.Module):

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def loss(self, xs, ys, **kwargs):
        raise NotImplementedError

    def size(self):
        raise NotImplementedError


class Leaf(Node, VisLeaf):

    def __init__(self, k: int):
        """
        Create a new Soft Decision Tree leaf that contains a probability distribution over all k output classes
        :param k: The number of output classes
        """
        assert k > 0
        super(Leaf, self).__init__()
        self.dist_params = torch.nn.Parameter(torch.randn(k))

    def forward(self, xs, **kwargs):
        """
        Obtain probability distributions for all x in xs
        :param xs: Batch of data points. shape: (bs, w * h)
        :return: a Tensor containing identical probability distributions for all data points
        """
        # Obtain the leaf's distribution (which is independent of the input data)
        dist = self.dist()  # shape: (k,)
        # Reshape the distribution to a matrix with one single row
        dist = dist.view(1, -1)  # shape: (1, k)
        # Duplicate the row for all x in xs
        return torch.cat((dist,) * xs.shape[0], 0)  # shape: (bs, k)

    def dist(self):
        """
        Apply a softmax function to the leaf's distribution parameters
        :return: a Tensor containing a probability distribution over all labels
        """
        return func.softmax(self.dist_params, dim=0)  # shape: (k,)

    def loss(self, xs, ys, **kwargs):
        """
        Compute the loss based on the train data batch xs and train labels batch ys
        :param xs: Batch of data points to compute the loss on. shape: (bs, w * h)
        :param ys: Batch of true labels to compute the loss on. shape: (bs, k)
        :return: a two-tuple containing
                    - a tensor with the loss values for each data/label pair
                    - a tensor with the output distributions
        """
        # Obtain the leaf's distribution for each x in xs
        dists = self.forward(xs)  # shape: (bs, k)
        # Compute the log of the distribution values  (log Q_k^l that is)
        log_dists = torch.log(dists)  # shape: (bs, k)
        # Reshape target distributions for batch matrix multiplication
        ys = ys.view(ys.shape[0], 1, -1)  # shape: (bs, 1, k)
        # Reshape log distributions for batch matrix multiplication
        log_dists = log_dists.view(xs.shape[0], -1, 1)  # shape: (bs, k, 1)
        # Multiply all target distributions with the leaf's distribution (for all x in xs)
        tqs = torch.bmm(ys, log_dists)  # shape: (bs, 1, 1)
        # Remove redundant dimensions
        return tqs.view(-1), dists  # loss shape: (bs,), out shape: (bs, k)

    def size(self):
        """
        Obtain the number of nodes in this subtree
        :return: 1, as this is a leaf
        """
        return 1


class Branch(Node, VisNode):

    def __init__(self, l: Node, r: Node, in_features: int):
        """
        Create a new Soft Decision Tree decision node
        :param l: The left subtree of this node
        :param r: The right subtree of this node
        :param in_features: The input size of the linear module
        """
        super(Branch, self).__init__()
        self.l, self.r = l, r

        self.linear = torch.nn.Linear(in_features=in_features,
                                      out_features=1)

    def forward(self, xs, **kwargs):
        """
        Do a forward pass for all data samples in the batch
        :param xs: The batch of data. shape: (bs, w * h)
        :return: a Tensor with probability distributions corresponding to all x in xs
        """
        # Apply the decision node's linear function to all x in xs
        z = self.linear(xs)  # shape: (bs, 1)
        # Apply the sigmoid function to obtain the probability of choosing the right subtree for all x in xs
        ps = torch.sigmoid(SoftDecisionTree.BETA * z)  # shape: (bs, 1)
        # Obtain the unweighted probability distributions from the child nodes
        l_dists = self.l.forward(xs)  # shape: (bs, k)
        r_dists = self.r.forward(xs)  # shape: (bs, k)
        # Weight the probability distributions by the decision node's output
        return (1 - ps) * l_dists + ps * r_dists  # shape: (bs, k)

    def loss(self, xs, ys, **kwargs):
        """
        Compute the loss based on the train data batch xs and train labels batch ys
        :param xs: Batch of data points to compute the loss on. shape: (bs, w * h)
        :param ys: Batch of true labels to compute the loss on. shape: (bs, k)
        :return: a two-tuple consisting of
            - a Tensor containing the computed loss
            - a Tensor containing the output distributions for all x in xs
        """
        # Apply the decision node's linear function to all x in xs
        z = self.linear(xs)  # shape: (bs, 1)
        # Remove redundant dimension
        z = z.view(-1)  # shape: (bs,)
        # Apply the sigmoid function to obtain the probability of choosing the right subtree for all x in xs
        ps = torch.sigmoid(SoftDecisionTree.BETA * z)  # shape: (bs,)
        # Obtain the unweighted loss/output values from the child nodes
        l_loss, l_out = self.l.loss(xs, ys)  # loss shape: (bs,), out shape: (bs, k)
        r_loss, r_out = self.r.loss(xs, ys)  # loss shape: (bs,), out shape: (bs, k)
        # Weight the loss values by their path probability (by element wise multiplication)
        w_loss = (1 - ps) * l_loss + ps * r_loss  # shape: (bs,)
        # Weight the output values by their path probability
        ps = ps.view(xs.shape[0], 1)
        w_out = (1 - ps) * l_out + ps * r_out  # shape: (bs,)
        return w_loss, w_out

    def size(self):
        """
        Obtain the number of nodes in this subtree
        :return: 1 + the number of nodes in both children
        """
        return 1 + self.l.size() + self.r.size()
