import torch
import torch.nn
import torch.cuda
import torch.autograd
import torch.optim
import torch.nn.functional as func

import argparse

from treevis import VisTree, VisLeaf, VisBranch


class SoftDecisionTree(torch.nn.Module, VisTree):
    """
        Soft Decision Tree implementation as described in 'Distilling a Neural Network Into a Soft Decision Tree' by
        Nicholas Frosst and Geoffrey Hinton of the Google Brain Team

    """

    # Inverse temperature parameter that is multiplied with all decision node filters
    BETA = 1.0

    # Different modes in which the tree can generate its output
    MODES = ['prob',  # Node output is weighted with the probability of arriving at that node
             'max',   # Output is taken from the leaf with highest path probability
             ]
    MODE = MODES[1]

    def __init__(self,
                 k: int,
                 in_features: int,
                 args: argparse.Namespace):
        """
        Create a new Soft Decision Tree
        :param d: Depth of the tree
        :param k: The number of output labels
        :param in_features: The size of the decision node input data
        :param args: Parsed arguments containing hyperparameters
        """
        super(SoftDecisionTree, self).__init__()
        assert args.depth > 0
        assert k > 0

        self.depth = args.depth
        self.k = k
        self.in_features = in_features
        self.lamb = args.lamb

        self.use_cuda = not args.disable_cuda and torch.cuda.is_available()

        self.num_nodes = 2 ** self.depth - 1
        self.num_leaves = 2 ** (self.depth - 1)

        self.root = self._init_tree(0, self.depth, args)

        self.nodes, self.leaves = self._get_nodes_leaves(self.root)

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
                          args
                          )  # TODO -- reverse depth for more readable code

    def _get_nodes_leaves(self, node):
        """
        Obtain two lists containing the tree's decision nodes and leaves, respectively
        :param node: The root node to start searching
        :return: a two-tuple consisting of
                    - a list containing all decision nodes
                    - a list containing all leaves
        """
        if isinstance(node, Leaf):
            return [], [node]
        if isinstance(node, Branch):
            nsl, lsl = self._get_nodes_leaves(node.l)
            nsr, lsr = self._get_nodes_leaves(node.r)
            return [node] + nsl + nsr, lsl + lsr

    def forward(self, xs, **kwargs):
        """
        Perform a forward pass for all data samples in the batch
        Depending on SoftDecisionTree.MODE the function has different behaviour:
            - prob: Node output is weighted with the probability of arriving at that node
            - max:  Output is taken from the leaf with highest path probability
        :param xs: The batch of data samples. shape: (bs, w * h)
        :param kwargs: Dictionary containing optional named arguments. Used to store the following:
                        - attr: A dict used to store these attributes for each node in the tree:
                                - pa: probability of arriving at this node
                                - ps: decision node output probabilities (only stored for decision nodes)
                                - ds: output distributions of the leaf node (only stored for leaf nodes)
        :return: a Tensor with an output distribution over all output classes for each data point
        """
        out, attr = self.root.forward(xs)

        if SoftDecisionTree.MODE is 'prob':
            return out
        if SoftDecisionTree.MODE is 'max':
            # Obtain path probabilities of arriving at each leaf
            pas = [attr[l.index, 'pa'].view(-1, 1) for l in self.leaves]  # All shaped (bs, 1)
            # Obtain output distributions of each leaf
            dss = [attr[l.index, 'ds'].view(-1, 1, self.k) for l in self.leaves]  # All shaped (bs, 1, k)
            # Prepare data for selection of most probable distributions
            # Let L denote the number of leaves in this tree
            pas = torch.cat(tuple(pas), dim=1)  # shape: (bs, L)
            dss = torch.cat(tuple(dss), dim=1)  # shape: (bs, L, k)
            # Select indices of leafs with highest path probability
            ix = torch.argmax(pas, dim=1).long()  # shape: (bs,)
            # Select distributions of leafs with highest path probability
            dists = []
            for j, i in zip(range(dss.shape[0]), ix):
                dists += [dss[j][i].view(1, -1)]  # All shaped (1, k)
            dists = torch.cat(tuple(dists), dim=0)  # shape: (bs, k)
            return dists
        raise Exception('Invalid value for Tree.MODE!')

    def loss(self, xs, ys, **kwargs):
        """
        Compute the mean loss for all data/label pairs in the train data
        :param xs: Train data batch. shape: (bs, w * h)
        :param ys: Train label batch. shape: (bs, k)
        :return: a two-tuple consisting of
                    - a Tensor containing the computed loss
                    - a Tensor containing the output distributions for all x in xs
        """
        loss, out, attr = self.root.loss(xs, ys)

        info = dict()

        if len(self.nodes) == 0 or self.lamb == 0:
            return -loss.mean(), out, info

        alphas = dict()
        # Compute alpha for each of the decision nodes
        for n in self.nodes:
            # Obtain the path probability of arriving at this decision node
            pa = attr[n.index, 'pa']  # shape: (bs,)
            # Obtain the output probability of this decision node
            ps = attr[n.index, 'ps']  # shape: (bs,)
            # Compute alpha as defined in the paper
            alphas[n.index] = torch.div(torch.sum(pa * ps), torch.sum(pa))
        # Compute the regularization term C using all alphas
        cs = dict()
        for n in self.nodes:
            cs[n.index] = 0.5 * n.lamb * (torch.log(alphas[n.index]) + torch.log(1 - alphas[n.index]))
        C = -torch.sum(torch.cat(tuple(c.view(1, 1) for c in cs.values()), 0))

        info['pa'] = pa
        info['ps'] = ps
        info['alphas'] = alphas
        info['C'] = C

        return -loss.mean() + C, out, info

    def size(self):
        """
        :return: The number of nodes in the tree
        """
        return self.root.size()


class Node(torch.nn.Module):

    def __init__(self, i: int):
        super(Node, self).__init__()
        self.index = i  # Keep a unique index for each node in the tree

    def forward(self, xs, **kwargs):
        raise NotImplementedError

    def loss(self, xs, ys, **kwargs):
        raise NotImplementedError

    def size(self):
        raise NotImplementedError


class Leaf(Node, VisLeaf):

    def __init__(self, i: int, k: int, args: argparse.Namespace):
        """
        Create a new Soft Decision Tree leaf that contains a probability distribution over all k output classes
        :param i: Unique index for this node
        :param k: The number of output classes
        :param args: Parsed arguments containing hyperparameters
        """
        assert k > 0
        super(Leaf, self).__init__(i)
        self.dist_params = torch.nn.Parameter(torch.randn(k))
        self.use_cuda = not args.disable_cuda and torch.cuda.is_available()

    def dist(self):
        """
        Apply a softmax function to the leaf's distribution parameters
        :return: a Tensor containing a probability distribution over all labels
        """
        return func.softmax(self.dist_params, dim=0)

    def forward(self, xs, **kwargs):
        """
        Obtain probability distributions for all x in xs
        :param xs: Batch of data points. shape: (bs, w * h)
        :param kwargs: Dictionary containing optional named arguments. Used to store the following:
                - attr: A dict used to store these attributes for each node in the tree:
                        - pa: probability of arriving at this node
                        - ps: decision node output probabilities (only stored for decision nodes)
                        - ds: output distributions of the leaf node (only stored for leaf nodes)
        :return: a two-tuple consisting of:
                    - a Tensor containing identical probability distributions for all data points
                    - a dictionary of attributes stored for each node during computation
        """
        device = torch.device('cuda' if self.use_cuda else 'cpu')
        # Keep a dict to assign attributes to nodes
        node_attr = kwargs.setdefault('attr', dict())
        # Set the probability of arriving at this node (if not set before)
        node_attr.setdefault((self.index, 'pa'), torch.ones(xs.shape[0], device=device, dtype=torch.float32))
        # Obtain the leaf's distribution (which is independent of the input data)
        dist = self.dist()  # shape: (k,)
        # Reshape the distribution to a matrix with one single row
        dist = dist.view(1, -1)  # shape: (1, k)
        # Duplicate the row for all x in xs
        dists = torch.cat((dist,) * xs.shape[0], 0)  # shape: (bs, k)
        # Store leaf distributions as node property
        node_attr[self.index, 'ds'] = dists
        # Return both the result of the forward pass as well as the node properties
        return dists, node_attr

    def loss(self, xs, ys, **kwargs):
        """
        Compute the loss based on the train data batch xs and train labels batch ys
        :param xs: Batch of data points to compute the loss on. shape: (bs, w * h)
        :param ys: Batch of true labels to compute the loss on. shape: (bs, k)
        :param kwargs: Dictionary containing optional named arguments. Used to store the following:
        - attr: A dict used to store these attributes for each node in the tree:
                - pa: probability of arriving at this node
                - ps: decision node output probabilities (only stored for decision nodes)
                - ds: output distributions of the leaf node (only stored for leaf nodes)
        :return: a three-tuple containing
                    - a tensor with the loss values for each data/label pair
                    - a tensor with the output distributions
                    - a dictionary of attributes stored for each node during computation
        """
        device = torch.device('cuda' if self.use_cuda else 'cpu')
        # Keep a dict to assign attributes to nodes
        node_attr = kwargs.setdefault('attr', dict())
        # Set the probability of arriving at this node (if not set before)
        node_attr.setdefault((self.index, 'pa'), torch.ones(xs.shape[0], device=device, dtype=torch.float32))
        # Obtain the leaf's distribution for each x in xs
        dists, _ = self.forward(xs, attr=node_attr)  # shape: (bs, k)
        # Store leaf distributions as node property
        node_attr[self.index, 'ds'] = dists
        # Compute the log of the distribution values  (log Q_k^l that is)
        log_dists = torch.log(dists)  # shape: (bs, k)
        # Reshape target distributions for batch matrix multiplication
        ys = ys.view(ys.shape[0], 1, -1)  # shape: (bs, 1, k)
        # Reshape log distributions for batch matrix multiplication
        log_dists = log_dists.view(xs.shape[0], -1, 1)  # shape: (bs, k, 1)
        # Multiply all target distributions with the leaf's distribution (for all x in xs)
        tqs = torch.bmm(ys, log_dists)  # shape: (bs, 1, 1)
        # Remove redundant dimensions
        return tqs.view(-1), dists, node_attr  # loss shape: (bs,), out shape: (bs, k)

    def size(self):
        """
        Obtain the number of nodes in this subtree
        :return: 1, as this is a leaf
        """
        return 1


class Branch(Node, VisBranch):

    def __init__(self,
                 i: int,
                 l: Node,
                 r: Node,
                 in_features: int,
                 lamb: float,
                 args: argparse.Namespace):
        """
        Create a new Soft Decision Tree decision node
        :param i: Unique index for this node
        :param l: The left subtree of this node
        :param r: The right subtree of this node
        :param in_features: The input size of the linear module
        :param lamb: Parameter controlling the infuence of regularization
                     Proportional to 2 ** -d (where d is depth of the tree)
        :param args: Parsed arguments containing hyperparameters
        """
        super(Branch, self).__init__(i)
        self.l, self.r = l, r
        self.lamb = lamb
        self.use_cuda = not args.disable_cuda and torch.cuda.is_available()

        self.linear = torch.nn.Linear(in_features=in_features,
                                      out_features=1
                                      )

    def forward(self, xs, **kwargs):
        """
        Do a forward pass for all data samples in the batch
        :param xs: The batch of data. shape: (bs, w * h)
        :param kwargs: Dictionary containing optional named arguments. Used to store the following:
                        - attr: A dict used to store these attributes for each node in the tree:
                                - pa: probability of arriving at this node
                                - ps: decision node output probabilities (only stored for decision nodes)
                                - ds: output distributions of the leaf node (only stored for leaf nodes)
        :return: a two-tuple consisting of:
            - a Tensor with probability distributions corresponding to all x in xs
            - a dictionary of attributes stored for each node during computation
        """
        device = torch.device('cuda' if self.use_cuda else 'cpu')
        # Keep a dict to assign attributes to nodes
        node_attr = kwargs.setdefault('attr', dict())
        # Set the probability of arriving at this node (if not set before)
        pa = node_attr.setdefault((self.index, 'pa'), torch.ones(xs.shape[0], device=device, dtype=torch.float32))
        # Apply the decision node's linear function to all x in xs
        z = self.linear(xs)  # shape: (bs, 1)
        # Apply the sigmoid function to obtain the probability of choosing the right subtree for all x in xs
        ps = torch.sigmoid(SoftDecisionTree.BETA * z)  # shape: (bs, 1)
        # Store decision node probabilities as node attribute
        node_attr[self.index, 'ps'] = ps
        # Store path probabilities of arriving at child nodes as node attributes
        node_attr[self.l.index, 'pa'] = (1 - ps.view(-1)) * pa
        node_attr[self.r.index, 'pa'] = ps.view(-1) * pa
        # Obtain the unweighted probability distributions from the child nodes
        l_dists, _ = self.l.forward(xs, attr=node_attr)  # shape: (bs, k)
        r_dists, _ = self.r.forward(xs, attr=node_attr)  # shape: (bs, k)
        # Weight the probability distributions by the decision node's output
        return (1 - ps) * l_dists + ps * r_dists, node_attr  # shape: (bs, k)

    def loss(self, xs, ys, **kwargs):
        """
        Compute the loss based on the train data batch xs and train labels batch ys
        :param xs: Batch of data points to compute the loss on. shape: (bs, w * h)
        :param ys: Batch of true labels to compute the loss on. shape: (bs, k)
        :param kwargs: Dictionary containing optional named arguments. Used to store the following:
                - attr: A dict used to store these attributes for each node in the tree:
                        - pa: probability of arriving at this node
                        - ps: decision node output probabilities (only stored for decision nodes)
                        - ds: output distributions of the leaf node (only stored for leaf nodes)
        :return: a three-tuple containing
            - a tensor with the loss values for each data/label pair
            - a tensor with the output distributions
            - a dictionary of attributes stored for each node during computation"""
        device = torch.device('cuda' if self.use_cuda else 'cpu')
        # Keep a dict to assign attributes to nodes
        node_attr = kwargs.setdefault('attr', dict())
        # Set the probability of arriving at this node (if not set before)
        pa = node_attr.setdefault((self.index, 'pa'), torch.ones(xs.shape[0], device=device, dtype=torch.float32))
        # Apply the decision node's linear function to all x in xs
        z = self.linear(xs)  # shape: (bs, 1)
        # Remove redundant dimension
        z = z.view(-1)  # shape: (bs,)
        # Apply the sigmoid function to obtain the probability of choosing the right subtree for all x in xs
        ps = torch.sigmoid(SoftDecisionTree.BETA * z)  # shape: (bs,)
        # Store decision node probabilities as node attribute
        node_attr[self.index, 'ps'] = ps
        # Store path probabilities of arriving at child nodes as node attributes
        node_attr[self.l.index, 'pa'] = (1 - ps) * pa
        node_attr[self.r.index, 'pa'] = ps * pa
        # Obtain the unweighted loss/output values from the child nodes
        l_loss, l_out, _ = self.l.loss(xs, ys, attr=node_attr)  # loss shape: (bs,), out shape: (bs, k)
        r_loss, r_out, _ = self.r.loss(xs, ys, attr=node_attr)  # loss shape: (bs,), out shape: (bs, k)
        # Weight the loss values by their path probability (by element wise multiplication)
        w_loss = (1 - ps) * l_loss + ps * r_loss  # shape: (bs,)
        # Weight the output values by their path probability
        ps = ps.view(xs.shape[0], 1)
        w_out = (1 - ps) * l_out + ps * r_out  # shape: (bs,)
        return w_loss, w_out, node_attr

    def size(self):
        """
        Obtain the number of nodes in this subtree
        :return: 1 + the number of nodes in both children
        """
        return 1 + self.l.size() + self.r.size()
