import torch
import torch.optim
import torch.cuda

import argparse

import sdt_frosst_hinton as sdt
from sdt_frosst_hinton import Leaf, Branch
from treevis import VisTree


def entropy(ps):
    """
    Compute the entropy of the probability distribution
    :param ps: The probability distribution
    :return: the entropy
    """
    nonzero_ps = ps[torch.nonzero(ps)]
    return torch.neg(torch.sum(nonzero_ps * torch.log2(nonzero_ps)))


class SoftDecisionTree(sdt.SoftDecisionTree, VisTree):

    def __init__(self,
                 k: int,
                 in_features: int,
                 args: argparse.Namespace):
        """
        Soft Decision Tree implementation based on 'Distilling a Neural Network Into a Soft Decision Tree' by
        Nicholas Frosst and Geoffrey Hinton of the Google Brain Team. The model is extended by supporting 'expansion' of
        leaf nodes. That is, leaf nodes can be replaced by newly initialized decision nodes. This allows for alternative
        training procedures in which the model architecture is not fixed.
        :param k: Number of output classes
        :param in_features: Size of the input vectors
        :param args: Parsed arguments containing hyperparameters
        """
        args.depth = 1
        self.parents = dict()
        super().__init__(k, in_features, args)
        self.i = 1

    def _init_tree(self, i: int, d: int, args: argparse.Namespace):
        """
        Initialize the tree as a single leaf
        :param i: Superfluous index parameter
        :param d: Superfluous depth parameter
        :param args: Parsed arguments containing hyperparameters
        :return: the single initialized leaf
        """
        t = Leaf(0, self.k, args)
        self.parents[t] = None
        return t

    def _get_node_depths(self, node, d=1):
        """
        Obtain a dictionary mapping all nodes to their depth in the tree
        :param node: The subtree on which depths should be obtained
        :param d: The current depth
        :return: a dictionary mapping all nodes in the given subtree to their depths in the complete tree
        """
        if isinstance(node, Leaf):
            return {node: d}
        if isinstance(node, Branch):
            ds_l = self._get_node_depths(node.l, d + 1)
            ds_r = self._get_node_depths(node.r, d + 1)
            return {node: d, **ds_l, **ds_r}

    def _new_index(self):
        """
        Gives a fresh node index
        :return: a node index not used before
        """
        i = self.i
        self.i += 1
        return i

    def expand(self, args: argparse.Namespace):
        """
        Select a leaf and substitute it with a decision node and two child leaves.
        Selection is based on the leaf distribution entropy and its depth
        :param args: Parsed arguments containing hyperparameters
        """
        # Obtain all leaves currently in the tree, as well as their depths and distributions
        _, leaves = self._get_nodes_leaves(self.root)
        depths = self._get_node_depths(self.root)
        dists = {l: l.dist() for l in leaves}
        # Select a leaf to expand the tree at
        leaf = max(leaves, key=lambda l: entropy(dists[l]) * 2 ** -depths[l])
        depth = depths[leaf]
        left = self.parents[leaf] is not None and self.parents[leaf].l is leaf
        # Create a new subtree that replaces the selected leaf
        r = Leaf(self._new_index(), self.k, args)
        l = Leaf(self._new_index(), self.k, args)
        b = Branch(leaf.index,
                   l,
                   r,
                   self.in_features,
                   self.lamb * 2 ** -(depth - 1),  # Lambda is proportional to 2 ** -d
                   args
                   )  # TODO -- depth properly in both trees
        self.parents[r] = b
        self.parents[l] = b
        # Perform the replacement
        if self.parents[leaf] is None:
            self.root = b
        else:
            if left:
                self.parents[leaf].l = b
            else:
                self.parents[leaf].r = b
        # Update the remaining instance variables of the tree
        self.nodes, self.leaves = self._get_nodes_leaves(self.root)  # TODO -- depth as well, num nodes, num leaves
