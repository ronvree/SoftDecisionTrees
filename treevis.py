import numpy as np
import os

from PIL import Image

try:
    os.makedirs('./node_vis')
except:
    print('directory ./node_vis already exists')


class VisNode:
    pass


class VisBranch(VisNode):
    pass


class VisLeaf(VisNode):
    pass


class VisTree:
    pass


class TreeVis:

    def __init__(self, tree: VisTree, image_shape: tuple):
        """
        Creates visualizations of the given tree by generating dot files
        :param tree: Tree which should be visualized
        :param image_shape: The shape of the input images. The tree needs to know this as the images are flattened when
                            passed through the model. The weight vectors need to be reshaped to the image size

        IMPORTANT NOTE -- GENERATING A TREE VISUALISATION BREAKS THE TREE. THIS IS A BUG WE FOUND ON VERY SHORT NOTICE
        BEFORE THE HAND-IN DEADLINE. WE DO KNOW FOR SURE THAT THIS BUG DID NOT AFFECT THE RESULTS DESCRIBED IN THE PAPER
        AS THE VISUALIZATIONS WERE NOT USED THERE.
        TODO - fix
        """
        self.tree = tree
        self.k = tree.k  # Number of output classes
        self.image_shape = image_shape

    def _branch_vis(self, node: VisBranch):
        [ws] = node.linear.weight.cpu().detach().numpy()

        ws += -min(ws)
        ws /= max(ws)
        ws *= 255

        ws = np.resize(ws, new_shape=self.image_shape)

        img = Image.new('F', ws.shape)
        pixels = img.load()

        for i in range(ws.shape[0]):
            for j in range(ws.shape[1]):
                pixels[i, j] = ws[i][j]

        cs = 64 // self.image_shape[0], 64 // self.image_shape[1]
        img = img.resize(size=(cs[0] * self.image_shape[0], cs[1] * self.image_shape[1]))

        return img

    def _leaf_vis(self, node: VisLeaf):
        ws = node.dist().cpu().detach().numpy()
        ws = np.ones(ws.shape) - ws
        ws *= 255

        img = Image.new('F', (ws.shape[0], ws.shape[0]))

        pixels = img.load()

        for i in range(ws.shape[0]):
            for j in range(ws.shape[0]):
                pixels[i, j] = ws[i]

        img = img.resize(size=(64, 64))

        return img

    def _node_vis(self, node: VisNode):
        if isinstance(node, VisLeaf):
            return self._leaf_vis(node)
        if isinstance(node, VisBranch):
            return self._branch_vis(node)

    def as_dot(self):
        s = 'digraph T {\n'
        s += 'node [shape=square, label=""];\n'
        s += self._gen_dot_nodes(self.tree.root)
        s += self._gen_dot_edges(self.tree.root)[0]
        s += '}\n'
        return s

    def save_dot(self, fn: str):
        with open(fn, 'w') as f:
            f.write(self.as_dot())

    def _gen_dot_nodes(self, node: VisNode, i: int = 0):
        img = self._node_vis(node).convert('RGB')
        filename = 'node_vis/node_{}_vis.jpg'.format(i)
        img.save(filename)
        s = '{}[image="{}"];\n'.format(i, filename)
        if isinstance(node, VisBranch):
            return s + self._gen_dot_nodes(node.l, i + 1) + self._gen_dot_nodes(node.r, i + node.l.size() + 1)
        if isinstance(node, VisLeaf):
            return s

    def _gen_dot_edges(self, node: VisNode, i: int = 0):
        if isinstance(node, VisBranch):
            edge_l, targets_l = self._gen_dot_edges(node.l, i + 1)
            edge_r, targets_r = self._gen_dot_edges(node.r, i + node.l.size() + 1)
            str_targets_l = ','.join(str(t) for t in targets_l) if len(targets_l) > 0 else ""
            str_targets_r = ','.join(str(t) for t in targets_r) if len(targets_r) > 0 else ""
            s = '{} -> {} [label="{}"];\n {} -> {} [label="{}"];\n'.format(i, i + 1, str_targets_l, i, i + node.l.size() + 1, str_targets_r)
            return s + edge_l + edge_r, sorted(list(set(targets_l + targets_r)))
        if isinstance(node, VisLeaf):
            ws = node.dist().cpu().detach().numpy()
            argmax = np.argmax(ws)
            targets = [argmax] if argmax.shape == () else argmax.tolist()
            return '', targets

