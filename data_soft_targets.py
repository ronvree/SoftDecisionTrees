import os
import pickle

from PIL import Image
from torchvision import datasets


class SoftMNIST(datasets.MNIST):
    def __init__(self, root, import_targets=True, init_target_transform=None, train=True, transform=None, target_transform=None, download=False):
        super().__init__(root, train, transform, target_transform, download)
        self.init_target_transform = init_target_transform
        if import_targets:
            filename = os.path.join(root, 'soft_targets.pickle')
            f = open(filename, 'rb')
            self.train_labels = pickle.load(f)
            f.close()
        else:
            self.train_labels = [self.get_soft_label(x) for x in self.train_data]
            filename = os.path.join(root, 'soft_targets.pickle')
            f = open(filename, 'wb')
            pickle.dump(self.train_labels, f)
            f.close()

    def get_soft_label(self, img):
        img = Image.fromarray(img.numpy(), mode='L')
        if self.transform is not None:
            img = self.transform(img)
        return self.init_target_transform(img)