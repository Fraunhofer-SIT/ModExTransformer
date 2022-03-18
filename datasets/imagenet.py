
# Apache 2.0 License
# Copyright (c) 2022, Fraunhofer e.V.
# All rights reserved.

import os

import numpy as np
from torchvision.datasets import ImageFolder


class ImageNet(ImageFolder):
    url = 'https://image-net.org/'
    base_folder = "imagenet2012/ILSVRC/Data/CLS-LOC/"

    def __init__(self, root, split='train', transform=None, target_transform=None):
        self.root = root
        self.split = split

        self.split_folder = os.path.join(self.root, self.base_folder, self.split)

        try:
            super(ImageNet, self).__init__(self.split_folder, transform, target_transform)
        except FileNotFoundError as e:
            raise FileNotFoundError(f'Dataset not found at {self.root}. Please download it from {self.url}.') from e


class ImageNetSubset(ImageNet):

    def __init__(self, root=None, train=True, indices_path=None, transform=None, target_transform=None):
        super(ImageNetSubset, self).__init__(root, split='train' if train else 'val', transform=transform,
                                             target_transform=target_transform)

        indices = np.load(indices_path)
        if not train:
            # sample from imagenet val dataset
            indices = np.random.choice(range(len(self.targets)), int(len(indices) / 10), replace=False)
        self.samples = [self.samples[i] for i in indices]
        self.imgs = self.samples
        self.targets = [self.targets[i] for i in indices]
