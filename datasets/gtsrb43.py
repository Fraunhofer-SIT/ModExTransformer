
# Apache 2.0 License
# Copyright (c) 2022, Fraunhofer e.V.
# All rights reserved.

import os

from torchvision.datasets import ImageFolder


class GTSRB43(ImageFolder):
    url = 'https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/published-archive.html'
    base_folder = 'GTSRB'

    def __init__(self, root=None, train=True, transform=None, target_transform=None):
        self.train = train
        self.root = root

        try:
            super().__init__(root=os.path.join(root, self.base_folder, 'Final_Training/Images' if train else 'Final_Test'),
                             transform=transform, target_transform=target_transform)
        except FileNotFoundError as e:
            raise FileNotFoundError(f'Dataset not found at {self.root}. Please download it from {self.url}.') from e

        if not train:
            with open(os.path.join(self.root, 'Images/GT-final_test.csv')) as f:
                self.targets = [int(line.split(';')[-1].strip()) for line in f.readlines()[1:]]
            self.samples = list(zip(next(zip(*self.samples)), self.targets))
            self.imgs = self.samples

        print('=> done loading {} ({}) with {} examples'.format(self.__class__.__name__, 'train' if train else 'test',
                                                                len(self.samples)))
