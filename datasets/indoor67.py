
# Apache 2.0 License
# Copyright (c) 2022, Fraunhofer e.V.
# All rights reserved.

import os

from torchvision.datasets import ImageFolder
from torchvision.datasets.utils import download_and_extract_archive, download_url


class Indoor67(ImageFolder):
    """Taken and modified from https://github.com/tribhuvanesh/knockoffnets/blob/master/knockoff/datasets/indoor67.py"""
    url = 'http://web.mit.edu/torralba/www/indoor.html'
    base_folder = 'indoorCVPR_09'

    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
        self.train = train
        self.root = root

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError(
                f'Dataset not found at {self.root}. Please use download=True or manually download it from {self.url}')

        # Initialize ImageFolder
        super().__init__(root=os.path.join(self._raw_folder(), 'Images'), transform=transform,
                         target_transform=target_transform)

        self.root = os.path.dirname(self.root)

        self.partition_to_idxs = self.get_partition_to_idxs()
        self.pruned_idxs = self.partition_to_idxs['train' if train else 'test']

        # Prune (self.imgs, self.samples to only include examples from the required train/test partition
        self.samples = [self.samples[i] for i in self.pruned_idxs]
        self.imgs = self.samples
        self.targets = [self.targets[i] for i in self.pruned_idxs]

        print('=> done loading {} ({}) with {} examples'.format(self.__class__.__name__, 'train' if train else 'test',
                                                                len(self.samples)))

    def get_partition_to_idxs(self):
        partition_to_idxs = {
            'train': [],
            'test': []
        }

        # ----------------- Load list of train images
        test_images = set()
        with open(os.path.join(self.root, 'TestImages.txt')) as f:
            for line in f:
                test_images.add(line.strip())

        for idx, (filepath, _) in enumerate(self.samples):
            filepath = filepath.replace(os.path.join(self.root, 'Images') + '/', '')
            if filepath in test_images:
                partition_to_idxs['test'].append(idx)
            else:
                partition_to_idxs['train'].append(idx)

        return partition_to_idxs

    def _raw_folder(self):
        return os.path.join(self.root, self.base_folder)

    def _check_integrity(self):
        return os.path.exists(self._raw_folder())

    def download(self):
        if self._check_integrity():
            print("Files already downloaded.")
            return
        download_and_extract_archive('http://groups.csail.mit.edu/vision/LabelMe/NewImages/indoorCVPR_09.tar',
                                     self._raw_folder())
        download_url('http://web.mit.edu/torralba/www/TestImages.txt', self._raw_folder())
