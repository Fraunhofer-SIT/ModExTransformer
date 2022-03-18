
# Apache 2.0 License
# Copyright (c) 2022, Fraunhofer e.V.
# All rights reserved.

import os

from torchvision.datasets import ImageFolder
from torchvision.datasets.utils import download_and_extract_archive


class TinyImageNet(ImageFolder):
    url = 'http://cs231n.stanford.edu/tiny-imagenet-200.zip'
    filename = "tiny-imagenet-200.zip"
    base_folder = 'tiny-imagenet-200'

    def __init__(self, root, split='train', transform=None, target_transform=None, download=False):
        self.root = root
        self.split = split

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError(
                f'Dataset not found at {self.root}. Please use download=True or manually download it from {self.url}')

        self.split_folder = os.path.join(self.root, self.base_folder, self.split)
        super(TinyImageNet, self).__init__(self.split_folder, transform, target_transform)

    def _check_integrity(self):
        return os.path.exists(os.path.join(self.root, self.base_folder))

    def download(self):
        if self._check_integrity():
            print("Files already downloaded.")
            return
        download_and_extract_archive(self.url, self.root, filename=self.filename)
