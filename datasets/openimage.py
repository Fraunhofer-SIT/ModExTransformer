import os

import numpy as np
from torchvision.datasets import ImageFolder


class OpenImage(ImageFolder):
    url = 'https://storage.googleapis.com/openimages/web/index.html'
    base_folder = "OpenImageDataset/"#OIDv6_100k"  

    def __init__(self, root, split='', transform=None, target_transform=None):
        self.root = root
        self.split_folder = os.path.join(self.root, self.base_folder)
        try:
            super(OpenImage, self).__init__(self.split_folder, transform, target_transform)
        except FileNotFoundError as e:
            raise FileNotFoundError(f'Dataset not found at {self.root}. Please download it from {self.url}.') from e