import os

import numpy as np
from torchvision.datasets import ImageFolder


class Caltech256(ImageFolder):
    """Taken and modified from https://github.com/tribhuvanesh/knockoffnets/blob/master/knockoff/datasets/caltech256.py"""
    url = 'http://www.vision.caltech.edu/Image_Datasets/Caltech256/'
    _google_drive_file_id = '1r6o0pSROcV1_VwT4oSjA2FBUSCWGuxLK'
    base_folder = 'caltech256'

    def __init__(self, root, train=True, transform=None, target_transform=None, ntest=25, seed=123):
        self.train = train
        self.seed = seed
        self.root = root

        try:
            super().__init__(root=os.path.join(self.root, self.base_folder, '256_ObjectCategories'), transform=transform,
                             target_transform=target_transform)
        except FileNotFoundError as e:
            raise FileNotFoundError(f'Dataset not found at {self.root}. Please download it from {self.url}.') from e

        self._cleanup()
        self.ntest = ntest  # Reserve these many examples per class for evaluation
        self.partition_to_idxs = self.get_partition_to_idxs()
        self.pruned_idxs = self.partition_to_idxs['train' if train else 'test']

        # Prune (self.imgs, self.samples to only include examples from the required train/test partition
        self.samples = [self.samples[i] for i in self.pruned_idxs]
        self.imgs = self.samples
        self.targets = [self.targets[i] for i in self.pruned_idxs]

        print('=> done loading {} ({}) with {} examples'.format(self.__class__.__name__, 'train' if train else 'test',
                                                                len(self.samples)))

    def _cleanup(self):
        # Remove examples belonging to class "clutter"
        clutter_idx = self.class_to_idx['257.clutter']
        self.samples = [s for s in self.samples if s[1] != clutter_idx]
        del self.class_to_idx['257.clutter']
        self.classes = self.classes[:-1]

    def get_partition_to_idxs(self):
        from collections import defaultdict as dd
        partition_to_idxs = {
            'train': [],
            'test': []
        }

        # Use this random seed to make partition consistent
        prev_state = np.random.get_state()
        np.random.seed(self.seed)

        # ----------------- Create mapping: classidx -> idx
        classidx_to_idxs = dd(list)
        for idx, s in enumerate(self.samples):
            classidx = s[1]
            classidx_to_idxs[classidx].append(idx)

        # Shuffle classidx_to_idx
        for classidx, idxs in classidx_to_idxs.items():
            np.random.shuffle(idxs)

        for classidx, idxs in classidx_to_idxs.items():
            partition_to_idxs['test'] += idxs[:self.ntest]  # A constant no. kept aside for evaluation
            partition_to_idxs['train'] += idxs[self.ntest:]  # Train on remaining

        # Revert randomness to original state
        np.random.set_state(prev_state)

        return partition_to_idxs