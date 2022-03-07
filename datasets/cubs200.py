import os

from torchvision.datasets import ImageFolder


class CUBS200(ImageFolder):
    """Taken and modified from https://github.com/tribhuvanesh/knockoffnets/blob/master/knockoff/datasets/cubs200.py"""
    url = 'http://www.vision.caltech.edu/visipedia/CUB-200-2011.html'
    _google_drive_file_id = '1hbzc_P1FuxMkcabkgn9ZKinBwW683j45'
    base_folder = 'CUB_200_2011'

    def __init__(self, root, train=True, transform=None, target_transform=None):
        self.train = train
        self.root = root

        if not os.path.exists(root):
            raise ValueError(f'Dataset not found at {root}. Please download it from {self.url}')

        try:
            super().__init__(root=os.path.join(self.root, self.base_folder, 'images'), transform=transform,
                             target_transform=target_transform)
        except FileNotFoundError as e:
            raise FileNotFoundError(f'Dataset not found at {self.root}. Please download it from {self.url}.') from e

        self.root = root

        self.partition_to_idxs = self.get_partition_to_idxs()
        self.pruned_idxs = self.partition_to_idxs['train' if train else 'test']

        # Prune self.imgs, self.samples to only include examples from the required train/test partition
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

        # ----------------- Create mapping: filename -> 'train' / 'test'
        # There are two files: a) images.txt containing: <imageid> <filepath>
        #            b) train_test_split.txt containing: <imageid> <0/1>

        imageid_to_filename = dict()
        with open(os.path.join(self.root, self.base_folder, 'images.txt')) as f:
            for line in f:
                imageid, filepath = line.strip().split()
                _, filename = os.path.split(filepath)
                imageid_to_filename[imageid] = filename
        filename_to_imageid = {v: k for k, v in imageid_to_filename.items()}

        imageid_to_partition = dict()
        with open(os.path.join(self.root, self.base_folder, 'train_test_split.txt')) as f:
            for line in f:
                imageid, split = line.strip().split()
                imageid_to_partition[imageid] = 'train' if int(split) else 'test'

        # Loop through each sample and group based on partition
        for idx, (filepath, _) in enumerate(self.samples):
            _, filename = os.path.split(filepath)
            imageid = filename_to_imageid[filename]
            partition_to_idxs[imageid_to_partition[imageid]].append(idx)

        return partition_to_idxs