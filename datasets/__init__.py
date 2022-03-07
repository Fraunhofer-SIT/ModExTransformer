import configparser

from timm.data import create_transform, IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torchvision import transforms
from torchvision.datasets import CIFAR10, CIFAR100, FashionMNIST, MNIST, SVHN

from .openimage import OpenImage
from .caltech256 import Caltech256
from .cubs200 import CUBS200
from .gtsrb43 import GTSRB43
from .imagenet import ImageNet, ImageNetSubset
from .indoor67 import Indoor67
from .tinyimagenet200 import TinyImageNet
from .thiefdataset import ThiefDataset


def get_dataset(cls, train=True, data_path=None, **kwargs):
    cls = cls.lower()
    if not data_path:
        print('The argument data_path is empty, so it is read from data_config.ini.')
        data_config = configparser.ConfigParser()
        with open('datasets/data_config.ini') as file:
            data_config.read_file(file)
        data_path = data_config[cls]['data_path']

    if cls == "caltech":
        dataset = Caltech256(root=data_path, train=train, **kwargs)

    if cls == 'openimage':
        dataset = OpenImage(root=data_path, **kwargs)

    elif cls == "cifar10":
        dataset = CIFAR10(root=data_path, train=train, download=True, **kwargs)

    elif cls == "cifar100":
        dataset = CIFAR100(root=data_path, train=train, download=True, **kwargs)

    elif cls == "cubs":
        dataset = CUBS200(root=data_path, train=train, **kwargs)

    elif cls == "fashionmnist":
        dataset = FashionMNIST(root=data_path, train=train, download=True, **kwargs)

    elif cls == "gtsrb":
        dataset = GTSRB43(root=data_path, train=train, **kwargs)

    elif cls == "imagenet":
        dataset = ImageNet(root=data_path, split='train' if train else 'val', **kwargs)

    elif cls.startswith("imagenet_subset"):
        indices_path = data_config[cls]['indices_path']
        dataset = ImageNetSubset(root=data_path, indices_path=indices_path, train=train, **kwargs)

    elif cls == "indoor":
        dataset = Indoor67(root=data_path, train=train, download=True, **kwargs)

    elif cls == "mnist":
        dataset = MNIST(root=data_path, train=train, download=True, **kwargs)

    elif cls == "svhn":
        dataset = SVHN(root=data_path, split='train' if train else 'test', download=True, **kwargs)

    elif cls == "tinyimagenet":
        dataset = TinyImageNet(root=data_path, split='train' if train else 'val', download=True, **kwargs)

    else:
        raise NotImplementedError('Only the following datasets are supported: '
                                  'caltech, cifar10, cifar100, cubs, fashionmnist, gtsrb, imagenet, '
                                  'imagenet_subset, indoor, mnist, openimage, svhn, tinyimagenet.')
    return dataset


def build_transform(is_train, input_size=224, **kwargs):
    resize_im = input_size > 32
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        # suppress the warning as this is a problem within the timm library
        import warnings
        warnings.filterwarnings(action='ignore', message='Argument interpolation')
        transform = create_transform(
            input_size=input_size,
            is_training=True,
            color_jitter=kwargs['color_jitter'],
            auto_augment=kwargs['aa'],
            interpolation=kwargs['train_interpolation'],
            re_prob=kwargs['reprob'],
            re_mode=kwargs['remode'],
            re_count=kwargs['recount'],
        )
        if not resize_im:
            # replace RandomResizedCropAndInterpolation with
            # RandomCrop
            transform.transforms[0] = transforms.RandomCrop(
                input_size, padding=4)
        return transform

    t = []
    if resize_im:
        size = int((256 / 224) * input_size)
        t.append(
            # to maintain same ratio w.r.t. 224 images
            transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC),
        )
        t.append(transforms.CenterCrop(input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
    return transforms.Compose(t)


def default_imagenet_transform(with_augmentation, size=(224, 224)):
    normalize = transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN,
                                     std=IMAGENET_DEFAULT_STD)
    if with_augmentation:
        return transforms.Compose([
            transforms.RandomResizedCrop(size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])

    else:
        return transforms.Compose([
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            normalize,
        ])
