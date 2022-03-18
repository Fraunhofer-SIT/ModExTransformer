
# Apache 2.0 License
# Copyright (c) 2022, Fraunhofer e.V.
# All rights reserved.

import ast
import argparse
import configparser

def get_value_with_correct_type(x):
    try:
        return ast.literal_eval(x)
    except:
        return x

def get_dict_values_with_correct_type(dct):
    return {k: get_value_with_correct_type(v) for k, v in dct.items()}

def get_args_parser():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--batch_size', default=50, type=int,
                        help='default: %(default)d')
    parser.add_argument('--epochs', default=100, type=int,
                        help='default: %(default)d')
    parser.add_argument('--output_dir', default='checkpoints',
                        help='path where to save (default: %(default)s)')
    parser.add_argument('--checkpoint_frequency', default=25, type=int,
                        help='number of epochs until new checkpoint (default: %(default)d)')
    parser.add_argument('--pretrained', action='store_true',
                        help='use pretrained model (default: %(default)s)')
    parser.add_argument('--no-pretrained', action='store_false', dest='pretrained')
    parser.set_defaults(pretrained=True)
    parser.add_argument('--augmentation', action='store_true',
                        help='use augmentation (default: %(default)s)')
    parser.add_argument('--no-augmentation', action='store_false', dest='augmentation')
    parser.set_defaults(augmentation=True)
    parser.add_argument('--resume', default='',
                        help="resume from checkpoint (default: don't resume)")
    parser.add_argument('--weights_path', default='',
                        help="path where weights are saved (default: don't use weighted sampling)")
    parser.add_argument('--num_samples', default=None, type=int,
                        help='number of samples (only for weighted sampling; default: None, use whole dataset)')
    parser.add_argument('--num_classes', default=None, type=int,
                        help='number of classes (default: None, infer from dataset or target model)')
    parser.add_argument('--with_eval', action='store_true',
                        help="evaluate during training (default: don't eval)")
    parser.add_argument('--log_target_predictions', action='store_true',
                        help='log target predictions (default)')
    parser.add_argument('--no-log_target_predictions', action='store_false', dest='log_target_predictions')
    parser.set_defaults(log_target_predictions=False)
    parser.add_argument('--softmax_target', action='store_true',
                        help='apply, if target has softmax output. For defenses will be set automatically.')
    parser.add_argument('--sample_seed', default=1, type=int,
                        help='sample seed. Only used for weighted sampling. (default: %(default)d)')

    # Model parameters
    model_parser = parser.add_argument_group('model arguments')
    model_parser.add_argument('--model', default='deit_base_patch16_224', type=str, metavar='MODEL',
                              help='Name of model to train (default: %(default)s)')
    model_parser.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                              help='Dropout rate (default: %(default)f')
    model_parser.add_argument('--drop_path', type=float, default=0.1, metavar='PCT',
                              help='Drop path rate (default: %(default)f)')
    model_parser.add_argument('--model_ema', action='store_true')
    model_parser.add_argument('--no-model_ema', action='store_false', dest='model_ema')
    model_parser.set_defaults(model_ema=True)
    model_parser.add_argument('--model_ema_decay', type=float, default=0.99996, metavar='PCT',
                              help='default: %(default)f')
    model_parser.add_argument('--model_ema_force_cpu', action='store_true')

    # Dataset parameters
    dataset_parser = parser.add_argument_group('dataset arguments')
    dataset_parser.add_argument('--data_path', default=None, type=str,
                                help='dataset path (default: %(default)s)')
    dataset_parser.add_argument('--dataset', default='cifar10',
                                type=str, help='dataset name (default: %(default)s)')
    dataset_parser.add_argument('--input_size', default=224, type=int,
                                help='image size (default: %(default)d)')

    # Distillation parameters
    distillation_parser = parser.add_argument_group('distillation arguments')
    distillation_parser.add_argument('--target_model', default='resnet34', type=str, metavar='MODEL',
                                     help='Name of target model to train. (default: %(default)s')
    distillation_parser.add_argument('--target_path', type=str, default='',
                                     help='path of the target model checkpoint.')
    distillation_parser.add_argument('--distillation_type', default='none', choices=['none', 'soft', 'hard'], type=str,
                                     help='default: %(default)s')
    distillation_parser.add_argument('--distillation_alpha', default=1.0, type=float,
                                     help='must be 1 for only distillation loss. (default: %(default)f)')
    distillation_parser.add_argument('--distillation_tau', default=3.0, type=float,
                                     help='temperature (default: %(default)f)')

    # Defense parameters
    defense_parser = parser.add_argument_group('defense arguments')
    defense_parser.add_argument('--defense', default='none', type=str,
                                choices=['none', 'random_noise', 'reverse_sigmoid', 'mad'],
                                help='defense to apply (default: %(default)s)')
    defense_parser.add_argument('--defense_log_path', default='', type=str,
                                help='path where to save defense logs (default: no log')
    defense_parser.add_argument('--mad_epsilon', default=0.5, type=float,
                                help='epsilon value of mad defense (default: %(default)f)')
    defense_parser.add_argument('--mad_oracle', default='extreme', type=str,
                                help='oracle used by mad defense (default: %(default)s)')
    defense_parser.add_argument('--rs_beta', default=0.7, type=float,
                                help='beta value of reverse sigmoid defense (default: %(default)f)')
    defense_parser.add_argument('--rs_gamma', default=0.4, type=float,
                                help='gamma value of reverse sigmoid defense (default: %(default)f)')
    defense_parser.add_argument('--rn_dist_z', default='l1', type=str,
                                help='norm used by random noise defense (default: %(default)s)')
    defense_parser.add_argument('--rn_epsilonz', default=0.5, type=float,
                                help='epsilon_z value of random noise defense (default: %(default)f)')

    # Optimizer parameters
    optimizer_parser = parser.add_argument_group('optimizer arguments')
    optimizer_parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                                  help='Optimizer (default: %(default)s)')
    optimizer_parser.add_argument('--opt_eps', default=1e-8, type=float, metavar='EPSILON',
                                  help='Optimizer Epsilon (default: %(default)f)')
    optimizer_parser.add_argument('--opt_betas', default=None, type=float, nargs='+', metavar='BETA',
                                  help='Optimizer Betas (default: None, use opt default)')
    optimizer_parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM',
                                  help='Clip gradient norm (default: None, no clipping)')
    optimizer_parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                                  help='SGD momentum (default: %(default)f)')
    optimizer_parser.add_argument('--weight_decay', type=float, default=0.05,
                                  help='weight decay (default: %(default)f)')

    # Learning rate schedule parameters
    scheduler_parser = parser.add_argument_group('scheduler arguments')
    scheduler_parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER',
                                  help='LR scheduler (default: %(default)s)')
    scheduler_parser.add_argument('--lr', type=float, default=5e-4, metavar='LR',
                                  help='learning rate (default: %(default)f)')
    scheduler_parser.add_argument('--lr_noise', type=float, nargs='+', default=None, metavar='pct, pct',
                                  help='learning rate noise on/off epoch percentages')
    scheduler_parser.add_argument('--lr_noise_pct', type=float, default=0.67, metavar='PERCENT',
                                  help='learning rate noise limit percent (default: %(default)f)')
    scheduler_parser.add_argument('--lr_noise_std', type=float, default=1.0, metavar='STDDEV',
                                  help='learning rate noise std-dev (default: %(default)f)')
    scheduler_parser.add_argument('--warmup_lr', type=float, default=1e-6, metavar='LR',
                                  help='warmup learning rate (default: %(default)f)')
    scheduler_parser.add_argument('--min_lr', type=float, default=1e-5, metavar='LR',
                                  help='lower lr bound for cyclic schedulers that hit 0 (default: %(default)f)')
    scheduler_parser.add_argument('--decay_epochs', type=float, default=30, metavar='N',
                                  help='epoch interval to decay LR (default: %(default)d)')
    scheduler_parser.add_argument('--warmup_epochs', type=int, default=5, metavar='N',
                                  help='epochs to warmup LR, if scheduler supports (default: %(default)d)')
    scheduler_parser.add_argument('--cooldown_epochs', type=int, default=10, metavar='N',
                                  help='epochs to cooldown LR at min_lr, after cyclic schedule ends (default: %('
                                       'default)d)')
    scheduler_parser.add_argument('--patience_epochs', type=int, default=10, metavar='N',
                                  help='patience epochs for Plateau LR scheduler (default: %(default)d)')
    scheduler_parser.add_argument('--decay_rate', '--dr', type=float, default=0.1, metavar='RATE',
                                  help='LR decay rate (default: %(default)f)')

    # Augmentation parameters
    augmentation_parser = parser.add_argument_group('augmentation arguments')
    augmentation_parser.add_argument('--color_jitter', type=float, default=0.4, metavar='PCT',
                                     help='Color jitter factor (default: %(default)f)')
    augmentation_parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                                     help='Use AutoAugment policy. "v0" or "original". (default: %(default)s)')
    augmentation_parser.add_argument('--smoothing', type=float, default=0.1,
                                     help='Label smoothing (default: %(default)f)')
    augmentation_parser.add_argument('--train_interpolation', type=str, default='bicubic',
                                     choices=['random', 'bilinear', 'bicubic'],
                                     help='default: %(default)s')
    augmentation_parser.add_argument('--repeated_aug', action='store_true')
    augmentation_parser.add_argument('--no-repeated_aug', action='store_false', dest='repeated_aug')
    augmentation_parser.set_defaults(repeated_aug=True)
    # * Random Erase params
    augmentation_parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                                     help='Random erase prob (default: %(default)f)')
    augmentation_parser.add_argument('--remode', type=str, default='pixel',
                                     help='Random erase mode (default: %(default)s)')
    augmentation_parser.add_argument('--recount', type=int, default=1,
                                     help='Random erase count (default: %(default)d)')
    augmentation_parser.add_argument('--resplit', action='store_true', default=False,
                                     help='Do not random erase first (clean) augmentation split')
    # * Mixup params
    augmentation_parser.add_argument('--mixup', type=float, default=0.0,
                                     help='mixup alpha, mixup enabled if > 0. (default: %(default)f)')
    augmentation_parser.add_argument('--cutmix', type=float, default=0.0,
                                     help='cutmix alpha, cutmix enabled if > 0. (default: %(default)f)')
    augmentation_parser.add_argument('--cutmix_minmax', type=float, nargs='+', default=None,
                                     help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: '
                                          'None)')
    augmentation_parser.add_argument('--mixup_prob', type=float, default=1.0,
                                     help='Probability of performing mixup or cutmix when either/both is enabled ('
                                          'default: %(default)f)')
    augmentation_parser.add_argument('--mixup_switch_prob', type=float, default=0.5,
                                     help='Probability of switching to cutmix when both mixup and cutmix enabled ('
                                          'default: %(default)f)')
    augmentation_parser.add_argument('--mixup_mode', type=str, default='batch', choices=['batch', 'pair', 'elem'],
                                     help='How to apply mixup/cutmix params. (default: %(default)s)')

    # Dataloader parameters
    dataloader_parser = parser.add_argument_group('dataloader arguments')
    dataloader_parser.add_argument('--device', default='cuda',
                                   help='device to use for training / testing (default: %(default)s)')
    dataloader_parser.add_argument('--num_workers', default=4, type=int,
                                   help='default: %(default)d')
    dataloader_parser.add_argument('--pin_mem', action='store_true',
                                   help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    dataloader_parser.add_argument('--no-pin_mem', action='store_false', dest='pin_mem')
    dataloader_parser.set_defaults(pin_mem=True)

    # other parameters
    other_parser = parser.add_argument_group('other arguments')
    other_parser.add_argument('--finetune', default='',
                              help='finetune from checkpoint')
    other_parser.add_argument('--world_size', default=1, type=int,
                              help='number of distributed processes (default: %(default)d)')
    other_parser.add_argument('--dist_url', default='env://',
                              help='url used to set up distributed training (default: %(default)s)')
    other_parser.add_argument('--seed', default=0, type=int,
                              help='default: %(default)d')
    other_parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                              help='start epoch (default: %(default)d)')
    other_parser.add_argument('--eval', action='store_true',
                              help='Perform evaluation only')
    other_parser.add_argument('--dist_eval', action='store_true', default=False,
                              help='Enabling distributed evaluation')

    return parser

# only for checking if config file is there
config_file_parser = argparse.ArgumentParser(add_help=False)
config_file_parser.add_argument('--config', '-c', nargs='+', help='path(s) to configuration file(s)')
args, _ = config_file_parser.parse_known_args()

# the actual parser
parser = argparse.ArgumentParser('DeiT training script', parents=[config_file_parser, get_args_parser()])

if args.config is not None:
    # override default argparse values
    config = configparser.ConfigParser(allow_no_value=True, inline_comment_prefixes='#')
    for filename in args.config:
        with open(filename) as file:
            config.read_file(file)
    parser.set_defaults(**get_dict_values_with_correct_type(config.defaults()))

args = parser.parse_args()
