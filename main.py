
""" Copied and modified from https://github.com/facebookresearch/deit/blob/main/main.py

 Modified passages are preceded and followed by #######################   """

# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.



import argparse
import datetime
import numpy as np
import time
import torch
import torch.backends.cudnn as cudnn
import json
from pathlib import Path

from timm.data import Mixup
from timm.models import create_model
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.scheduler import create_scheduler
from timm.optim import create_optimizer
from timm.utils import NativeScaler, get_state_dict, ModelEma

from datasets import build_transform
from engine import train_one_epoch, evaluate
from losses import DistillationLoss, KLDivLossWithTemperature
from samplers import RASampler
import models
import utils
from defenses.victim import MAD, ReverseSigmoid, RandomNoise
from datasets import get_dataset, ThiefDataset

"""
def get_args_parser() relocated to file 'basic_config.ini'
"""

def main(args):
    utils.init_distributed_mode(args)

    print(args)

    if args.distillation_type != 'none' and args.finetune and not args.eval:
        raise NotImplementedError("Finetuning with distillation not yet supported")

    if args.device == 'cuda:random':
        utils.get_free_gpu()
        args.device = 'cuda'  #  randomly choose a gpu
    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    dataset_train = get_dataset(args.dataset, train=True, data_path=args.data_path, transform=build_transform(args.augmentation, **vars(args)))
    dataset_val = get_dataset(args.dataset, train=False, data_path=args.data_path, transform=build_transform(False, **vars(args)))

#### Begin modifications
   
    if not args.num_samples:
        args.num_samples = len(dataset_train)
       
#### End modifications

    if args.mixup > 0.:
        # smoothing is handled with mixup label transform
        criterion = SoftTargetCrossEntropy()
    elif args.smoothing:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        criterion = torch.nn.CrossEntropyLoss()


    target_model = None
    if args.distillation_type != 'none':
        assert args.target_path, 'need to specify target-path when using distillation'
        print(f"Creating target model: {args.target_model}")

#### Begin modifications

        # Load full torch model
        if args.target_model == 'full':
            target_model = torch.load(args.target_path)
            if not args.num_classes:
                print('Infer number of classes from target model.')
                args.num_classes = utils.infer_num_classes_from_full_model(target_model)
        else:
            if args.target_path.startswith('https'):
                checkpoint = torch.hub.load_state_dict_from_url(
                    args.target_path, map_location='cpu', check_hash=True)
            else:
                checkpoint = torch.load(args.target_path, map_location='cpu')

            if not args.num_classes:
                print('Infer number of classes from target model.')
                args.num_classes = utils.infer_num_classes_from_model_checkpoint(checkpoint['model'])
                if not args.num_classes:
                    print('Could not infer from target model. Infer now from dataset.')
                    args.num_classes = utils.infer_num_classes_from_dataset(dataset_train)
            target_model = create_model(
                args.target_model,
                pretrained=False,
                num_classes=args.num_classes,
                global_pool='avg',
            )
            target_model.load_state_dict(checkpoint['model'])


        target_model.to(device)
        target_model.eval()


        if args.defense != 'none':
            args.softmax_target = True
            print(f'Applying {args.defense} defense.')
            if args.defense == 'random_noise':
                target_model = RandomNoise(
                    model=target_model,
                    out_path=args.defense_log_path,
                    dist_z=args.rn_dist_z,
                    epsilon_z=args.rn_epsilonz)
            elif args.defense == 'reverse_sigmoid':
                target_model = ReverseSigmoid(
                    model=target_model,
                    out_path=args.defense_log_path,
                    beta=args.rs_beta,
                    gamma=args.rs_gamma)
            elif args.defense == 'mad':
                model_adv = create_model(args.model, pretrained=False, num_classes=args.num_classes)
                target_model = MAD(
                    model=target_model,
                    out_path=args.defense_log_path,
                    epsilon=args.mad_epsilon,
                    model_adv_proxy=model_adv,
                    oracle=args.mad_oracle)
            else:
                raise NotImplementedError

             
        if not args.augmentation:
            print('Train without augmentation')
            label_only = args.distillation_type == 'hard'
            dataset_train = ThiefDataset(dataset_train, args, target_model=target_model, label_only=label_only)
            dataset_val = ThiefDataset(dataset_val, args, target_model=target_model, label_only=label_only)

            if not label_only:
                criterion = KLDivLossWithTemperature(args.distillation_tau, args.softmax_target)
            args.distillation_type = 'none'
    else:
        if not args.num_classes:
            args.num_classes = utils.infer_num_classes_from_dataset(dataset_train)

   #### End modifications

    if args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()
        if args.repeated_aug:
            sampler_train = RASampler(
                dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
            )
        else:
            sampler_train = torch.utils.data.DistributedSampler(
                dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
            )
        if args.dist_eval:
            if len(dataset_val) % num_tasks != 0:
                print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                      'This will slightly alter validation results as extra duplicate entries are added to achieve '
                      'equal num of samples per-process.')
            sampler_val = torch.utils.data.DistributedSampler(
                dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=False)
        else:
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    else:

#### Begin modifications

        if args.weights_path:
            print(f'Loading weights from {args.weights_path}')
            weights = torch.load(args.weights_path)
            sampler_train = torch.utils.data.WeightedRandomSampler(
                weights,
                args.num_samples,
                generator=torch.Generator().manual_seed(args.sample_seed))
        else:
            sampler_train = torch.utils.data.RandomSampler(
                dataset_train,
                generator=torch.Generator().manual_seed(args.sample_seed))

        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

#### End modifications

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=int(1.5 * args.batch_size),
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )

    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        mixup_fn = Mixup(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=args.num_classes)

    print(f"Creating model: {args.model}")
    model = create_model(
        args.model,
        pretrained=args.pretrained,
        num_classes=args.num_classes,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path,
        drop_block_rate=None,
    )

    if args.finetune:
        if args.finetune.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.finetune, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.finetune, map_location='cpu')

        checkpoint_model = checkpoint['model']
        state_dict = model.state_dict()
        for k in ['head.weight', 'head.bias', 'head_dist.weight', 'head_dist.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        # interpolate position embedding
        pos_embed_checkpoint = checkpoint_model['pos_embed']
        embedding_size = pos_embed_checkpoint.shape[-1]
        num_patches = model.patch_embed.num_patches
        num_extra_tokens = model.pos_embed.shape[-2] - num_patches
        # height (== width) for the checkpoint position embedding
        orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
        # height (== width) for the new position embedding
        new_size = int(num_patches ** 0.5)
        # class_token and dist_token are kept unchanged
        extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
        # only the position tokens are interpolated
        pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
        pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
        pos_tokens = torch.nn.functional.interpolate(
            pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
        pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
        new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
        checkpoint_model['pos_embed'] = new_pos_embed

        model.load_state_dict(checkpoint_model, strict=False)

    model.to(device)

    model_ema = None
    if args.model_ema:
        # Important to create EMA model after cuda(), DP wrapper, and AMP but before SyncBN and DDP wrapper
        model_ema = ModelEma(
            model,
            decay=args.model_ema_decay,
            device='cpu' if args.model_ema_force_cpu else '',
            resume='')

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    linear_scaled_lr = args.lr * args.batch_size * utils.get_world_size() / 512.0
    args.lr = linear_scaled_lr
    optimizer = create_optimizer(args, model_without_ddp)
    loss_scaler = NativeScaler()

    lr_scheduler, _ = create_scheduler(args, optimizer)
    

    output_dir = Path(args.output_dir)
    
    # wrap the criterion in our custom DistillationLoss, which
    # just dispatches to the original criterion if args.distillation_type is 'none'
    criterion = DistillationLoss(
        criterion, target_model, args.distillation_type, args.distillation_alpha, args.distillation_tau,
        args.softmax_target, output_dir if args.log_target_predictions else None)

    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.start_epoch = checkpoint['epoch'] + 1
            if args.model_ema:
                utils._load_checkpoint_for_ema(model_ema, checkpoint['model_ema'])
            if 'scaler' in checkpoint:
                loss_scaler.load_state_dict(checkpoint['scaler'])

    if args.eval:
        test_stats = evaluate(data_loader_val, model, device)
        print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
        return

    print(f"Start training for {args.epochs} epochs on {args.num_samples} samples")
    start_time = time.time()
    max_accuracy = 0.0
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)

        train_stats = train_one_epoch(
            model, criterion, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            args.clip_grad, model_ema, mixup_fn,
            set_training_mode=args.finetune == ''  # keep in eval mode during finetuning
        )

        lr_scheduler.step(epoch)

#### Begin modifications

        checkpoint_paths = [output_dir / f'checkpoint.pth']
        if args.output_dir and (epoch+1) % args.checkpoint_frequency == 0:
            checkpoint_paths.append(output_dir / f'checkpoint_epoch_{epoch+1}.pth')

#### End modifications

        for checkpoint_path in checkpoint_paths:
            utils.save_on_master({
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': epoch,
                'model_ema': get_state_dict(model_ema),
                'scaler': loss_scaler.state_dict(),
                'args': args,
            }, checkpoint_path)

#### Begin modifications

        if args.with_eval:
            test_stats = evaluate(data_loader_val, model, device)
            print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
            max_accuracy = max(max_accuracy, test_stats["acc1"])
            print(f'Max accuracy: {max_accuracy:.2f}%')
        else:
            test_stats = {'acc1': '-'}

#### End modifications

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'test_{k}': v for k, v in test_stats.items()},
                     'epoch': epoch,
                     'n_parameters': n_parameters}

        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    from parser import args
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
