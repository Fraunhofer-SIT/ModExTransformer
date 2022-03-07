import argparse
import datetime
import os
import time

import torch
from timm.models import create_model

from datasets import build_transform
import models
from utils import get_free_gpu
from datasets import get_dataset

args = \
{'model': 'resnet34',
 'model_path': 'checkpoints/checkpoint.pth',
 'nb_classes': 10,
 'data_set': 'imagenet',
 'output_filename': 'imagenet_weights.pt',
 'input_size': 224,
 'batch_size': 250,
 'num_gpus': 1,
 'device': 'cuda',
 'num_workers': 10,
 'pin_mem': True,
 'log_frequency': 100}

args = argparse.Namespace(**args)

num_gpus = args.num_gpus
gpu_chosen = get_free_gpu(num_gpus)
device = torch.device(args.device)

dataset = get_dataset(args.dataset, train=True, transform=build_transform(False, **vars(args)))

data_loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=args.batch_size,
    num_workers=args.num_workers,
    pin_memory=args.pin_mem,
    drop_last=False
)

model = create_model(
    args.model,
    num_classes=args.nb_classes
)
model.load_state_dict(torch.load(args.model_path)['model'])

model.to(device)
model.eval()

print(f"Compute weights for {args.model_path} on {args.data_set}")
preds = []
step = 1
start_time = time.time()
with torch.no_grad():
    for x, y in data_loader:
        if step % args.log_frequency == 0:
            print(f"Step: [{step}/{len(data_loader)}]")
        x = x.to(device, non_blocking=True)
        logits = model(x)
        preds.append(logits.argmax(-1).to('cpu'))
        step += 1

preds = torch.cat(preds)
torch.save(preds, os.path.join(os.path.dirname(args.model_path), 'predictions.pt'))

weights = len(preds) / torch.unique(preds, return_counts=True)[1][preds]
torch.save(weights, os.path.join(os.path.dirname(args.model_path), args.output_filename))

total_time = time.time() - start_time
total_time_str = str(datetime.timedelta(seconds=int(total_time)))
print(f'Computing time {total_time_str}')
