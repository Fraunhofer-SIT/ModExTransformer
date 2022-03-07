import torch
from torch.utils.data import Dataset

try:
    from tqdm import tqdm
except ImportError:
    # If tqdm is not available, provide a mock version of it
    def tqdm(x):
        return x


class ThiefDataset(Dataset):
    def __init__(self, dataset, args, target_model=None, target_predictions=None, label_only=False):
        self.dataset = dataset
        self.target_model = target_model

        if target_predictions is not None:
            self.target_predictions = target_predictions
        else:
            if not target_model:
                raise TypeError('Please provide either target predictions or a target model.')
            self.target_predictions = self.get_target_predictions(args.device, args.batch_size, args.num_workers)

        if label_only:
            self.target_predictions = self.target_predictions.argmax(1)

    def get_target_predictions(self, device='cuda', batch_size=100, num_workers=4):
        print(f'Compute target predictions...')
        data_loader = torch.utils.data.DataLoader(self.dataset, batch_size=batch_size, num_workers=num_workers)
        preds_target = []
        with torch.no_grad():
            for x, y in tqdm(data_loader):
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                preds_target.append(self.target_model(x))
        return torch.cat(preds_target).to('cpu')

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        x, y = self.dataset[idx]
        y = self.target_predictions[idx]
        return x, y