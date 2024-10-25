from typing import Optional
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader

from config import settings

cfg = settings.mnist
C, H, W = INPUT_SHAPE = cfg.input_shape
N_FEATS = C * H * W
NUM_CLASSES = cfg.num_classes
SEED = cfg.seed

def mi_random_mask():
    mask = torch.ones(N_FEATS, dtype=torch.float32)
    num_missing = int(0.5 * N_FEATS)
    if num_missing > 0:
        indices = torch.randperm(N_FEATS)[:num_missing]
        mask[indices] = 0
    return mask.view(H, W).type(torch.bool)

def mi_horizontal_mask(rows: list = list(range(14, 18))):
    mask = torch.ones(H, W, dtype=torch.float32)
    mask[rows, :] = 0
    return mask.type(torch.bool)

def mi_vertical_mask(cols: list = list(range(14, 18))):
    mask = torch.ones(H, W, dtype=torch.float32)
    mask[:, cols] = 0
    return mask.type(torch.bool)

MI_METHODS = (
    ('random', mi_random_mask),
    ('horizontal', mi_horizontal_mask),
    ('vertical', mi_vertical_mask)
)

def one_hot(x):
    return F.one_hot(x, num_classes=NUM_CLASSES).type(torch.float32)

def load_stratified_split():
    assert cfg.train_size + cfg.test_size + cfg.mi_size == 1
    
    transform = transforms.Compose([transforms.ToTensor()])
    root = r'./mnist_v2/data/original'
    train_mnist = datasets.MNIST(root=root, train=True, download=True, transform=transform)
    val_mnist = datasets.MNIST(root=root, train=False, download=True, transform=transform)

    data = torch.cat([train_mnist.data, val_mnist.data], dim=0).numpy()
    labels = torch.cat([train_mnist.targets, val_mnist.targets], dim=0).numpy()

    train_idx, temp_idx, _, temp_labels = train_test_split(
        torch.arange(len(data)),
        labels,
        train_size=cfg.train_size,
        stratify=labels,
        random_state=cfg.seed
    )

    test_idx, mi_idx = train_test_split(
        temp_idx,
        temp_labels,
        train_size=cfg.test_size / (1 - cfg.train_size),
        stratify=temp_labels,
        random_state=cfg.seed
    )[:2]
    
    train_data, train_labels = data[train_idx], labels[train_idx]
    test_data, test_labels = data[test_idx], labels[test_idx]
    mi_data, mi_labels = data[mi_idx], labels[mi_idx]
    
    return (
        (train_data, train_labels),
        (test_data, test_labels),
        (mi_data, mi_labels)
    )


class MnistDataset(Dataset):
    
    def __init__(self, data, labels, name: str, masks: Optional[torch.Tensor] = None, flatten: bool = False):
        super().__init__()
        self.name = name
        self.data = torch.Tensor(data).type(torch.float32) / 255.0
        self.data = self.data.unsqueeze(1)
        self.labels = torch.Tensor(labels).type(torch.long)
        self.masks = torch.Tensor(masks).type(torch.float32) if masks is not None else None
        self.flatten = flatten
        
        if self.flatten:
            self.data = self.data.view(self.data.shape[0], -1)
            if self.masks is not None:
                self.masks = self.masks.view(self.masks.shape[0], -1)
        
    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, index):
        sample = {'x': self.data[index], 'y': one_hot(self.labels[index])}
        if self.masks is not None:
            sample['m'] = self.masks[index]
        return sample
    
    def sample(self):
        assert self.masks is not None
        samples = {'x': [], 'm': [], 'y': []}
        for label_id in self.labels.unique():
            indices = torch.nonzero(self.labels == label_id).squeeze(1)
            i = np.random.randint(indices.numel())
            i = indices[i]

            samples['x'].append(self.data[i])
            samples['m'].append(self.masks[i])
            samples['y'].append(one_hot(label_id))

        return {k: torch.stack(v) for k, v in samples.items()}


class DataLoaderWrapper:
    def __init__(self, loader: DataLoader):
        self.loader = loader
        self.loader_iter = iter(self.loader)

    def get_batch(self):
        try:
            batch = next(self.loader_iter)
        except StopIteration:
            self.loader_iter = iter(self.loader)
            batch = next(self.loader_iter)
        return batch


class MissingDataImputationDataLoader:
    
    def __init__(self, dataset):
        self.dataset = dataset
        
        self.gen_loader = DataLoaderWrapper(DataLoader(self.dataset, batch_size=settings.hexa.train_batch_size, shuffle=True))
        self.disc_loader = DataLoaderWrapper(DataLoader(self.dataset, batch_size=settings.hexa.train_batch_size, shuffle=True))
    
    def get_gen_batch(self):
        return self.gen_loader.get_batch()
    
    def get_disc_batch(self):
        return self.disc_loader.get_batch()
    
    def __len__(self):
        return len(self.gen_loader.loader)
