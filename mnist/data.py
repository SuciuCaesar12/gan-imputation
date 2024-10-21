from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torch
import numpy as np

from config import settings


MISSING_DATA_PERCENT = settings.data.missing_data_percent
BATCH_SIZE = settings.training.batch_size
NUM_CLASSES = settings.model.num_classes


def one_hot(x):
    return F.one_hot(x, num_classes=NUM_CLASSES).type(torch.float32)


class MnistDataset(Dataset):
    
    def __init__(self, train: bool = True):
        super().__init__()
        print("=" * 40)
        print(f"{'Training' if train else 'Validation'} MNIST Dataset Creation".center(40))
        print("=" * 40)
        
        print("Loading the dataset...".ljust(35), end="")
        transform = transforms.Compose([transforms.ToTensor()])
        mnist = datasets.MNIST(root='./mnist_data', train=train, download=True, transform=transform)
        rand_idx = torch.randperm(mnist.data.shape[0])
        self.data = mnist.data.view(mnist.data.shape[0], -1)[rand_idx].float() / 255.0
        self.labels = mnist.targets[rand_idx]
        print("Done.")

        print("Creating missing data masks...".ljust(35), end="")
        self.data_masks = self._generate_missing_data_masks()
        print("Done.")
        
        print("=" * 40)
        print()

    def _generate_missing_data_masks(self):
        data_masks = torch.ones(self.data.shape)
        num_missing = int(MISSING_DATA_PERCENT * self.data.shape[1])

        if num_missing > 0:
            for i in range(self.data.shape[0]):
                indices = torch.randperm(self.data.shape[1])[:num_missing]
                data_masks[i][indices] = 0

        return data_masks
    
    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, index):
        return {
            "x": self.data[index], 
            "m": self.data_masks[index],
            'y': self.labels[index]
        }
    
    def sample(self):
        samples = {'x': [], 'm': [], 'y': []}
        for label_id in self.labels.unique():
            indices = torch.nonzero(self.labels == label_id).squeeze(1)
            i = np.random.randint(indices.numel())
            i = indices[i]

            samples['x'].append(self.data[i])
            samples['m'].append(self.data_masks[i])
            samples['y'].append(one_hot(label_id))

        return {k: torch.stack(v) for k, v in samples.items()}

    def __getitem__(self, index):
        return {
            "x": self.data[index], 
            "m": self.data_masks[index], 
            "y": one_hot(self.labels[index])
        }


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
    
    def __init__(self, train: bool = True):
        self.dataset = MnistDataset(train=train)
        
        self.gen_loader = DataLoaderWrapper(DataLoader(self.dataset, batch_size=BATCH_SIZE, shuffle=True))
        self.disc_loader = DataLoaderWrapper(DataLoader(self.dataset, batch_size=BATCH_SIZE, shuffle=True))
    
    def get_gen_batch(self):
        return self.gen_loader.get_batch()
    
    def get_disc_batch(self):
        return self.disc_loader.get_batch()

    def summary(self):
        print("=" * 40)
        print("Summary of MNIST Dataset".center(40))
        print("=" * 40)
        
        # Display missing rate with percentage formatting
        print(f"{'Missing rate:':<25} {MISSING_DATA_PERCENT:.2%}")
        
        # Print the number of classes
        print(f"{'# of classes:':<25} {NUM_CLASSES}")
        
        # Print the number of samples with comma formatting
        print(f"{'# of samples:':<25} {len(self.dataset):,}")
        
        print("=" * 40)
        print()


if __name__ == "__main__":
    loader = MissingDataImputationDataLoader()
    loader.summary()
    
    batch = loader.get_gen_batch()
    for k, v in batch.items():
        print(k, v.shape)
    