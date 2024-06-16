import torch
import torchvision
from torchvision import transforms
import numpy as np

from config import CFG


class DataLoaderPreparer:
    def __init__(self, mean=0.5, std=0.5):
        self.train_transform = None
        self.valid_transform = None
        self.train_dataset = None
        self.valid_dataset = None
        self.train_loader = None
        self.valid_loader = None
        self.processed_mean = mean
        self.processed_std = std
        self.data_transformer()

    def postprocess(self):
        self.data_transformer()
        self.prepare_datasets()
        train_dataloader, valid_dataloader = self.prepare_dataloaders()
        return train_dataloader, valid_dataloader

    def data_transformer(self):
        self.train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=self.processed_mean, std=self.processed_std),
            transforms.RandomAffine(degrees=(-5, 5), translate=(0, 0.5), scale=(0.9, 1.1)),
            transforms.RandomResizedCrop((CFG.image_size, CFG.image_size), scale=(0.35, 1))
        ])
        self.valid_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=self.processed_mean, std=self.processed_std),
        ])

    @staticmethod
    def load_file(path):
        return np.load(path).astype(np.float32)

    def prepare_datasets(self):
        self.train_dataset = torchvision.datasets.DatasetFolder(CFG.processed_data.joinpath('train'),
                                                                loader=self.load_file,
                                                                extensions=('npy',), transform=self.train_transform)
        self.valid_dataset = torchvision.datasets.DatasetFolder(CFG.processed_data.joinpath('valid'),
                                                                loader=self.load_file,
                                                                extensions=('npy',), transform=self.valid_transform)

    def prepare_dataloaders(self, batch_size=32, num_workers=4):
        self.train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=batch_size,
                                                        shuffle=True, num_workers=num_workers)
        self.valid_loader = torch.utils.data.DataLoader(self.valid_dataset, batch_size=batch_size,
                                                        shuffle=False, num_workers=num_workers)
        return self.train_loader, self.valid_loader


if __name__ == '__main__':
    data_loader = DataLoaderPreparer()
    train_loader, valid_loader = data_loader.postprocess()
    print('DataLoader prepared successfully')