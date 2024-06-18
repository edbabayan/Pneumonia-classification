import torch
import torchvision
from torchvision import transforms
import numpy as np

from config import CFG


class DataLoaderPreparer:
    def __init__(self, mean=0.5, std=0.5, batch_size=16, num_workers=2):
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_loader = None
        self.valid_loader = None
        self.processed_mean = mean
        self.processed_std = std
        self.data_transformer()

    def postprocess(self, data_path):
        train_transforms, valid_transforms = self.data_transformer()
        train_dataset, valid_dataset = self.prepare_datasets(train_transforms, valid_transforms, data_path)
        train_dataloader, valid_dataloader = self.prepare_dataloaders(train_dataset, valid_dataset,
                                                                      self.batch_size, self.num_workers)
        return train_dataloader, valid_dataloader

    def data_transformer(self):
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=self.processed_mean, std=self.processed_std),
            transforms.RandomAffine(degrees=(-5, 5), translate=(0, 0.5), scale=(0.9, 1.1)),
            transforms.RandomResizedCrop((CFG.image_size, CFG.image_size), scale=(0.35, 1))
        ])
        valid_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=self.processed_mean, std=self.processed_std),
        ])
        return train_transform, valid_transform

    @staticmethod
    def load_file(path):
        return np.load(path).astype(np.float32)

    def prepare_datasets(self, train_transform, valid_transform, data_path):
        train_dataset = torchvision.datasets.DatasetFolder(data_path.joinpath('train'),
                                                           loader=self.load_file,
                                                           extensions=('npy',),
                                                           transform=train_transform)
        valid_dataset = torchvision.datasets.DatasetFolder(data_path.joinpath('valid'),
                                                           loader=self.load_file,
                                                           extensions=('npy',),
                                                           transform=valid_transform)
        return train_dataset, valid_dataset

    def prepare_dataloaders(self, train_dataset, valid_dataset, batch_size, num_workers):
        self.train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                                        shuffle=True, num_workers=num_workers)
        self.valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size,
                                                        shuffle=False, num_workers=num_workers)
        return self.train_loader, self.valid_loader


if __name__ == '__main__':
    data_loader = DataLoaderPreparer()
    train_loader, valid_loader = data_loader.postprocess()
    print('DataLoader prepared successfully')