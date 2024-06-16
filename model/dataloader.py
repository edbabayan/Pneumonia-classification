import torch
import torchvision
from torchvision import transforms
import torchmetrics
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt


from config import CFG

def load_file(path):
    return np.load(path).astype(np.float32)


train_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=processed_mean, std=processed_std),
    transforms.RandomAffine(degrees=(-5, 5), translate=(0, 0.5), scale=(0.9, 1.1)),
    transforms.RandomResizedCrop((CFG.image_size, CFG.image_size), scale=(0.35, 1))
])

valid_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=processed_mean, std=processed_std),
])

train_dataset = torchvision.datasets.DatasetFolder(CFG.processed_data.joinpath('train'), loader=load_file,
                                                   extensions='npy', transform=train_transform)
valid_dataset = torchvision.datasets.DatasetFolder(CFG.processed_data.joinpath('valid'), loader=load_file,
                                                   extensions='npy', transform=valid_transform)

batch_size = 32
num_workers = 4

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                           shuffle=True, num_workers=num_workers)
valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size,
                                           shuffle=False, num_workers=num_workers)

