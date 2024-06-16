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

