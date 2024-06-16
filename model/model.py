from typing import Any

import torch
import torchvision
import torchmetrics
import pytorch_lightning as pl
from pytorch_lightning.utilities.types import STEP_OUTPUT


class PneumoniaModel(pl.LightningModule):
    def __init__(self):
        super(PneumoniaModel, self).__init__()
        self.model = torchvision.models.resnet18(pretrained=True)
        self.model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.model.fc = torch.nn.Linear(512, 1, bias=True)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([3]))

        self.train_acc = torchmetrics.Accuracy(task='binary')
        self.val_acc = torchmetrics.Accuracy(task='binary')

    def forward(self, data):
        pred = self.model(data)
        return pred

    def training_step(self, batch, batch_idx):
        x_ray, label = batch
        label = label.float()
        pred = self(x_ray)[:, 0]
        loss = self.loss_fn(pred, label)

        self.log("Train loss", loss, on_step=True, on_epoch=True)
        self.log("Train accuracy", self.train_acc(torch.sigmoid(pred), label.int()),
                 on_step=True, on_epoch=True)
        return loss

    def training_epoch_end(self, outputs):
        self.log("Train accuracy", self.train_acc.compute())

    def validation_step(self, batch, batch_idx):
        x_ray, label = batch
        label = label.float()
        pred = self(x_ray)[:, 0]
        loss = self.loss_fn(pred, label)

        self.log("Val loss", loss, on_step=True, on_epoch=True)
        self.log("Val accuracy", self.val_acc(torch.sigmoid(pred), label.int()),
                 on_step=True, on_epoch=True)
        return loss

    def validation_epoch_end(self, outputs):
        self.log("Val accuracy", self.val_acc.compute())

    def configure_optimizers(self):
        return [self.optimizer]


if __name__ == '__main__':
    model = PneumoniaModel()
    print('Model created successfully')