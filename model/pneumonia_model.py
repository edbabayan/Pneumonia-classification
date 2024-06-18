import torch
import torchvision
import torchmetrics
import pytorch_lightning as pl
from torchvision import transforms

from model.input_process import InputProcessor


class PneumoniaModel(pl.LightningModule):

    def __init__(self):
        super(PneumoniaModel, self).__init__()

        self.model = torchvision.models.resnet18()
        self.model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.model.fc = torch.nn.Linear(512, 1, bias=True)

        self.feature_map = torch.nn.Sequential(*list(self.model.children())[:-2])
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)
        self.loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1]))

        self.train_f1 = torchmetrics.F1Score(task='binary')
        self.val_f1 = torchmetrics.F1Score(task='binary')

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485], std=[0.229])
        ])

        self.input_processor = InputProcessor()

    def forward(self, data):
        pred = self.model(data)
        return pred

    def training_step(self, batch, batch_idx):
        x_ray, label = batch
        label = label.float()
        pred = self(x_ray)[:, 0]
        loss = self.loss_fn(pred, label)

        self.log("Train Loss", loss)
        self.log("Train f1 score", self.train_f1(torch.sigmoid(pred), label.int()))
        return loss

    def on_train_epoch_end(self):
        self.log("Train f1 score", self.train_f1.compute())

    def validation_step(self, batch, batch_idx):
        x_ray, label = batch
        label = label.float()
        pred = self(x_ray)[:, 0]
        loss = self.loss_fn(pred, label)

        self.log("Val loss", loss)
        self.log("Val f1 score", self.val_f1(torch.sigmoid(pred), label.int()))
        return loss

    def on_validation_epoch_end(self):
        self.log("Val f1 score", self.val_f1.compute())

    def configure_optimizers(self):
        return [self.optimizer]

    def calc_feature_map(self, data):
        data = data.to(self.device)
        with torch.no_grad():
            feature_map = self.feature_map(data)
            avg_pool = torch.nn.functional.adaptive_avg_pool2d(feature_map, (1, 1))
            avg_output_flattened = torch.flatten(avg_pool)
            pred = self.model.fc(avg_output_flattened)
        return pred, feature_map

    def predict(self, file_path):
        image = self.input_processor.load_image(file_path)
        image = self.transform(image).unsqueeze(0).to(self.device)
        self.model.eval()
        with torch.no_grad():
            pred = self(image)
            pred = torch.sigmoid(pred).item()
        return pred


if __name__ == '__main__':
    from model.config import CFG
    model = PneumoniaModel.load_from_checkpoint(CFG.model_weights, strict=False)
    prediction = model.predict('')