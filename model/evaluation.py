import torch
from tqdm import tqdm
import torchmetrics

from model.config import CFG
from model.dataloader import DataLoaderPreparer
from model.pneumonia_model import PneumoniaModel

model = PneumoniaModel.load_from_checkpoint(CFG.model_weights)
model.eval()
_, val_dataset = DataLoaderPreparer().prepare_datasets()

preds = []
labels = []

with torch.no_grad():
    for data, label in tqdm(val_dataset):
        data = data.to(CFG.device).float().unsqueeze(0)
        pred = torch.sigmoid(model(data)[0].cpu())
        preds.append(pred)
        labels.append(label)

preds = torch.tensor(preds).int()
labels = torch.tensor(labels).int()

accuracy = torchmetrics.Accuracy(task='binary')(preds, labels)
precision = torchmetrics.Precision(task='binary')(preds, labels)
recall = torchmetrics.Recall(task='binary')(preds, labels)
cm = torchmetrics.ConfusionMatrix(num_classes=2, task='binary')(preds, labels)

print('Accuracy:', accuracy.item())
print('Precision:', precision.item())
print('Recall:', recall.item())
print('Confusion Matrix:', cm)
