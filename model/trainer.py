import os
from model.pneumonia_model import PneumoniaModel
from model.dataloader import DataLoaderPreparer
from model.config import CFG
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from preprocess.preprocessing import Preprocessor
from preprocess.preprocess_config import CFG as preprocess_config


class PneumoniaModelTrainer:
    def __init__(self):
        self.checkpoints_path = CFG.checkpoints
        self.logs_path = CFG.logs
        self.max_epochs = 50
        self.device = CFG.device
        self.preprocess_config = preprocess_config

    def setup_directories(self):
        os.makedirs(self.checkpoints_path, exist_ok=True)
        os.makedirs(self.logs_path, exist_ok=True)

    def get_trainer(self):
        checkpoint_callback = ModelCheckpoint(
            monitor='Val f1 score',
            save_top_k=5,
            mode='max',
            dirpath=self.checkpoints_path
        )

        return pl.Trainer(
            accelerator='gpu',
            devices=1 if self.device == 'cuda' else 0,
            logger=TensorBoardLogger(save_dir=self.logs_path),
            log_every_n_steps=1,
            callbacks=[checkpoint_callback],
            max_epochs=self.max_epochs
        )

    def preprocess_data(self):
        preprocessor = Preprocessor(self.preprocess_config)
        preprocessor.process()
        mean, std = preprocessor.get_statistics()
        print("Mean:", mean, "Std:", std)
        return mean, std

    def train(self):
        self.setup_directories()
        trainer = self.get_trainer()
        mean, std = self.preprocess_data()
        train_loader, val_loader = self.prepare_data_loaders(mean=mean, std=std)

        model = PneumoniaModel()
        trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    @staticmethod
    def prepare_data_loaders(mean=0.5, std=0.5):
        return DataLoaderPreparer(mean=mean, std=std).postprocess()


if __name__ == '__main__':
    _trainer = PneumoniaModelTrainer()
    _trainer.train()
