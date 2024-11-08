import os
import numpy as np
from pathlib import Path
from typing import TypedDict

import pandas as pd
import torch, torch.nn as nn, torch.optim as optim
import lightning as L
from sklearn import metrics

from lungbl.utils.dl import WarmupCosineSchedule

import lungbl.definitions as definitions
from lungbl.models import init_model


class BinaryClassifier_Lightning(L.LightningModule):
    def __init__(self,
        config: object,
        total_steps: int=10,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.config = config
        self.model = init_model(config.model_name)
        # self.model.load_state_dict(torch.load(config.checkpoint))
        self.total_steps = total_steps
        # self.loss = nn.BCELoss()
        self.loss = nn.BCEWithLogitsLoss()
        self.y_one_hot = config.y_one_hot
        self.one_hot = lambda x: nn.functional.one_hot(x, num_classes=2).to(torch.float32)
        self.save_hyperparameters()
    
    def training_step(self, batch, batch_idx):
        data, y = batch['data'], batch['label']
        y_hat, *_ = self.model(*data)
        if self.y_one_hot:
            y = self.one_hot(y)
        loss = self.loss(y_hat, y.float())
        acc = (y_hat.argmax(dim=1) == y.argmax(dim=1)).float().mean() if self.y_one_hot else (y_hat.round()==y).float().mean()
        log = {'train_acc': acc, 'train_loss': loss}
        self.log_dict(log, prog_bar=True, sync_dist=True)
        return loss
    
    def forward(self, x):
        return self.model(x)
    
    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.config.lr, betas=(0.9, 0.95))
        scheduler = WarmupCosineSchedule(optimizer, warmup_steps=self.config.warmup_steps, t_total=self.total_steps)
        return [optimizer], [scheduler]
    
    def validation_step(self, batch, batch_idx):
        data, y = batch['data'], batch['label']
        y_hat, *_ = self.model(*data)
        if self.y_one_hot:
            y = self.one_hot(y)
        loss = self.loss(y_hat, y.float())
        acc = (y_hat.argmax(dim=1) == y.argmax(dim=1)).float().mean() if self.y_one_hot else (y_hat.round()==y).float().mean()
        log = {'val_acc': acc, 'val_loss': loss}
        self.log_dict(log, prog_bar=True, sync_dist=True)
        return y_hat, y

    def test_step(self, batch, batch_idx):
        data = batch['data']
        y_hat, *_ = self.model(*data)
        # acc = (y_hat.argmax(dim=1) == y.argmax(dim=1)).float().mean()
        # self.log("test_acc", acc)
        return y_hat

    def predict_step(self, batch, batch_idx):
        data = batch['data']
        y_hat, latent = self.model(*data)
        return y_hat, latent


class ValAUC(L.Callback):
    def __init__(self) -> None:
        super().__init__()
        self.y_hats = []
        self.ys = []
        # self.save_path = os.path.join(configid, "val_pred.csv") if k is None else os.path.join(configid, f"fold_{k}", "val_pred.csv")

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        """assumes batch siz > 1"""
        y_hat, y = outputs
        # y_hat = y_hat.unsqueeze(1)
        self.ys.append(y)
        self.y_hats.append(y_hat)
        
    def on_validation_epoch_end(self, trainer, pl_module):
        y_hats, ys = torch.cat(self.y_hats).cpu().numpy(), torch.cat(self.ys).cpu().numpy()
        if len(np.unique(ys)) == 1:
            print("Only one class in validation set. Skipping AUC calculation.")
        else:
            auc = metrics.roc_auc_score(ys, y_hats)
            print(f"Val AUC: {auc}")
            self.log_dict({'val_auc': auc})

def train(datamodule, Lmodel, trainer, checkpoint=None):
    if checkpoint:
        Lmodel = Lmodel.load_from_checkpoint(checkpoint)
        print(f"Loaded model from checkpoint: {checkpoint}")

    trainer.fit(Lmodel, datamodule=datamodule)
    trained_model = Lmodel.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)
    print(f"Trained model saved at {trainer.checkpoint_callback.best_model_path}")

    val_result = trainer.validate(trained_model, datamodule=datamodule)
    result = val_result[0]["val_acc"]
    print(f"Val acc: {result}")

def test(datamodule, Lmodel, trainer, checkpoint):
    print(f"Testing model at {checkpoint}")
    if checkpoint:
        Lmodel = Lmodel.load_from_checkpoint(checkpoint)
    test_result = trainer.test(Lmodel, datamodule)
    # print(f"Test acc: {test_result[0]['test_acc']}")

def predict(datamodule, Lmodel, trainer, checkpoint):
    print(f"Predicting model at {checkpoint}")
    if checkpoint:
        Lmodel = Lmodel.load_from_checkpoint(checkpoint)
    trainer.predict(Lmodel, datamodule)