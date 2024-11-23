import os, yaml
from dataclasses import dataclass
import lightning as L
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger
from lungbl import definitions
from lungbl.train import BinaryClassifier_Lightning, ValAUC
from lungbl.test import TestPredStore, LongPredStore, Predict_Liao
from lungbl.datamodule import AbstractDataModule
from lungbl.liao.datasets import DataBowl3Classifier
from lungbl.sybil_bl.datasets import SerieDataset
from lungbl.tdvit.datasets import LongitudinalFeatDataset
from lungbl.dlstm.datasets import DLSTMFeatDataset
from lungbl.dls.datasets import DLSDataset
from lungbl.cachedcohorts import NAMES, NLST_CACHE

@dataclass
class Baselines:
    liao: str = "liao"
    sybil: str = "sybil"
    tdvit: str = "tdvit"
    dlstm: str = "dlstm"
    dls: str = "dls"
BASELINES = Baselines()

LMODELS = {
    "BinaryClassifier": BinaryClassifier_Lightning,
}
DATASETS = {
    BASELINES.liao: DataBowl3Classifier,
    BASELINES.sybil: SerieDataset,
    BASELINES.tdvit: LongitudinalFeatDataset,
    BASELINES.dlstm: DLSTMFeatDataset,
    BASELINES.dls: DLSDataset,
}
CACHED_COHORTS = {
    NAMES.nlst: NLST_CACHE,
}

class Config():
    def __init__(self, configf) -> None:
        self.id = os.path.splitext(os.path.basename(configf))[0]
        config = self.load_config(configf)

        # ----------------------------------------------------------------------- #
        #  PASSED IN PARAMS FROM CONFIG FILE
        # ----------------------------------------------------------------------- #
        self.dataset = config['data']['dataset']
        self.datacache = config['data']['datacache']
        self.batch_size = config['data']['batch_size']
        self.val_split = config['data']['val_split']
        self.date_format = config['data']['date_format']
        # self.n_splits = config['data']['n_splits']

        self.log_every_n_steps = config['logging']['log_every_n_steps']
        self.val_every_n_epoch = config['logging']['val_every_n_epoch']

        # self.checkpoint = config['model']['checkpoint']
        self.model_name = config['model']['model_name']
        self.lmodel = config['model']['lmodel']
        self.noduleft_dim = config['model']['noduleft_dim']

        self.lr = config['optimization']['lr']
        self.warmup_steps = config['optimization']['warmup_steps']
        self.epochs = config['optimization']['epochs']
        self.patience = config['optimization']['patience']

        # optional params
        optional_params = [
            ('data', 'label'),
            ('data', 'cancer_year'),
            ('data', 'n_splits'),
            ('data', 'max_followup'),
            ('data', 'multi_class'),
            ('model', 'checkpoint'),
            ('model', 'y_one_hot'),
            ('model', 'output_logit'),
            ('optimization', 'val_metric')
        ]
        for p1, p2 in optional_params:
            if p2 in config[p1].keys():
                setattr(self, p2, config[p1][p2])
            else:
                setattr(self, p2, None)

    @staticmethod
    def load_config(configf):
        with open(os.path.join(definitions.CONFIG_DIR, configf), 'r') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        return config

def datamodule_from_config(cohort, config: Config, k: int=None, label: str='lung_cancer'):
    return AbstractDataModule(
        cohort,
        dataset=DATASETS[config.dataset],
        data_cache=CACHED_COHORTS[config.datacache],
        k=k,
        n_splits=config.n_splits,
        batch_size=config.batch_size,
        val_split=config.val_split,
        max_followup=config.max_followup,
        date_format=config.date_format,
        label=config.label if config.label else label,
    )

def trainer_from_config(config: Config, phase:str, suffix:str="", predict_dst: str=None):
    return L.Trainer(
        default_root_dir=os.path.join(definitions.CHECKPOINT_DIR, config.id),
        accelerator="auto",
        devices=1,
        max_epochs=config.epochs,
        callbacks=callbacks_from_config(config, suffix, predict_dst),
        logger=TensorBoardLogger(save_dir=definitions.LOG_DIR, name=config.id),
        enable_progress_bar=True,
        log_every_n_steps=config.log_every_n_steps,
        check_val_every_n_epoch=config.val_every_n_epoch,
    )

def callbacks_from_config(config: Config, suffix, predict_dst):
    metric = config.val_metric if config.val_metric else "val_loss"
    mode = "min" if metric == "val_loss" else "max"
    callbacks = [
        # ModelCheckpoint(dirpath=os.path.join(definitions.CHECKPOINT_DIR, config.id), 
        #     filename='{epoch}-{val_auc:.2f}', mode=mode, monitor=metric, save_top_k=1),
        ModelCheckpoint(
            dirpath=os.path.join(definitions.CHECKPOINT_DIR, config.id),
            filename='{epoch}-{val_auc:.2f}', 
            mode=mode, 
            monitor=metric, 
            save_top_k=1,
            save_last=True,),
        LearningRateMonitor(logging_interval="step"),
        EarlyStopping(monitor="val_loss", min_delta=0.00, patience=config.patience, mode='min'),
        ValAUC(),
    ]
    callbacks += callbacks_from_baseline(config, suffix, predict_dst)
    return callbacks

def callbacks_from_baseline(
        config: Config, 
        suffix, 
        predict_dst,
    ) -> list:
    """
    Rules for adding baseline-specific callbacks to lightning Trainer
    returns: list - additional callbacks
    """
    callbacks = []
    if config.dataset == BASELINES.liao:
        # save latent liao features
        callbacks.append(Predict_Liao(predict_dst))
    if config.dataset == BASELINES.sybil:
        # if sybil, use longitudinal predictions callback
        callbacks.append(LongPredStore(config.id, cancer_year=config.cancer_year, suffix=suffix))
    if config.dataset != BASELINES.sybil:
        # standard test predictions callback
        output_logit=config.output_logit if config.output_logit else False
        callbacks.append(TestPredStore(config.id, output_logit=config.output_logit, suffix=suffix))
        
    return callbacks

def checkpoint_from_config(config: Config):
    """
    max_epoch: Find the highest epoch in among saved checkpoints
    else use path provided in config
    """
    if config.checkpoint == "max_epoch":
        checkpoints = os.listdir(os.path.join(definitions.CHECKPOINT_DIR, config.id))
        epochs = [int(filename[6:-5]) for filename in checkpoints]  # 'epoch={int}.ckpt' filename format
        max_idx = epochs.index(max(epochs)) # idnex of highest epoch
        return os.path.join(definitions.CHECKPOINT_DIR, config.id, checkpoints[max_idx])
    else:
        return config.checkpoint


