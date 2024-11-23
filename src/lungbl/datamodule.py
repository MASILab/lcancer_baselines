
import pandas as pd
import numpy as np
from typing import TypedDict
from typing import Optional
import torch
from torch.utils.data import Dataset, DataLoader
import lightning as L
from sklearn.model_selection import StratifiedKFold
from lungbl.cachedcohorts import CachedCohort

from lungbl import definitions

class Item(TypedDict):
    pid: str
    scandate: str
    data: list
    label: torch.Tensor

class PandasDataset(Dataset):
    def __init__(self,
        df: pd.DataFrame,
        data_cache: CachedCohort,
        date_format: str="%Y%m%d",
        max_followup: int=None,
        label: str="lung_cancer",
    ):
        self.df = df
        self.data_cache = data_cache
        self.date_format = date_format
        self.max_followup = max_followup
        self.label = label

    def __getitem__(self, index) -> Item:
        return Item()
    
    def __len__(self) -> int:
        return len(self.df)
    

class AbstractDataModule(L.LightningDataModule):
    def __init__(self, 
        dataframe: object, 
        dataset: PandasDataset, 
        data_cache: CachedCohort,
        k: int=None,
        n_splits: int=5,
        batch_size: int=15,
        val_split: float=0.2, 
        max_followup: int=None,
        date_format: str="%Y%m%d",
        label: str="lung_cancer",
        **kwargs):

        super().__init__(**kwargs)
        self.dataframe = dataframe.reset_index(drop=True)
        self.dataset = dataset
        self.data_cache = data_cache
        self.k = k
        self.n_splits = n_splits
        self.batch_size = batch_size
        self.val_split = val_split
        self.max_followup = max_followup
        self.date_format = date_format
        self.label = label

    def prepare_data(self):
        # called before setup
        if self.k is not None:
            kf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=definitions.RANDOM_SEED)
            all_splits = [k for k in kf.split(self.dataframe.index, self.dataframe[self.label])]
            trainfold_idx, valfold_idx = all_splits[self.k]
            self.trainfold, self.valfold = self.dataframe.loc[trainfold_idx], self.dataframe.loc[valfold_idx]

    def setup(self, stage: str):
        if self.max_followup is None:
            print(f"Class balance of cohort: {self.dataframe[self.label].value_counts()}")

        if stage in ["fit", "validate"]:
            df = self.dataframe if self.k is None else self.trainfold
            val_items = df.groupby(self.label, group_keys=False).apply(lambda x: x.sample(frac=self.val_split))
            train_items = df.drop(val_items.index)
            self.mm_train = self.dataset(df=train_items, data_cache=self.data_cache, date_format=self.date_format, max_followup=self.max_followup, label=self.label)
            self.mm_val = self.dataset(df=val_items, data_cache=self.data_cache, date_format=self.date_format, max_followup=self.max_followup, label=self.label)

        if stage in ["test", "predict"]:
            df = self.dataframe if self.k is None else self.valfold
            self.mm_test = self.dataset(df=df, data_cache=self.data_cache, date_format=self.date_format, max_followup=self.max_followup, label=self.label)

    def train_dataloader(self):
        return DataLoader(self.mm_train, batch_size=self.batch_size, num_workers=6, persistent_workers=True, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.mm_val, batch_size=self.batch_size, num_workers=6, persistent_workers=True, shuffle=True)

    def test_dataloader(self):
        return DataLoader(self.mm_test, batch_size=1, num_workers=3)

    def predict_dataloader(self):
        return DataLoader(self.mm_test, batch_size=1, num_workers=6, persistent_workers=True)
