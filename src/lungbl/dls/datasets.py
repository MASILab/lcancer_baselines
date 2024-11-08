import os, numpy as np, torch

from lungbl.datamodule import PandasDataset, Item

class DLSDataset(PandasDataset):
    def __init__(self, df, data_cache, **kwargs):
        super().__init__(df, data_cache, **kwargs)
        self.biomarkers = ["age", "education", "bmi", "phist", "fhist", "smo_status", "quit_time", "pkyr"]
        self.df = self.df[self.df[self.biomarkers].notnull().all(axis=1)] # remove subjects with null biomarkers
        
    def __getitem__(self, index) -> Item:
        item = self.df.iloc[index]
        pid, scandate = item.pid, item.scandate
        feat = np.load(os.path.join(self.data_cache.noduleft_data, f"{pid}time{scandate}.npy"))
        biomarker = np.zeros(10).astype(np.float32)
        # first two features are with_image, with_marker
        biomarker[:2] = 1.
        biomarker[2:] = item[self.biomarkers].values.astype(np.float32)
        
        label = int(item[self.label])
        
        return Item(
            pid=pid,
            scandate=scandate,
            data=[feat, biomarker, feat, biomarker],
            label=torch.tensor(label, dtype=torch.int64)
        )
