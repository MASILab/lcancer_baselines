
import torch
from sybil import Serie
from lungbl.utils.tabular import format_datetime_str 

from lungbl.datamodule import Item, PandasDataset


class SerieDataset(PandasDataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def __getitem__(self, index) -> Item:
        row = self.df.iloc[index]
        pid, scandate, fpath = str(row.pid), format_datetime_str(row.scandate), row.fpath
        label = int(row[self.label])
        return Item(
            pid=pid, 
            scandate=scandate, 
            data=[fpath],
            label=torch.tensor(label, dtype=torch.float32)
        )
    

