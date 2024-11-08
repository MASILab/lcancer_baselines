import os
import numpy as np
import torch
from lungbl.utils.tabular import format_datetime_str
from lungbl.datamodule import Item, PandasDataset


class DLSTMFeatDataset(PandasDataset):
    def __init__(
        self,
        df,
        data_cache,
        seq_len: int=2,
        feat_dim: int=128,
        fup_label: str='scan_fup_days',
        **kwargs,
    ):
        super().__init__(df, data_cache, **kwargs)
        self.seq_len = seq_len
        self.feat_dim = feat_dim
        self.df = self.df[self.df[fup_label].notnull()] # filter out rows without fup data
        self.pids = self.df['pid'].unique().tolist()
        self.fup_label = fup_label

    def __getitem__(self, index):
        pid = self.pids[index]
        pid_rows = self.df[self.df['pid'] == pid].sort_values(by='scanorder', ascending=True)
        pid_rows = pid_rows.iloc[-self.seq_len:]

        # get features vectors
        scandates = ["", ""]
        seq = torch.zeros((self.seq_len, 5, self.feat_dim), dtype=torch.float32)
        
        for i, (idx, row) in enumerate(pid_rows.iterrows()):
            scandate = format_datetime_str(row.scandate, format=self.date_format)
            scandates[i] = scandate
            feat = np.load(os.path.join(self.data_cache.noduleft_data, f"{pid}time{scandate}.npy"))[:5].astype('float32')
            seq[i] = torch.tensor(feat, dtype=torch.float32)
            
        # convert follow up duration to relative time distance
        times = torch.zeros(self.seq_len, dtype=torch.float32)
        fup = pid_rows[self.fup_label].tolist()
        fup = [i - fup[-1] for i in fup]
        times[:len(fup)] = torch.tensor(fup, dtype=torch.float32)
        times = times / 365 # transform into fractional months relative to latest scan, with latest scan at time 0
        
        label = int(pid_rows.iloc[0][self.label])

        return Item(
            pid=pid,
            scandate=scandates,
            data=[seq, times],
            label=torch.tensor(label, dtype=torch.int64)
        )

    def __len__(self) -> int:
        return len(self.pids)