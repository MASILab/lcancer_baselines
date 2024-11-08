import os, torch, numpy as np
from typing import TypedDict
from pathlib import Path
import pandas as pd
import lightning as L
from sklearn import metrics
from lungbl.utils.modeling import binary_prob_output

import lungbl.definitions as definitions

class TestPred(TypedDict):
    pid: str
    scandate: str
    label: torch.Tensor
    prob: torch.Tensor

class TestPredStore(L.Callback):
    def __init__(self, 
        configid,
        k:int=None,
        suffix: str="",
        output_logit: bool=True,
        **kwargs,
        ):
        super().__init__(**kwargs)
        self.test_pred = []
        self.output_logit = output_logit
        fname = f"test_pred_{suffix}.csv"
        self.save_path = os.path.join(configid, fname) if k is None else os.path.join(configid, f"fold_{k}", fname)

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        """
        y_hat should have shape b c (b=bach size, c=classes)
        y should have shape b
        """
        pids, scandates, labels = batch['pid'], batch['scandate'], batch['label']
        probs = torch.nn.functional.sigmoid(outputs) if self.output_logit else outputs # convert to prob if output is a logit
        for i, pid in enumerate(pids):
            scandate = scandates[i]
            label = labels[i].detach().cpu().numpy()
            prob = binary_prob_output(probs).detach().cpu().numpy()
            self.test_pred.append(
                TestPred(
                    pid=pid,
                    scandate=scandate,
                    label=label.item(),
                    prob=prob.item(),
            ))


    def on_test_end(self, trainer, pl_module):
        # Save predictions
        df = pd.DataFrame(self.test_pred)
        dst = os.path.join(definitions.DATA_DIR, self.save_path)
        Path.mkdir(Path(dst).parent, parents=True, exist_ok=True)
        df.to_csv(dst)
        
        auc = metrics.roc_auc_score(df['label'], df['prob'])
        print(f"Test AUC: {auc}")


def sybil_single_yr_prob(pred, cancer_year):
    return pred.scores[0][cancer_year]

class LongPredStore(L.Callback):
    def __init__(self, 
        configid, 
        cancer_year: int=1, # index of year to predict cancer [0...6]
        k:int=None,
        suffix: str="",
        **kwargs,
        ):
        """Stores computed logit and labels for a single year"""
        super().__init__()
        self.cancer_year = cancer_year
        self.test_pred = []
        fname = f"test_pred_{suffix}.csv"
        self.save_path = os.path.join(configid, fname) if k is None else os.path.join(configid, f"fold_{k}", fname)
        
    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        pid, scandate, label = batch['pid'], batch['scandate'], batch['label']
        label = label.squeeze().detach().cpu().numpy()
        prob = sybil_single_yr_prob(outputs, self.cancer_year)
        # pred = torch.sigmoid(outputs.squeeze())[1]
        self.test_pred.append(TestPred(pid=pid[0], scandate=scandate[0], label=label.item(), prob=prob))

    def on_test_end(self, trainer, pl_module):
        # Save predictions
        df = pd.DataFrame(self.test_pred)
        dst = os.path.join(definitions.DATA_DIR, self.save_path)
        Path.mkdir(Path(dst).parent, parents=True, exist_ok=True)
        df.to_csv(dst)
        
        auc = metrics.roc_auc_score(df['label'], df['prob'])
        print(f"Test AUC: {auc}")
        
        
class Predict_Liao(L.Callback):
    def __init__(self,
        out_dir,
        **kwargs,
        ):
        super().__init__(**kwargs)
        if out_dir is not None:
            self.dir64 = os.path.join(out_dir, "feat64")
            self.dir128 = os.path.join(out_dir, "feat128")
            Path.mkdir(Path(os.path.join(self.dir64)), parents=True, exist_ok=True)
            Path.mkdir(Path(os.path.join(self.dir128)), parents=True, exist_ok=True)
        
    def on_predict_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        pids, scandates = batch['pid'], batch['scandate']
        *_, latent = outputs
        feat128, feat64 = latent
        feat128 = feat128.detach().cpu().numpy()
        feat64 = feat64.detach().cpu().numpy()
        for i, pid in enumerate(pids):
            scandate = scandates[i]
            feat64path = os.path.join(self.dir64, f"{pid}time{scandate}.npy")
            feat128path = os.path.join(self.dir128, f"{pid}time{scandate}.npy")
            if not os.path.exists(feat64path):
                np.save(feat64path, feat64[i])
            if not os.path.exists(feat128path):
                np.save(feat128path, feat128[i])


