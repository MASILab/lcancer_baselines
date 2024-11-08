import os, pandas as pd
import itertools
import matplotlib.pyplot as plt
from sklearn.isotonic import IsotonicRegression
from sklearn.calibration import CalibrationDisplay
from sklearn.model_selection import StratifiedKFold

from cohorts.cli import NAMES
from lungbl.utils.tabular import read_tabular, format_datetime_str
from lungbl.analysis.stats import BL, RESULTS
import lungbl.definitions as D


class Calibration():
    def __init__(self):
        
        self.regressor = lambda: IsotonicRegression(y_min=0, y_max=1, increasing='auto', out_of_bounds='clip')
        self.n_folds = 10
        self.stratified_cv = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=D.RANDOM_SEED)

        # self.read_results = lambda x: read_tabular(os.path.join(D.DATA_DIR, x), fargs={'dtype':{'pid': str}})
        
    def CVcalibrate(self, results):
        # use cross validation to train k calibrators 
        df = read_tabular(os.path.join(D.DATA_DIR, results), fargs={'dtype': {'pid': str}}) if isinstance(results, str) else results
        df = df.reset_index(drop=True)
        pids, prob, label = df['pid'], df['prob'], df['label']
        split_idxs = self.stratified_cv.split(df.index, df['label'])
        regressor = self.regressor()
        
        # train and test calibrator on k folds
        folds = []
        for kfold in split_idxs:
            train_idxs, test_idxs = kfold
            test_pids = pids.loc[test_idxs]
            train_X, train_y, test_X, test_y = prob.loc[train_idxs], label.loc[train_idxs], prob.loc[test_idxs], label.loc[test_idxs]
            regressor.fit(train_X, train_y)
            test_yhat = regressor.predict(test_X)
            fold_df = pd.concat(
                [test_pids, test_X, pd.Series(test_yhat, index=test_X.index), test_y],
                keys=['pid', 'prob', 'prob_calibrated', 'label'],
                axis=1, 
            )
            folds.append(fold_df)
            
        folds = pd.concat(folds, axis=0)
        return folds

class CalibrationPlot():
    def __init__(self):
        self.bl_idxs = {
            BL.mayo: 0,
            BL.brock: 1,
            BL.liao: 2,
            BL.sybil: 3,
            BL.tdvit: 4,
            BL.dlstm: 5,
            BL.dls: 6,
        }
        self.plot_idxs ={
            NAMES.nlst: 0,
            NAMES.vlsp: 1,
            NAMES.livu: 2,
            NAMES.mcl: 3,
            NAMES.bronch: 4,
        }
        self.plot_2col_idxs = {
            NAMES.nlst: 0,
            NAMES.vlsp: 1,
            NAMES.livu: 0,
            NAMES.bronch: 1,
            "VUMC": 0,
            "UPMC": 1,
            "DECAMP": 0,
            "UCD": 1,
        }
        self.mcl_institutes = {
            "VUMC": 1,
            "UPMC": 2, 
            "DECAMP": 3,
            "UCD": 4,
        }
        self.mcl_cohort = MCL_CohortWrapper().cs
        self.mcl_cohort['scandate'] = self.mcl_cohort['scandate'].apply(lambda x: format_datetime_str(x))
        
    def plot_single(self, df, dst=None, ax=None, title=None):
        prob, prob_calibrated, label = df['prob'], df['prob_calibrated'], df['label']
        n_bins = 5
        if ax is None:
            _, ax = plt.subplots(1)
        CalibrationDisplay.from_predictions(label, prob, ax=ax, n_bins=n_bins, name='uncalibrated')
        CalibrationDisplay.from_predictions(label, prob_calibrated, ax=ax, n_bins=n_bins, name='calibrated')
        ax.get_legend().remove()
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.tick_params(axis='x', labelsize=14)
        ax.tick_params(axis='y', labelsize=14)
        plt.tight_layout()
        if dst is not None:
            plt.savefig(dst)

    def plot_all(self, dst):
        # 7x6 grid of calibration plots condensed into one figure
        fig, axs = plt.subplots(7, 5, figsize=(30, 20))
        for cohort, results in RESULTS.items():
            for bl, preds in results.items():
                cal = Calibration()
                cohort_idx, model_idx = self.plot_idxs[cohort], self.bl_idxs[bl]
                df = cal.CVcalibrate(os.path.join(D.DATA_DIR, preds))
                
                self.plot_single(df, ax=axs[model_idx][cohort_idx])
        plt.savefig(dst)
        
    
    def plot_all_2col(self, dst_dir: str):
        plot1_results = {k: v for k, v in RESULTS.items() if k in (NAMES.nlst, NAMES.vlsp)}
        plot2_results = {k: v for k, v in RESULTS.items() if k in (NAMES.livu, NAMES.bronch)}
        mcl_results = {k: v for k, v in RESULTS.items() if k in (NAMES.mcl)}
        # self.plot_2col(plot1_results, os.path.join(dst_dir, "nlst_vlsp.png"))
        # self.plot_2col(plot2_results, os.path.join(dst_dir, "livu_bronch.png"))
        self.plot_2col_mcl(mcl_results, os.path.join(dst_dir))
        
    
    def plot_2col(self, plot_results: dict, dst: str):
        # figure with 2 columns of calibration plots
        fig, axs = plt.subplots(7, 2, figsize=(10, 21)) # fig size of 5x3
        for cohort, results in plot_results.items():
            for bl, preds in results.items():
                cal = Calibration()
                cohort_idx, model_idx = self.plot_2col_idxs[cohort], self.bl_idxs[bl]
                df = cal.CVcalibrate(os.path.join(D.DATA_DIR, preds))
                
                self.plot_single(df, ax=axs[model_idx][cohort_idx])
        plt.savefig(dst)
        
    def plot_2col_mcl(self, plot_results: dict, dst_dir: str):
        
        def calplot(preds, institute, bl):
            print(f"Plotting {bl} on {institute}")
            df = read_tabular(os.path.join(D.DATA_DIR, preds), fargs={'dtype': {'pid': str, 'scandate': str}})
            df = self.mcl_cohort.merge(df, on=['pid', 'scandate'])
            df = df[df['institute']==self.mcl_institutes[institute]]
            
            if len(df) > 0:
                cal = Calibration()
                df = cal.CVcalibrate(df)
                cohort_idx, model_idx = self.plot_2col_idxs[institute], self.bl_idxs[bl]
                self.plot_single(df, ax=axs[model_idx][cohort_idx])
        
        for cohort, results in plot_results.items():
            fig, axs = plt.subplots(7, 2, figsize=(10, 21))
            for institute in ["VUMC", "UPMC"]:
                
                for bl, preds in results.items():
                    calplot(preds, institute, bl)
                    
            plt.savefig(os.path.join(dst_dir, f"vumc_upmc.png"))
            
            fig, axs = plt.subplots(7, 2, figsize=(10, 21))
            for institute in ["DECAMP", "UCD"]:
                for bl, preds in results.items():
                    calplot(preds, institute, bl)
                    
            plt.savefig(os.path.join(dst_dir, f"decamp_ucd.png"))
                        
                
        
    
if __name__ == '__main__':
    # for cohort, results in RESULTS.items():
    #     for bl, preds in results.items():
    #         cal = Calibration()
    #         df = cal.CVcalibrate(cohort, os.path.join(D.DATA_DIR, preds))
    #         cal.plot(df, f"{bl} on {cohort}", os.path.join(D.FIGURE_DIR, "cal_curves", f"{cohort}_{bl}.png"))
            # break
    # df = cal.CVcalibrate(NAMES.nlst, RESULTS[NAMES.nlst][BL.sybil])
    # cal.plot(df)
    calplot = CalibrationPlot()
    # calplot.plot_all(os.path.join(D.FIGURE_DIR, "cal_curves", "all.png"))
    calplot.plot_all_2col(os.path.join(D.FIGURE_DIR, "cal_curves"))
