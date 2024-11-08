import os, math, random, numpy as np, pandas as pd
from dataclasses import dataclass
from scipy.stats import wilcoxon
from sklearn.metrics import confusion_matrix

from cohorts.cli import NAMES
from lungbl.utils.statutils import bootstrap_samples, bootstrap
from lungbl.utils.tabular import read_tabular, format_datetime_str

from lungbl.utils.modeling import auc_from_df
from lungbl import definitions as D

@dataclass
class Baselines:
    plco: str = "PLCO"
    mayo: str = "Mayo"
    brock: str = "Brock"
    liao: str = "Liao et al."
    # liao: str = "Cross-sectional Imaging"
    sybil: str = "Sybil"
    # sybil: str = "DL on Single Chest CT"
    tdvit: str = "TDViT"
    # tdvit: str = "Longitudinal Imaging"
    dlstm: str = "DLSTM"
    dls: str = "DLS"
    dli: str = "DLI"
    # dls: str = "Multi-path Fusion"
    
BL = Baselines()

RESULTS = {
    # nlst-test
    "nlst-test": {
        # BL.plco: "linear/plco_nlst.test_cs_nodule.csv",
        BL.liao: "liao_nlst_081823/test_pred_ft_test_cs_unconfirmed.csv",
        BL.sybil: "sybil_nlst_081123/test_pred.csv",
        BL.tdvit: "tevit_nlst_082423/test_pred_ft_test_scan_unconfirmed.csv",
        BL.dlstm: "dlstm_nlst_082423/test_pred.csv",
        BL.dls: "dls_imgprep/nlst_test_cs.csv",
    },
    # nlst-nodule
    NAMES.nlst: {
        # BL.plco: "linear/plco_nlst.test_cs_nodule.csv",
        BL.mayo: "linear/mayo_nlst.test_cs_nodule.csv",
        BL.brock: "linear/brock_nlst.test_cs_nodule.csv",
        BL.liao: "liao_nlst_081823/test_pred_ft_test_nodule.csv",
        BL.sybil: "sybil_nlst_081123/test_pred_nodule.csv",
        BL.tdvit: "tevit_nlst_082423/test_pred_ft_test_nodule_unconfirmed.csv",
        BL.dlstm: "dlstm_nlst_082423/test_pred_ft_test_nodule_unconfirmed.csv",
        BL.dls: "dls_imgprep/nlst_test_cs_nodule.csv",
    },
    NAMES.vlsp: {
        BL.liao: "liao_vlsp_090123/test_pred_ft_cs.csv",
        BL.sybil: "sybil_vlsp_083023/test_pred_cs.csv",
        BL.dls: "dls_imgprep/vlsp_test_cs.csv",
    },
    NAMES.livu: {
        BL.liao: "liao_livuspn_090823/test_pred_ft_spn_cs.csv",
        BL.sybil: "sybil_livuspn_090823/test_pred_ft_spn_cs.csv",
        BL.tdvit: "tevit_livuspn_090823/test_pred_ft_spn.csv",
        BL.dlstm: "dlstm_livu_103123/test_pred_ft_spn.csv",
    },
    NAMES.bronch: {
        BL.mayo: "linear/mayo_bronch.cs.csv",
        BL.brock: "linear/brock_bronch.cs.csv",
        BL.liao: "liao_bronch_091223/test_pred_ft_cs.csv",
        BL.sybil: "sybil_bronch_090623/test_pred_cs.csv",
    },
    NAMES.mcl: {
        BL.mayo: "precomputed/mayo_mcl_pred.csv",
        BL.brock: "precomputed/brock_mcl_pred.csv",
        BL.liao: "precomputed/liao_mcl_pred.csv",
        BL.sybil: "sybil_mcl_090123/test_pred_cs.csv",
        BL.tdvit: "precomputed/tdvit_mcl_pred.csv",
        BL.dlstm: "precomputed/dlstm_mcl_pred.csv",
        BL.dls: "precomputed/dls_mcl_pred.csv",
    }    
}

class BestMethodStats():
    """
    Runs a wilcoxon signed rank test in each cohort b/w the best method and all other methods
    """
    def __init__(self):
        self.auc_agg = lambda x: auc_from_df(x, cols=('label', 'prob'))
        self.results = RESULTS
        self.best = {
            NAMES.nlst: BL.sybil,
            NAMES.vlsp: BL.dls,
            NAMES.livu: BL.tdvit,
            "VUMC": BL.dls,
            "UPMC": BL.dls,
            "DECAMP": BL.tdvit,
            "UCD": BL.dls,
            NAMES.bronch: BL.sybil,
        }
        self.mcl_institutes = {
            "VUMC": 1,
            "UPMC": 2, 
            "DECAMP": 3,
            "UCD": 4,
        }
        self.read_results = lambda x: read_tabular(os.path.join(D.DATA_DIR, x), fargs={'dtype':{'pid': str, 'scandate': str}})
        self.mcl_cohort = MCL_CohortWrapper().cs
        self.mcl_cohort['scandate'] = self.mcl_cohort['scandate'].apply(lambda x: format_datetime_str(x))
        self.clinical_cols = ["age", "gender", "bmi", "emphysema", "phist", "fhist", "smo_status", "pkyr"]
    
    @staticmethod
    def seed_random():
        random.seed(D.RANDOM_SEED)
        os.environ['PYTHONHASHSEED'] = str(D.RANDOM_SEED)
        np.random.seed(D.RANDOM_SEED)
    
    def wilcoxon_signed_rank(self, dfa, dfb):
        self.seed_random()
        sample_a = bootstrap_samples(
            dfa,
            agg=self.auc_agg,
            n=1000
        )
        sample_b = bootstrap_samples(
            dfb,
            agg=self.auc_agg,
            n=1000
        )
        statobj = wilcoxon(sample_a['agg'], sample_b['agg'])
        print(statobj.statistic, statobj.pvalue)
        
    def test(self):
        for cohort, results in self.results.items():
            if cohort == NAMES.mcl:
                for institute in self.mcl_institutes:
                    best_method = self.best[institute]
                    pred = self.read_results(results[best_method])
                    dfa = self.mcl_cohort.merge(pred, on=['pid', 'scandate'])
                    dfa = dfa[dfa['institute'] == self.mcl_institutes[institute]]
                    
                    for method, path in results.items():
                        if method != best_method:
                            pred = self.read_results(path)
                            dfb = self.mcl_cohort.merge(pred, on=['pid', 'scandate'])
                            dfb = dfb[dfb['institute'] == self.mcl_institutes[institute]]
                            if len(dfb) > 0:
                                print(f"{best_method} vs {method} on {institute}")
                                self.wilcoxon_signed_rank(dfa, dfb)
            else:
                best_method = self.best[cohort]
                dfa = self.read_results(results[best_method])
                for method, path in results.items():
                    if method != best_method:
                        dfb = self.read_results(path)
                        print(f"{best_method} vs {method} on {cohort}")
                        self.wilcoxon_signed_rank(dfa, dfb)
            
    def print_aucs(self, remove_null=False):
        self.seed_random()
        if remove_null:
            mcl_cohort = self.mcl_cohort.dropna(subset=self.clinical_cols)
        else:
            mcl_cohort = self.mcl_cohort
        for cohort, results in self.results.items():
            if cohort == NAMES.mcl:
                for institute in self.mcl_institutes:
                    for method, path in results.items():
                        pred = self.read_results(path)
                        df = mcl_cohort.merge(pred, on=['pid', 'scandate'])
                        df = df[df['institute'] == self.mcl_institutes[institute]]
                        if len(df) > 0:
                            print(f"{method} on {institute}")
                            bs_auc = bootstrap(df, self.auc_agg, n=1000)
                            print(bs_auc.applymap('{:.3f}'.format))
            else:
                for method, path in results.items():
                    pred = self.read_results(path)
                    print(f"{method} on {cohort}")
                    bs_auc = bootstrap(pred, self.auc_agg, n=1000)
                    print(bs_auc.applymap('{:.3f}'.format))
    
    @staticmethod
    def index_of_union(df, n_steps=100):
        # cut-point c that minimizes |se(c) - AUC| + |sp(c) - AUC|
        auc = auc_from_df(df)
        min_prob, max_prob = df['prob'].min(), df['prob'].max()
        step_size = (max_prob - min_prob)/n_steps
        min_iu = 1
        opt_cpoint, opt_se, opt_sp = np.nan, np.nan, np.nan
        for i in range(1, n_steps):
            cpoint = min_prob + step_size*i
            pred = (df['prob'] > cpoint).astype(int)
            true = df['label'].astype(int)
            tn, fp, fn, tp = confusion_matrix(true, pred).ravel()
            se, sp = tp/(tp+fn), tn/(tn+fp)
            iu = abs(auc-se) + abs(auc-sp)
            if iu < min_iu:
                min_iu = iu
                opt_cpoint, opt_se, opt_sp = cpoint, se, sp
        return opt_cpoint, opt_se, opt_sp, min_iu
    
    def opt_sesp(self):
        """
        compute sensitivity and specificity at optimal cut-point determined by Index of Union method
        https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5470053/
        """
        def print_metrics(df):
            cpoint, se, sp, iu = self.index_of_union(df, n_steps=500)
            print("SE/SP: {:.3f}, {:.3f}".format(se, sp))
            print("Cut-point, IU: {:.3f}, {:.3f}".format(cpoint, iu))
        def format(x):
            return f"{x:.3f}"
        
        rows, cols = [], ['method', 'cohort', 'se', 'sp', 'cut-point', 'IU']
        for cohort, results in self.results.items():
            if cohort == NAMES.mcl:
                for institute in self.mcl_institutes:
                    for method, path in results.items():
                        df = self.read_results(path)
                        df = self.mcl_cohort.merge(df, on=['pid', 'scandate'])
                        df = df[df['institute']==self.mcl_institutes[institute]]
                        if len(df) > 0:
                            cpoint, se, sp, iu = self.index_of_union(df, n_steps=1000)
                            rows.append([method, institute, format(se), format(sp), format(cpoint), format(iu)])
                            # print(f"{method} on {institute}")
                            # print_metrics(df)
            else:
                for method, path in results.items():
                    df = self.read_results(path)
                    cpoint, se, sp, iu = self.index_of_union(df, n_steps=1000)
                    rows.append([method, cohort, format(se), format(sp), format(cpoint), format(iu)])
                    # print(f"{method} on {cohort}")
                    # print_metrics(df)
        return pd.DataFrame(rows, columns=cols)

if __name__ == '__main__':
    a = BestMethodStats()
    # a.test()
    
    # compare performance with imputation vs subjects with missing variables removed
    # a.print_aucs(remove_null=False)
    # print("NOT IMPUTED ==============================")
    # a.print_aucs(remove_null=True)
    
    # dataframe of sensitivity and specificity at optimal cut-point determined by Index of Union method
    opt_df = a.opt_sesp()
    opt_df.to_csv(os.path.join(D.DATA_DIR, "stats", "optimal_sesp.csv"), index=False)
