import os, math, functools, random, numpy as np, pandas as pd
from scipy.stats import wilcoxon
from cohorts.cli import NAMES
from cohorts.definitions import SUBTYPES
from lungbl.utils.statutils import bootstrap_samples, bootstrap
from lungbl.utils.tabular import read_tabular, format_datetime_str

from lungbl import definitions as D
from lungbl.utils.modeling import auc_from_df, ppv_from_df
from lungbl.analysis.stats import BL, RESULTS

# SUBTYPE_RESULTS = {k: RESULTS[k] for k in (NAMES.nlst, NAMES.mcl, NAMES.bronch)}
SUBTYPE_RESULTS = {k: RESULTS[k] for k in (NAMES.nlst, NAMES.mcl)}

class LCSubtype_Analaysis():
    def __init__(self):
        self.read_results = lambda x: read_tabular(os.path.join(D.DATA_DIR, x), fargs={'dtype':{'pid': str, 'scandate': str}})
        self.auc_agg = lambda x: auc_from_df(x, cols=('label', 'prob'))
        self.ppv_agg = lambda x: ppv_from_df(x, cols=('label', 'prob'))
        self.min_class_size = 10
        self.cohorts = {
            # load in cohorts from pandas dataframes
        }
        self.mcl_institutes = {
            "VUMC": 1,
            "UPMC": 2, 
            "DECAMP": 3,
            "UCD": 4,
        }
        self.subtype_map = {
            SUBTYPES.adenocarcinoma: SUBTYPES.adenocarcinoma,
            SUBTYPES.sclc: SUBTYPES.sclc,
            SUBTYPES.squamous: SUBTYPES.squamous,
            SUBTYPES.nsclc: SUBTYPES.other_nsclc,
            SUBTYPES.neuroendocrine: SUBTYPES.other_nsclc,
            SUBTYPES.large_cell: SUBTYPES.other_nsclc,
            SUBTYPES.mixed: SUBTYPES.other_nsclc,
            SUBTYPES.nos: SUBTYPES.nos,
            SUBTYPES.secondary_metastatic: np.nan,
            SUBTYPES.other: np.nan,
        }
        # best method for cohort: (SCLC, NSCLC)
        self.best ={
            NAMES.nlst: (BL.sybil, BL.sybil),
            "VUMC": (BL.dls, BL.dls),
        }
    @staticmethod
    def seed_random():
        random.seed(D.RANDOM_SEED)
        os.environ['PYTHONHASHSEED'] = str(D.RANDOM_SEED)
        np.random.seed(D.RANDOM_SEED)
    
        
    def all_results(self):
        self.seed_random()
        for cohort, results in SUBTYPE_RESULTS.items():
            cohort_df = self.cohorts[cohort]()
            if cohort == NAMES.mcl:
                cohort_df = cohort_df[cohort_df['institute'] == self.mcl_institutes['VUMC']]
            cohort_df['lc_subtype'] = cohort_df['lc_subtype'].apply(self.map_subtype)
            
            best_sclc, best_nsclc = self.best[cohort]
            best_sclc_df = self.results_by_subtype(cohort_df, self.read_results(results[best_sclc]))
            best_nsclc_df = self.results_by_subtype(cohort_df, self.read_results(results[best_nsclc]))
            for method, path in results.items():
                print(f"======== {method} on {cohort} ========")
                pred_df = self.read_results(path)
                results_df = self.results_by_subtype(cohort_df, pred_df)
                sclc_auc, nsclc_auc = self.auc_sclc_vs_nsclc(results_df)
                print(f"SCLC --- {sclc_auc}")
                if method != best_sclc:
                    self.wilcoxon_signed_rank(best_sclc_df, results_df, on_sclc=True)
                print(f"NSCLC --- {nsclc_auc}")
                if method != best_nsclc:
                    self.wilcoxon_signed_rank(best_nsclc_df, results_df, on_sclc=False)
                
    def wilcoxon_signed_rank(self, dfa, dfb, on_sclc=False):
        subtype = 1.0 if on_sclc else 0.0
        dfa_benign, dfb_benign = dfa[dfa['label']==0], dfb[dfb['label']==0]
        dfa_case, dfb_case = dfa[dfa['sclc']==subtype], dfb[dfb['sclc']==subtype]
        if len(dfb_case) >= self.min_class_size:
            sample_a = bootstrap_samples(
                pd.concat([dfa_case, dfa_benign]),
                agg=self.auc_agg,
                n=1000
            )
            sample_b = bootstrap_samples(
                pd.concat([dfb_case, dfb_benign]),
                agg=self.auc_agg,
                n=1000
            )
            statobj = wilcoxon(sample_a['agg'], sample_b['agg'])
            print(statobj.statistic, statobj.pvalue)
                
            # if method != best_sclc:
            #     self.wilcoxon_signed_rank(best_sclc, results_df)
            # if method != best_nsclc:
            #     self.wilcoxon_signed_rank(best_nsclc, results_df)
            
            # else:
            #     # mcl cohorts, only VUMC has enough subtypes for comparison
            #     # for institute in self.mcl_institutes:
            #     cohort = 'VUMC'
            #     for method, path in results.items():
            #         print(f"======== {method} on {} ========")
            #         cohort_df = self.cohorts[cohort]()
            #         cohort_df = cohort_df[cohort_df['institute'] == self.mcl_institutes[institute]]
            #         cohort_df['lc_subtype'] = cohort_df['lc_subtype'].apply(self.map_subtype)
            #         pred_df = self.read_results(path)
            #         results_df = self.results_by_subtype(cohort_df, pred_df)
            #         sclc_auc, nsclc_auc = self.auc_sclc_vs_nsclc(results_df)
            #         print(f"SCLC --- {sclc_auc}")
            #         print(f"NSCLC --- {nsclc_auc}")
            
            # p-value test

    
                
    
    def map_subtype(self, x):
        if pd.notnull(x):
            return self.subtype_map[x]
        else:
            return np.nan
    
    @staticmethod
    def results_by_subtype(cohort, pred):
        cohort['scandate'] = cohort['scandate'].apply(format_datetime_str)
        pred['scandate'] = pred['scandate'].apply(format_datetime_str)
        scohort = cohort[['pid', 'scandate', 'lc_subtype', 'sclc']]
        merged = pred.merge(scohort, on=['pid', 'scandate'], how='inner')
        merged = merged.groupby('pid', as_index=False).first()
        return merged

    def auc_sclc_vs_nsclc(self, df):
        benign = df[df['label']==0]
        sclc = df[df['sclc'] == 1.0]
        nsclc = df[df['sclc'] == 0.0]
        print(f"SCLC: {len(sclc)}, NSCLC: {len(nsclc)}, Benign: {len(benign)}")

        if len(sclc) >= self.min_class_size:
            sclc_auc = bootstrap(pd.concat([sclc, benign]), self.auc_agg, n=1000)
        else:
            sclc_auc = None
        
        if len(nsclc) >= self.min_class_size:
            nsclc_auc = bootstrap(pd.concat([nsclc, benign]), self.auc_agg, n=1000)
        else:
            nsclc_auc = None

        return sclc_auc, nsclc_auc
    
    def auc_subtype(self, df):
        subtypes = df['lc_subtype'].unique().tolist()
        for subtype in subtypes:
            df_subtype = df[df['lc_subtype'] == subtype]
            auc = self.bootstrap_auc(df_subtype)
            print(f"{subtype}: {auc}")

if __name__ == "__main__":
    analysis = LCSubtype_Analaysis()
    analysis.all_results()