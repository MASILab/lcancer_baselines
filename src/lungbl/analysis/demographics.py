import os, pandas as pd, numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, SimpleImputer
from collections import OrderedDict

from lungbl.cli import COHORTS
import lungbl.definitions as D

class CohortDemographics():
    def __init__(self):
        self.COHORTS = OrderedDict()
        # load cohorts as pandas dataframes

        self.mcl_institutes = {
                1: "VUMC",
                2: "UPMC",
                3: "DECAMP",
                4: "UCD",
            }
        
        self.imputer = IterativeImputer(max_iter=10, random_state=D.RANDOM_SEED)
        self.cat_imputer = SimpleImputer(strategy='most_frequent')
        self.clinical_cols = ["age", "gender", "bmi", "emphysema", "phist", "fhist", "smo_status", "pkyr", 
            "nodule_size", "nodule_attenuation", "nodule_spiculation", "upper_lobe",
        ]
        self.multi_categorical = ["smo_status", "nodule_attenuation", "nodule_spiculation", "upper_lobe"]

    def print_all(self):
        for cname, init in self.COHORTS.items():
            print(cname)
            print("="*10)
            label = "cancer_year1" if cname.split(".")[0] == "nlst" else "lung_cancer"
            self.print_cohort(init(), label=label)
            print("\n")
            
    def print_mcl(self, df):
        for k, institute in self.mcl_institutes.items():
            df_inst = df[df['institute'] == k]
            print(institute)
            print("="*10)
            self.print_cohort(df_inst)
            print("\n")
            
    def print_imputed_mcl(self, df):
        for k, institute in self.mcl_institutes.items():
            df_inst = df[df['institute'] == k]
            imputed, missing = self.imputed(df_inst), self.missing(df_inst)
            print(institute)
            print(missing)
            print("="*10)
            self.print_cohort(imputed)
            print("\n")
    
    def imputed(self, df):
        id_df, impute_df = df[df.columns.difference(self.clinical_cols)], df[self.clinical_cols]
        scalar_impute = impute_df[impute_df.columns.difference(self.multi_categorical)]
        # categorical imputation first with mode
        cat_impute = impute_df[self.multi_categorical]
        cat_imputed = self.cat_imputer.fit_transform(cat_impute)
        cat_imputed = pd.DataFrame(cat_imputed, columns=cat_impute.columns, index=cat_impute.index)
        # scalar imputation with iterative imputer
        scalar_imputed = self.imputer.fit_transform(scalar_impute)
        scalar_imputed = pd.DataFrame(scalar_imputed, columns=scalar_impute.columns, index=scalar_impute.index)
        imputed = pd.concat([scalar_imputed, cat_imputed], axis=1)
        imputed = id_df.merge(imputed, left_index=True, right_index=True)
        return imputed
    
    def missing(self, df):
        missing = []
        for col in self.clinical_cols:
            missing.append([col, len(df[df[col].isnull()])])
        return pd.DataFrame(missing, columns=['variable', 'n_null'])

    def print_cohort(self,
        df: pd.DataFrame,
        label: str="lung_cancer",
        ):
        grp = df.groupby(['pid'], as_index=False).max()
        printmap = OrderedDict()
        
        # printmap["n"] = lambda: len(grp)
        # printmap["cancer"] = lambda: (sum(grp[label]), sum(grp[label])/len(grp))
        # printmap["n_scan"] = lambda: len(df)
        # printmap["cancer_scan"] = lambda: (sum(df[label]), sum(df[label])/len(df))
        # printmap["age"] = lambda: self.mean_std(grp['age'])
        # printmap["sex (male)"] = lambda: self.proportion(grp["gender"])
        # printmap["bmi"] = lambda: self.mean_std(grp["bmi"])
        # printmap["phist"] = lambda: self.proportion(grp["phist"])
        # printmap["fhist"] = lambda: self.proportion(grp["fhist"])
        # printmap["smoke status"] = lambda: self.proportion(grp["smo_status"])
        # printmap["pkyr"] = lambda: self.mean_std(grp["pkyr"])
        # printmap["edu"] = lambda: self.proportion(grp["education"])
        # printmap["nodule_size"] = lambda: self.mean_std(df["nodule_size"])
        # printmap["nodule_count"] = lambda: self.mean_std(df["nodule_count"])
        # printmap["nodule_attenuation"] = lambda: self.proportion(df["nodule_attenuation"])
        # printmap["nodule_spic"] = lambda: self.proportion(df["nodule_spiculation"])
        # printmap["upper_lobe"] = lambda: self.proportion(df["upper_lobe"])
        printmap["slice_thickness"] = lambda: self.mean_std(df['slice_thickness'])

        for k, v in printmap.items():
            # print(f"{k}: {v()}")
            try:
                print(f"{k}: {v()}")
            except KeyError as e:
                print(e)
            
    @staticmethod
    def mean_std(s: pd.Series):
        return (s.mean(), s.std())

    @staticmethod
    def proportion(p: pd.Series):
        return (p.value_counts(dropna=False), p.value_counts(dropna=False, normalize=True))

if __name__ == "__main__":
    c = CohortDemographics()
    c.print_all()
    # c.print_mcl(c.COHORTS['mcl.scan_cohort_df']())
    
    # print demographics after imputation
    # df = COHORTS['nlst.test_cs_nodule']()
    # imputed = c.imputed(df)
    # missing = c.missing(df)
    # print(missing)
    # c.print_cohort(imputed)
    
    # print MCL demographics after imputation
    # df = COHORTS['mcl.cs']()
    # c.print_imputed_mcl(df)