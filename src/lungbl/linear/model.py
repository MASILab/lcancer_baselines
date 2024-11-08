import os, numpy as np, pandas as pd
from functools import cached_property
from typing import TypedDict, OrderedDict
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

import lungbl.definitions as definitions

class LogisticRegression():
    def __init__(self, 
        impute: bool=True, 
        label: str="lung_cancer",
        key_covar: str=None,
        ):
        self.name = None
        self.imputer = IterativeImputer(max_iter=10, random_state=definitions.RANDOM_SEED) if impute else None
        self.intercept = 0
        self.key_covar = key_covar
    
    def impute(self, df):
        imputed = self.imputer.fit_transform(df)
        return pd.DataFrame(imputed, columns=df.columns, index=df.index)

    def logit_from_df(self, df):
        logit = pd.Series([self.intercept]*len(df), name=f"{self.name}_risk")
        for col, beta in self.betas.items():
            logit += beta*df[col]
        return logit
    
    def sigmoid(self, x):
        return 1/(1+np.exp(-x))
    
    def forward(self, covars: pd.DataFrame) -> pd.Series:
        raise NotImplementedError

    def predict(self, cohort: pd.DataFrame,) -> pd.DataFrame:
        df = cohort.reset_index(drop=True)
        
        if self.key_covar is not None:
            df = df[df[self.key_covar].notnull()] # drop rows where key_covar is null
            print(f"Dropped {len(cohort) - len(df)} / {len(cohort)} rows where {self.key_covar} is null")
            
        covars = df[self.transforms.keys()]
        risk = self.forward(covars)
        return df.merge(risk, left_index=True, right_index=True)

class Brock(LogisticRegression):
    def __init__(self, key_covar="nodule_size", **kwargs):
        """
        key_covar: str - drop rows where key_covar is null. asserts all subjects have a value for key_covar
        """
        super().__init__(key_covar=key_covar, **kwargs)

        self.name = "brock"
        self.multi_categorical = ['nodule_attenuation'] # vars with more than two categories to one-hot encode
        self.intercept = -6.7892
        self.betas={
            "age": 0.0287,
            "gender": 0.6011,
            "fhist": 0.2961,
            "emphysema": 0.2953,
            "nodule_size": -5.3854,
            "nodule_attenuation_0.0": 0.0000,
            "nodule_attenuation_1.0": 0.377,
            "nodule_attenuation_2.0": -0.1276,
            "nodule_count": -0.0824,
            "nodule_spiculation": 0.7729,
            "upper_lobe": 0.6581,
        }

        self.transforms = {}
        self.transforms["age"] = lambda x: x - 62
        self.transforms["gender"] = lambda x: x.astype(float)
        self.transforms["fhist"] = lambda x: x.astype(float)
        self.transforms["emphysema"] = lambda x: x.astype(float)
        self.transforms["nodule_size"] = lambda x: np.sqrt(10/np.maximum(x - 4, 1e-6)) - 1.58113883
        self.transforms["nodule_attenuation"] = lambda x: x.astype(float)
        self.transforms["nodule_count"] = lambda x: x - 4
        self.transforms["nodule_spiculation"] = lambda x: x.astype(float)
        self.transforms["upper_lobe"] = lambda x: x.astype(float)


    def forward(self, covars: pd.DataFrame) -> pd.Series:
        post_transform = {col:func(covars[col]) for col, func in self.transforms.items() if col in covars}
        # one hot encoding
        post_transform = pd.get_dummies(pd.DataFrame(post_transform), columns=self.multi_categorical)
        # impute
        if self.imputer:
            post_transform = self.impute(post_transform)
            # self.imputer = self.imputer.fit(post_transform)
            # post_transform = pd.DataFrame(self.imputer.transform(post_transform), columns=post_transform.columns)

        # compute risk
        risk = self.logit_from_df(post_transform)
        risk = self.sigmoid(risk)
        return risk

class Mayo(LogisticRegression):
    def __init__(self, key_covar="nodule_size", **kwargs):
        super().__init__(key_covar=key_covar, **kwargs)

        self.name = "mayo"
        self.intercept = -6.8272
        self.betas={
            "age": 0.0391,
            "phist": 1.3388,
            "smo_status": 0.7917,
            "nodule_size": 0.1274,
            "upper_lobe": 0.7838,
            "nodule_spiculation": 1.0407,
        }
        self.transforms = {}
        self.transforms["age"] = lambda x: x
        self.transforms["phist"] = lambda x: x.astype(float)
        self.transforms["smo_status"] = lambda x: x.astype(float)
        self.transforms["nodule_size"] = lambda x: x
        self.transforms["upper_lobe"] = lambda x: x.astype(float)
        self.transforms["nodule_spiculation"] = lambda x: x.astype(float)

    def forward(self, covars: pd.DataFrame) -> pd.Series:
        post_transform = {col:func(covars[col]) for col, func in self.transforms.items() if col in covars}
        post_transform = pd.DataFrame(post_transform)
        # impute
        if self.imputer:
            post_transform = self.impute(post_transform)

        # compute risk
        risk = self.logit_from_df(post_transform)
        risk = self.sigmoid(risk)
        return risk

class PLCO2012(LogisticRegression):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.name = "plco"
        self.multi_categorical = ["race"]
        self.intercept = -4.532506
        self.betas={
            "age": 0.0778868,
            "race_0": 0.0,
            "race_1": 0.0,
            "race_2": 0.3944778, # black
            "race_3": -0.7434744, # hispanic
            "race_4": -0.466585, # asian
            "race_5": 0, # american indian or alaskan native
            "race_6": 1.027152, # native hawaiian or pacific islander
            "education": 0.0812744,
            "bmi": -0.0274194,
            "copd": 0.3553063,
            "phist": 0.4589971,
            "fhist": 0.587185,
            "smo_status": 0.2597431,
            "smo_intensity": - 1.822606,
            "smo_duration": 0.0317321,
            "quit_time": -0.0308572,
        }
        self.transforms = {}
        self.transforms["age"] = lambda x: x - 62
        self.transforms["race"] = lambda x: x.astype(int)
        self.transforms["education"] = lambda x: x - 4
        self.transforms["bmi"] = lambda x: x - 27
        self.transforms["copd"] = lambda x: x.astype(float)
        self.transforms["phist"] = lambda x: x.astype(float)
        self.transforms["fhist"] = lambda x: x.astype(float)
        self.transforms["smo_status"] = lambda x: x - 1
        self.transforms["smo_intensity"] = lambda x: (10 / x) - 0.4021541613
        self.transforms["smo_duration"] = lambda x: x - 27
        self.transforms["quit_time"] = lambda x: x - 10

    def forward(self, covars: pd.DataFrame) -> pd.Series:
        post_transform = {col:func(covars[col]) for col, func in self.transforms.items() if col in covars}
        # one hot encoding
        post_transform = pd.get_dummies(pd.DataFrame(post_transform), columns=self.multi_categorical)
        for col in self.multi_categorical:
            post_transform.loc[post_transform[f"{col}_nan"]==True, post_transform.columns.str.startswith(f"{col}_")] = np.nan
        # impute
        if self.imputer:
            post_transform = self.impute(post_transform)
            # self.imputer = self.imputer.fit(post_transform)
            # post_transform = pd.DataFrame(self.imputer.transform(post_transform), columns=post_transform.columns)

        # compute risk
        risk = self.logit_from_df(post_transform)
        risk = self.sigmoid(risk)
        return risk