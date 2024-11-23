import os
from functools import cached_property
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
from dataclasses import dataclass
from collections.abc import Callable
from typing import Optional

from lungbl.utils.tabular import read_tabular, format_datetime_str

import lungbl.definitions as D

@dataclass
class Names:
    nlst: str="nlst"
NAMES = Names()

@dataclass
class CachedCohort:
    name: str
    init: Callable
    noduleft_data: Optional[str] = None # pretrained nodule features
    img_data: Optional[str] = None      # raw nifti images
    imgprep_data: Optional[str] = None  # preprocessed images
    imgbbox_data: Optional[str] = None  # bounding boxes of nodule ROIs
    imgprep_list: Optional[str] = None  # list of preprocessed images
    dlsft64_data: Optional[str] = None  # pretrained 64-dim DLS features
    dlsft128_data: Optional[str] = None # pretrained 128-dim DLS features


NLST_CACHE = CachedCohort(
    name=NAMES.nlst,
    cohort=os.path.join(D.DATASET_DIR, 'nlst/nlst.csv'),
    scan_cohort=os.path.join(D.DATASET_DIR, 'nlst/nlst_scan.csv'),
    test=os.path.join(D.DATASET_DIR, 'nlst/ardila_test_set.xlsx'),
    img_data=os.path.join(D.DATASET_DIR, 'nlst/DeepLungScreening/nifti'),
    imgprep_list=os.path.join(D.DATASET_DIR, 'nlst/nlst_prep.csv'),
    imgprep_data=os.path.join(D.DATASET_DIR, 'nlst/DeepLungScreening/prep'),
    imgbbox_data=os.path.join(D.DATASET_DIR, 'nlst/DeepLungScreening/bbox'),
    noduleft_data=os.path.join(D.DATASET_DIR, 'nlst/liao/feat128/'),
    dlsft64_data=os.path.join(D.DATASET_DIR, 'nlst/DeepLungScreening/feat64'),
    dlsft128_data=os.path.join(D.DATASET_DIR, 'nlst/DeepLungScreening/feat128'),
)
####################################################################################################
## Cohort Wrappers
####################################################################################################

class CohortWrapper():
    def __init__(
        self,
        cache: CachedCohort,
    ):
        self.cache = cache

    @cached_property
    def cohort(self):
        return read_tabular(self.cache.cohort)

    @cached_property
    def scan_cohort_df(self):
        df = read_tabular(self.cache.scan_cohort)
        df = df[df['scandate'].notnull()]
        return df
    

class NLST_CohortWrapper(CohortWrapper):
    def __init__(
        self,
        cache=NLST_CACHE,
        label="lung_cancer",
        ):
        super().__init__(cache=cache)

        self.label = label
        self.ft = pd.read_csv(self.cache.imgprep_list, dtype={'pid':str})
        
    @cached_property
    def scan_cohort_df(self):
        df = self.cohort.scan_cohort_df
        df = df[df['scandate'].notnull()]
        return df

    @cached_property
    def train_cohort(self):
        df = self.scan_cohort_df[self.scan_cohort_df['lung_cancer'].notnull()]
        df = df.groupby(['pid', 'scandate'], as_index=False).max() # if multiple scans within a scandate, take a random one
        return df[~df['pid'].isin(self.test_scan['pid'])]

    @cached_property
    def test_scan(self):
        test = read_tabular(self.cache.test, {'dtype':{'patient_id':str}})
        df = self.scan_cohort_df[self.scan_cohort_df['lung_cancer'].notnull()]
        scan = df.groupby(['pid', 'scandate'], as_index=False).first()
        scan['study_yr'] = (scan.groupby('pid')['scandate'].rank(method='dense', ascending=True) - 1).astype(int)
        test = scan.merge(test, left_on=['pid', 'study_yr'], right_on=['patient_id', 'study_yr'], how='inner')
        return test.groupby(['pid', 'study_yr'], as_index=False).max() # duplicate rows

    @cached_property
    def test_nodule(self):
        return self.test_scan[self.test_scan['nodule_count'].notnull()]

    @cached_property
    def test_cs(self):
        """One scan per subject, using the scan closest to outcome"""
        return self.test_scan.iloc[self.test_scan.groupby('pid')['study_yr'].idxmax()]
    
    @cached_property
    def test_cs_nodule(self):
        return self.test_cs[self.test_cs['nodule_count'].notnull()]

    
    # TRAIN for Liao ================================================================================
    @cached_property
    def ft_train(self):
        ft = self.ft.copy()
        ft['scandate'] = ft['scandate'].apply(lambda x: format_datetime_str(x, format="%Y"))
        self.train_cohort['scandate'] = self.train_cohort['scandate'].apply(lambda x: format_datetime_str(x, format="%Y"))
        df = ft[['pid', 'scandate', 'imgprep_path']].merge(self.train_cohort, on=['pid', 'scandate'], how='inner')
        return df
    

    # TEST for Liao ================================================================================
    @cached_property
    def ft_test_nodule(self):
        ft = self.ft.copy()
        ft['scandate'] = ft['scandate'].apply(lambda x: format_datetime_str(x, format="%Y"))
        self.test_nodule['scandate'] = self.test_nodule['scandate'].apply(lambda x: format_datetime_str(x, format="%Y"))
        df = ft[['pid', 'scandate', 'imgprep_path']].merge(self.test_nodule, on=['pid', 'scandate'], how='inner')
        return df
    
    @cached_property
    def ft_test_scan(self):
        ft = self.ft.copy()
        ft['scandate'] = ft['scandate'].apply(lambda x: format_datetime_str(x, format="%Y"))
        self.test_scan['scandate'] = self.test_scan['scandate'].apply(lambda x: format_datetime_str(x, format="%Y"))
        df = ft[['pid', 'scandate', 'imgprep_path']].merge(self.test_scan, on=['pid', 'scandate'], how='inner')
        return df
