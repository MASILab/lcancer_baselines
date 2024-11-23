import os, shutil, tempfile, argparse, pandas as pd
from pathlib import Path
from functools import cached_property

from lungbl.utils.tabular import read_tabular, format_datetime_str
from lungbl.utils.statutils import bootstrap
from lungbl.utils.modeling import auc_from_df
from lungbl.cachedcohorts import NAMES, CachedCohort, NLST_CACHE, NLST_CohortWrapper
import lungbl.definitions as D

CACHED_COHORTS = {
    NAMES.nlst: (NLST_CACHE),
}

class DLS_ImgPrep():
    """
    Run Liao/DeepLungScreening pipeline on a cached cohort
    1. make dir with flat layout of all scans in format {pid}time{scandate}.nii.gz
    2. run DLS step 1 -> prep dir
    3. run DLS step 2 -> bbox dir
    """
    
    def __init__(self, 
            cache: CachedCohort,
            cohort: pd.DataFrame=None,
            date_format: str="%Y%m%d",
        ):
        self.cache = cache
        self.date_format = date_format
        c = cache.init().scan_cohort_df if cohort is None else cohort
        self.cohort = c[c['scandate'].notnull()]
        self.cohort['scandate'] = self.cohort['scandate'].apply(lambda x: format_datetime_str(x, format=self.date_format))
        
        self.dlsroot = D.DLS_ROOT
        self.pyenv = os.path.join(self.dlsroot, "env/bin/python")
        self.step1 = os.path.join(self.dlsroot, "1_preprocess/step1_main.py")
        self.step2 = os.path.join(self.dlsroot, "./2_nodule_detection/step2_main.py")
        self.step3 = os.path.join(self.dlsroot, "./3_feature_extraction/step3_main.py")
        self.step4 = os.path.join(self.dlsroot, "./4_co_learning/step4_main.py")
        self.step4_factors = ['id', 'with_image', 'with_marker', 'age',  'education',  'bmi', 'phist', 'fhist','smo_status', 'quit_time', 'pkyr']
        
    def make_nifti_dir(self, symlink=True):
        for idx, row in self.cohort.iterrows():
            scanid, fpath = row.id, row.fpath
            dst = os.path.join(self.cache.img_data, f"{scanid}.nii.gz")
            if symlink:
                os.symlink(fpath, dst)
            else:
                shutil.copyfile(fpath, dst)

    @cached_property
    def scan_df(self):
        scanlist = []
        for f in os.scandir(self.cache.imgprep_data):
            fname = os.path.basename(f.path)
            if fname.split("_")[-1] == "clean.nii.gz":
                pid, scandate, *_ = fname.split("_")[0].split("time")
                scanlist.append([pid, scandate, f.path])
                    
        self._scan_df = pd.DataFrame(scanlist, columns=['pid', 'scandate', 'imgprep_path'])
        self._scan_df['scandate'] = self._scan_df['scandate'].apply(lambda x: self._parse_scandate(x))
        return self._scan_df

    @staticmethod
    def _parse_scandate(x):
        if x.isdigit():
            return x
        else:
            date = x.split("-")[2]
            if date.isdigit():
                return date
            else:
                return None
            
    @cached_property
    def prep_df(self):
        if not os.path.exists(self.cache.imgprep_list):
            scans = self.cohort.groupby(['pid', 'scandate'], as_index=False).max()
            scans['scandate'] = scans['scandate'].apply(lambda x: format_datetime_str(x, format=self.date_format))
            self.scan_df['scandate'] = self.scan_df['scandate'].apply(lambda x: format_datetime_str(x, format=self.date_format))
            self._prep_df = scans.merge(self.scan_df, on=['pid', 'scandate'], how='inner')
            self._prep_df.to_csv(self.cache.imgprep_list, index=False)
        else:
            self._prep_df = read_tabular(self.cache.imgprep_list, {'dtype':{'pid': str}})
        return self._prep_df
            
    def dls_step2(self, n_jobs=1):
        with tempfile.NamedTemporaryFile(mode='w') as f:
            self.prep_df['id'] = self.prep_df.apply(lambda x: f"{x['pid']}time{format_datetime_str(x['scandate'], format=self.date_format)}", axis=1)
            self.prep_df['id'].to_csv(f.name) # write a temp file with scanids
            bbox_root = self.cache.imgbbox_data
            prep_root = self.cache.imgprep_data
            Path(bbox_root).mkdir(exist_ok=True, parents=True)
            
            cmd = f"{self.pyenv} {self.step2} --sess_csv {f.name} --prep_root {prep_root} --bbox_root {bbox_root} --n_jobs {n_jobs}"
            os.system(cmd)

    def dls_step1(self, n_jobs=1):
        with tempfile.NamedTemporaryFile(mode='w') as f:
            self.cohort['id'] = self.cohort.apply(lambda x: f"{x['pid']}time{format_datetime_str(x['scandate'], format=self.date_format)}", axis=1)
            self.cohort['id'].to_csv(f.name) # write a temp file with scanids
            ori_root = self.cache.img_data
            prep_root = self.cache.imgprep_data
            Path(ori_root).mkdir(exist_ok=True, parents=True)
            Path(prep_root).mkdir(exist_ok=True, parents=True)
            
            cmd = f"{self.pyenv} {self.step1} --sess_csv {f.name} --prep_root {prep_root} --ori_root {ori_root} --n_jobs {n_jobs}"
            os.system(cmd)
            
    def dls_step3(self, cohort: pd.DataFrame=None):
        """cohort: run step3 on a cohort other than self.prep_df"""
        with tempfile.NamedTemporaryFile(mode='w') as f:
            cohort['scandate'] = cohort['scandate'].apply(lambda x: format_datetime_str(x, format=self.date_format))
            self.prep_df['scandate'] = self.prep_df['scandate'].apply(lambda x: format_datetime_str(x, format=self.date_format))
            prep = self.prep_df.merge(cohort, on=['pid', 'scandate']) if cohort is not None else self.prep_df
            prep['id'] = prep.apply(lambda x: f"{x['pid']}time{format_datetime_str(x['scandate'], format=self.date_format)}", axis=1)
            prep['id'].to_csv(f.name)
            bbox_root = self.cache.imgbbox_data
            prep_root = self.cache.imgprep_data
            feat64 = self.cache.dlsft64_data
            feat128 = self.cache.dlsft128_data
            Path(feat64).mkdir(exist_ok=True, parents=True)
            Path(feat128).mkdir(exist_ok=True, parents=True)
            cmd = f"{self.pyenv} {self.step3} --sess_csv {f.name} --prep_root {prep_root} --bbox_root {bbox_root} --feat64 {feat64} --feat128 {feat128}"
            os.system(cmd)
    
    def dls_step4(self, 
            save_pred, 
            cohort: pd.DataFrame=None,
            label: str='cancer_year1', # label column name
            ):
        with tempfile.NamedTemporaryFile(mode='w') as f:
            cohort['scandate'] = cohort['scandate'].apply(lambda x: format_datetime_str(x, format=self.date_format))
            self.prep_df['scandate'] = self.prep_df['scandate'].apply(lambda x: format_datetime_str(x, format=self.date_format))
            prep = self.prep_df[['pid', 'scandate']].merge(cohort, on=['pid', 'scandate']) if cohort is not None else self.prep_df
            prep['id'] = prep.apply(lambda x: f"{x['pid']}time{format_datetime_str(x['scandate'], format=self.date_format)}", axis=1)
            prep[self.step4_factors].to_csv(f.name)
            feat_root = self.cache.dlsft128_data
            cmd = f"{self.pyenv} {self.step4} --sess_csv {f.name} --feat_root {feat_root} --save_csv_path {save_pred}"
            os.system(cmd)
        
        pred = read_tabular(save_pred)
        pred = pred.merge(prep[['pid', 'scandate', 'id', label]], on=['id'])
        
        # remove NaN results and flag
        pred_nonan = pred[pred['pred'].notnull()]
        print(f"WARNING: {len(pred[pred['pred'].isnull()])} results were removed due to NaN")
        
        # export results
        pred_nonan['label'] = pred_nonan[label]
        pred_nonan['prob'] = pred_nonan['pred']
        pred_nonan[['pid', 'scandate', 'label', 'prob']].to_csv(save_pred)
        
        # compute bootstrap CIs
        self.dls_bootstrap(pred_nonan, label=label)
            
    def dls_bootstrap(self, 
            pred: str, # predictions with cols [id, prob, label]
            label: str='label', # label column name
        ):
        """ compute the bootstrap CIs for predictions from step4 """
        df = read_tabular(pred)
        metrics = bootstrap(
            df,
            agg=lambda x: auc_from_df(x, cols=('label', 'prob')),
        )
        print(metrics)

COHORTS = {
    'nlst.train_cohort': lambda: NLST_CohortWrapper().train_cohort,
    'nlst.test_scan': lambda: NLST_CohortWrapper().test_scan,
}

def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("step", choices=['1', '2', '3', '4', 'bootstrap', 'prep'])
    parser.add_argument("cache", choices=CACHED_COHORTS.keys()) 
    parser.add_argument("--cohort", choices=COHORTS.keys(), default=None) # step3 and 4 accept optional cohort to restrict results on
    parser.add_argument("--date_format", default="%Y%m%d") # format of scandate
    parser.add_argument("--predictions", default=None) # location to save results from step 4
    parser.add_argument("--label", default='lung_cancer') # label column name
    parser.add_argument("--prep_dst") # location to save list of scans that passed first step
    args = parser.parse_args()
    
    cache = CACHED_COHORTS[args.cache]
    cohort = COHORTS[args.cohort]() if args.cohort is not None else None
    pipeline = DLS_ImgPrep(
        cache=cache, 
        cohort=cohort, 
        date_format=args.date_format
    )
    if args.step == '1':
        pipeline.dls_step1()
    
    elif args.step == '2':
        pipeline.dls_step2()
    
    elif args.step == '3':
        pipeline.dls_step3(cohort)
        
    elif args.step == '4':
        assert args.predictions is not None, "Must specify --predictions location to save results"
        pipeline.dls_step4(args.predictions, cohort, label=args.label)
        
    elif args.step == 'bootstrap':
        assert args.predictions is not None, "Must specify --predictions location to save results"
        pipeline.dls_bootstrap(args.predictions)
        
    elif args.step == 'prep':
        # generate list to prep
        pipeline.prep_df.to_csv(args.prep_dst)
        
if __name__ == '__main__':
    cli()