# Benchmarking lung cancer models and datasets

T.Z. Li, K. Xu, A. Krishnan, R. Gao, M.N. Kammer, S. Antic, D. Xiao, M. Knight, Y. Martinez, R. Paez, R.J. Lentz, S. Deppen, E.L. Grogan, T.A. Lasko, K.L. Sandler, F. Maldonado, B.A. Landman, No winners: Performance of lung cancer prediction models depends on screening-detected, incidental, and biopsied pulmonary nodule use cases, (2024). https://arxiv.org/abs/2405.10993v1

We ran 8 predictive models for lung cancer diagnosis across 9 different cohorts to evaluate their performance in different clinical settings. This repo supports training and inference of these models for a public lung screening dataset [NLST](https://cdas.cancer.gov/nlst/), but other datasets from this study are private. 

# Usage
## Install
1. `pip install -r requirements.txt`
2. Clone https://github.com/MASILab/DeepLungScreening
3. Edit `definitions.py` to point to working directories

## Datasets
Thsi repo can be run with any lung CT dataset with the following setup. We will use the [NLST](https://cdas.cancer.gov/nlst/) in this example. Make the corresponding name and path replacements in `cachedcohorts.py` like so:
```
# cachedcohorts.py
NLST_CACHE = CachedCohort(
    name=NAMES.nlst,
    cohort=os.path.join(D.DATASET_DIR, 'nlst/nlst.csv'),
    scan_cohort=os.path.join(D.DATASET_DIR, 'nlst/nlst_scan.csv'),
    noduleft_data=os.path.join(D.DATASET_DIR, 'nlst/liao/feat128/'),
    img_data=os.path.join(D.DATASET_DIR, 'nlst/DeepLungScreening/nifti'),
    imgprep_data=os.path.join(D.DATASET_DIR, 'nlst/DeepLungScreening/prep'),
    imgbbox_data=os.path.join(D.DATASET_DIR, 'nlst/DeepLungScreening/bbox'),
    imgprep_list=os.path.join(D.DATASET_DIR, 'nlst/nlst_prep.csv'),
    dlsft64_data=os.path.join(D.DATASET_DIR, 'nlst/DeepLungScreening/feat64'),
    dlsft128_data=os.path.join(D.DATASET_DIR, 'nlst/DeepLungScreening/feat128'),
)
```
### nlst.csv
NLST_CACHE.cohort should point to a csv with the format
| pid        | lung_cancer | nodule_count |
| ----------- | ----- | ----- |
| unique patient ID | 0 or 1 label | int (optional) |
### nlst_scan.csv
NLST_CACHE.scan_cohort should point to a csv with the format
| pid        | scandate          | scanorder | lung_cancer | nodule_count |
| ----------- | ------------ | ----- | ----- | ----- |
| unique patient ID | %Y%m%d | int with 0 being earliest scan | 0 or 1 label | int (optional) |
### test_set.csv (optional)
NLST_CACHE.test should point to a csv with the format
Here we use the test set given [Ardila et al.](https://www.nature.com/articles/s41591-019-0447-x) test set 

### nifti/
NLST_CACHE.img_data should point to a directory of CT scans in NIfTI format (`.nii.gz`)

### Liao and DeepLungScreening pipelines
Some models rely on the features from the Liao et al. model. The following pipeline will compute intermediate data and features in the locations specified in `imgprep_data`, `imgbbox_data`, `dlsft64_data`, and `dlsft128_data`. 

1. Preprocessing CT scans and generating list of scans that passed this step in `imgprep_list`:
```
#!/bin/bash
python imgprep.py 1 nlst.test_scan
python imgprep.py prep --prep_dst nlst_prep.csv

```
2. Computing bounding boxes for using a pretrained nodule detection model from Liao et al.
```
python imgprep.py 2 nlst.test_scan
```
1. Computing feature vectors using a pretrained ResNet from Liao et al.
```
python imgprep.py 3 nlst.test_scan
```
1. Make predictions with a multimodal model from DeepLungScreening.
```
python imgprep.py 4 nlst.test_scan --predictions dls.csv
```

## Model Training and Inference
```
#!/bin/bash
python cli.py train nlst.train_cohort
python cli.py test nlst.test_scan
```
Replace `nlst.train_cohort` with `nlst.ft_train` and `nlst.test_scan` with `nlst.ft_test_scan` if you are running a model that uses the Liao or DLS pipelines. This change leaves out the subjects that were not able to be processed by the Liao pipeline.
