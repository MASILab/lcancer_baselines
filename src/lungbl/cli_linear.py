import os, argparse, random, numpy as np
import pandas as pd
from lungbl.linear.model import Brock, Mayo, PLCO2012
from lungbl.utils.modeling import auc_from_df
import lungbl.definitions as definitions 
from lungbl.utils.statutils import bootstrap


MODELS = {
    'brock': Brock,
    'mayo': Mayo,
    'plco': PLCO2012,
}

COHORTS = {
    # Load cohorts as pandas dataframes
}

def export_pred(df, model, cohort, label):
    df = df[['pid', 'scandate', 'label', 'prob']]
    

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=MODELS.keys())
    parser.add_argument("cohort", choices=COHORTS.keys())
    parser.add_argument("--label", type=str, default='cancer_year1')
    parser.add_argument("--noimpute", default=True, action='store_false')
    args = parser.parse_args()
    
    random.seed(definitions.RANDOM_SEED)
    os.environ['PYTHONHASHSEED'] = str(definitions.RANDOM_SEED)
    np.random.seed(definitions.RANDOM_SEED)

    model = MODELS[args.model](label=args.label, impute=args.noimpute)
    cohort = COHORTS[args.cohort](args.label)
    if not args.noimpute:
        print(len(cohort))
        clinical_cols = ["age", "gender", "bmi", "emphysema", "phist", "fhist", "smo_status", "pkyr", 
            "nodule_size", "nodule_attenuation", "nodule_count"]
        iddf, nonull = cohort[cohort.columns.difference(clinical_cols)], cohort[clinical_cols]
        nonull = nonull[nonull[clinical_cols].notnull().all(axis=1)]
        cohort = iddf.merge(nonull, left_index=True, right_index=True)
    
    risk_df = model.predict(cohort)
    risk_col = f"{args.model}_risk"
    
    # export predictions to csv
    risk_df = risk_df.rename(columns={args.label: 'label', risk_col: 'prob'})
    if args.noimpute:
        dst = os.path.join(definitions.DATA_DIR, "linear", f"{args.model}_{args.cohort}_noimpute.csv")
    else:
        dst = os.path.join(definitions.DATA_DIR, "linear", f"{args.model}_{args.cohort}.csv")
    
    risk_df[['pid', 'scandate', 'label', 'prob']].to_csv(dst)

    # bootstrap CIs
    bsmetrics = bootstrap(
        risk_df,
        agg=lambda x: auc_from_df(x, cols=('label', 'prob')),
        )
    return bsmetrics
    
if __name__ == "__main__":
    bsmetrics = main()
    print(bsmetrics)