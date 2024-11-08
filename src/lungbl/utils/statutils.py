import pandas as pd
import numpy as np
import scipy.stats as stats
import math
import itertools

def z_test(r, n, c=0.95):
    """
    Preferred method of estimating proportions and CI from [Altman et al. Statistics With Confidence, 2ed]
    Handles CI for very small proportions, bounding between [0,1]. Without this, CI might be negative or above 1 for rare events.
    Can compute sample size needed to bound a rare event with a CI.
    Parameters
    ---------- 
    r: int. numerator ()
    n: int. denominator (sample size)
    c: float. confidence
    Returns
    ---------- 
    point estimate, lower CI, upper CI: float
    """
    q = 1-r/n
    z = stats.norm.ppf(1-(1-c)/2)
    A = 2*r+z^2
    B = z*math.sqrt(z^2+4*r*q)
    C = 2*(n+z^2)
    return r/n, (A-B)/C, (A+B)/C

def is_nested_list(l):
    try:
        next(x for x in l if isinstance(x, (list, tuple)))
        return True
    except StopIteration:
        return False

def ci(data, confidence=0.95):
    d = 1.0*np.array(data)
    n = len(d)
    mu, std = np.mean(d), np.std(d)
    z = stats.norm.ppf(1-(1-confidence)/2)
    me = z*std/math.sqrt(n)
    return mu, mu-me, mu+me

def bootstrap_samples(df, agg, grps=[], n=100):
    """
    Parameters
    ----------
    df: pandas.DataFrame. metrics in long format
    agg: func. method of computing the aggregate metric (i.e. mean, AUC, F1score, etc.)
    *grps: str or list. name of col to group by
    n: int. number of bootstrap samples
    Returns
    ----------
    bstrap: pandas.DataFrame. n rows with agg metric for each bootstrap sample. cols [grp0, grp1, ..., agg_0, agg_1, ...]
    """
    grps = grps if isinstance(grps, list) else [grps]
    grpnames = [df[g].unique().tolist() for g in grps]
    grpcomb = itertools.product(*grpnames) # all combinations of the groups
    bstrap_grps = []

    for comb in grpcomb:
        if comb: # if grp exists
            query = ' & '.join(f"{g}=={c}" for g, c in zip(grps, comb)) # str with 'col1==grp1 & col2==grp2 & ...'
            dfgrp = df.query(query) # get the group
        else:
            dfgrp = df

        # compute aggregate metric on n samples
        metrics = []
        for i in range(n):
            sample = dfgrp.sample(frac=1.0, replace=True)
            metrics.append(agg(sample))
        
        if is_nested_list(metrics):
            bstrap_grp = pd.DataFrame(metrics, columns=[f"agg_{i}" for i in metrics[0]]) # all samples for this group
        else:
            bstrap_grp = pd.DataFrame(metrics, columns=["agg"])
        if comb:
            for g, c in zip(grps, comb):
                bstrap_grp[g] = c # set col g to value c
        bstrap_grps.append(bstrap_grp)
    
    return pd.concat(bstrap_grps)

def bootstrap(df, agg, grps=[], n=100, confidence=0.95):
    """
    Compute mean and CI from n bootstrap samples, sampling with replacement.
    Per central limit theorem, sample means are normally distributed.

    Parameters
    ----------
    df: pandas.DataFrame. metrics in long format
    agg: func. method of computing the aggregate metric (i.e. mean, AUC, F1score, etc.)
    *grps: str or list. name of col to group by
    n: int. number of bootstrap samples
    confidence: float.

    Returns
    ----------
    bstrap: pandas.DataFrame. mean metrics and CI grouped by grps
    """
    grps = grps if isinstance(grps, list) else [grps]
    grpnames = [df[g].unique().tolist() for g in grps]
    grpcomb = itertools.product(*grpnames) # all combinations of the groups
    resultrows = []

    for comb in grpcomb:
        if comb: # if grp exists
            query = ' & '.join(f"{g}=={c}" for g, c in zip(grps, comb)) # str with 'col1==grp1 & col2==grp2 & ...'
            dfgrp = df.query(query) # get the group
        else:
            dfgrp = df

        # compute aggregate metric on n samples
        metrics = []
        for i in range(n):
            sample = dfgrp.sample(frac=1.0, replace=True)
            metrics.append(agg(sample))
        
        metrics = list(zip(*metrics)) if is_nested_list(metrics) else [metrics]

        # get mean and CIs for each metric type
        cis = [] # [(mu1, lci_1, uci_1), (mu2, lci_2, uci_2), ...]
        for m in metrics:
            cis.append(ci(m, confidence=confidence))
        cidict = [{f'mean_{i}': c[0], f'lci_{i}': c[1], f'uci_{i}': c[2]} for i, c in enumerate(cis)]
        cidict = {k: v for dict in cidict for k, v in dict.items()}
        grpdict = {k[0]: k[1] for k in zip(grps, comb)} if comb else {}
        resultrows.append({**grpdict, **cidict})
    
    return pd.DataFrame(resultrows)
