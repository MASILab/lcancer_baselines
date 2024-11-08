import os
import pandas as pd
from datetime import datetime

def read_tabular(fpath, fargs={}):
    suffix = os.path.basename(fpath).split('.')[-1]
    if suffix == 'csv':
        return pd.read_csv(fpath, **fargs)
    elif suffix in ['xlsx', 'xlsm']:
        return pd.read_excel(fpath, **fargs, engine='openpyxl')
    elif suffix == 'xls':
        return pd.read_excel(fpath, **fargs)
    raise NotImplementedError(f"Unsupported file type: {fpath}")

def format_datetime_str(x, format="%Y%m%d") -> str:
    if pd.isnull(x):
        return None
    elif isinstance(x, str):
        dt = pd.to_datetime(x, errors='coerce')
    elif isinstance(x, float) or isinstance(x, int):
        dt = pd.to_datetime(str(int(x)), errors='coerce')
    elif isinstance(x, datetime):
        dt = x
    else:
        dt = pd.to_datetime(str(x), errors='coerce')
    
    if pd.notnull(dt): # check if NaT
        return dt.strftime(format)
    else:
        return None