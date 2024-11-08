import os
from sklearn import metrics

def binary_prob_output(output):
    """ 
    accepts torch.tensor output from a binary classifier and returns the prob of the positive class.
    first dim must be 1
    """
    assert output.shape in [(1, 2), (1, 1), (2,), (1,)], f"output shape must be (1, 2), (1, 1), (2,), or (1,), but got {output.shape}"
    if output.shape == (1, 2):
        return output[0][1]
    if output.shape == (1, 1):
        return output[0][0]
    if output.shape == (2,):
        return output[1]
    if output.shape == (1,):
        return output[0]
    
def auc_from_df(df, cols=('label', 'prob')):
    y_col, yhat_col = cols
    return metrics.roc_auc_score(df[y_col], df[yhat_col])

def ppv_from_df(df, cols=('label', 'prob'), threshold=0.1):
    y_col, yhat_col = cols
    yhats = (df[yhat_col] >= threshold).astype(int)
    return metrics.precision_score(df[y_col], yhats)

def recall_from_df(df, cols=('label', 'prob'), threshold=0.1):
    y_col, yhat_col = cols
    yhats = (df[yhat_col] >= threshold).astype(int)
    return metrics.recall_score(df[y_col], yhats)