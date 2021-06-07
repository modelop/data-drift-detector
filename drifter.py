# modelop.slot.0:in-use
# modelop.slot.1:in-use
# modelop.slot.2:in-use


import pandas as pd
import numpy as np
from scipy.stats import ks_2samp
from scipy.stats import binom_test

baseline=None
sample=None


#modelop.metrics
def metrics(sample, baseline):
    if sample is not None and baseline is not None:
        numerical_features = baseline.select_dtypes(['int64', 'float64']).columns
        ks_tests = [ks_2samp(baseline.loc[:, feat], sample.loc[:, feat]) \
                    for feat in numerical_features]
        pvalues = [x[1] for x in ks_tests]
        list_of_pval = [f"{feat}_p-value" for feat in numerical_features]
        ks_pvalues = dict(zip(list_of_pval, pvalues))
        ks_pvalues["pvalues"] = dict(zip(numerical_features, pvalues))
        yield ks_pvalues
    else: return
