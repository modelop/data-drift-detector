import pandas as pd
import numpy as np
from scipy.stats import ks_2samp
from scipy.stats import binom_test

#modelop.init
def begin():
    global train, numerical_features
    train = pd.read_csv('training_data.csv')
    numerical_features = train.select_dtypes(['int64', 'float64']).columns
    pass

#modelop.score
def action(datum):
    yield datum

#modelop.metrics
def metrics(data):
    ks_tests = [ks_2samp(train.loc[:, feat], data.loc[:, feat]) \
                for feat in numerical_features]
    pvalues = [x[1] for x in ks_tests]
    list_of_pval = [f"{feat}_p-value" for feat in numerical_features]
    ks_pvalues = dict(zip(list_of_pval, pvalues))
    
    yield ks_pvalues
