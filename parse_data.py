from sklearn.decomposition import PCA
from sklearn.externals import joblib
from sklearn.preprocessing import *
from fancyimpute import MICE
import pandas as pd
import numpy as np

"""
Read csv file and return X and Y as numpy arrays
"""
def read_csv(filename):
    res = np.array(pd.read_csv(filename))
    
    return res[:, :-1], res[:, -1]

"""
Discard columns with too many NaN (set by nan_treshold) and constant columns 
(all zeroes)
"""
def discard_columns(x, nan_treshold=0.7):
    valid_columns = []

    for i in range(x.shape[1]):
        not_nan = np.count_nonzero(~np.isnan(x[:, i]))
        nan_perc = 1 - (not_nan/x.shape[0])
       
        if nan_perc >= nan_treshold:
            continue
        
        if np.all(x[:, i] == 0):
            continue

        valid_columns.append(i)

    x = x[:, valid_columns]

    return valid_columns

"""
Classify features as categorical, continuous or incomplete.
It's considered categorical if it has only 2 values (0 and 1);
Otherwise, continuous.
"""
def separate_features(x):
    categorical = []
    continuous = []
    incomplete = []

    for i in range(x.shape[1]):
        n = np.unique(x[:, i]).shape[0]
        not_nan = np.count_nonzero(~np.isnan(x[:, i]))
        nan_perc = 1 - (not_nan/x.shape[0])

        if n == 2:
            categorical.append(i)
        else:
            continuous.append(i)

        if not_nan < x.shape[0]:
            incomplete.append(i)

        
    return categorical, continuous, incomplete

"""
Check if incomplete columns are monotonic
"""
def check_for_monotonicity(x, incomplete):
    monotonic = []
    
    for inc in incomplete:
        xsorted = x[:, inc].copy()
        xsorted.sort()
        ind = np.argmax(np.isnan(xsorted))
        
        if np.all(np.isnan(xsorted[ind:])):
            monotonic.append(inc)

    return monotonic == incomplete

"""
Perform multiple imputation for NaN
"""
def perform_multiple_imputation(x, incomplete, n_imputations=100):
    mi = MICE(impute_type='col', n_imputations=n_imputations)
    x[:, incomplete] = mi.complete(x[:, incomplete])

"""
Standardize and bound data
"""
def normalize_data(x, continuous):
    mms = MinMaxScaler()
    x[:, continuous] = mms.fit_transform(scale(x[:, continuous]))

"""
Apply PCA to remove multicollinearity
"""
def reduce_components(x, continuous, categorical):
    dec = PCA(n_components=int(0.7*len(continuous)))
    new_x_continuous = dec.fit_transform(x[:, continuous])

    x = np.concatenate((new_x_continuous, x[:, continuous]), axis=1)
    return x

def apply_polynomial_interaction(x, columns, n=2):
    poly = PolynomialFeatures(n)
    other_cols = list(set(list(range(x.shape[1]))) - set(columns))
    transformed = poly.fit_transform(x[:, columns])
    
    if not other_cols:
        x = transformed
        return x

    x = np.concatenate((transformed, x[:, other_cols]), axis=1)
    return x

"""
Do the whole round of data preparation
"""
def parse_training(filename, nan_treshold=0.7, n_imputations=10, apply_pca=True, poly=1,
        valid_columns_file='valid_columns.pkl'):
    print('---> Reading csv...')
    x, y = read_csv(filename)
    
    print('---> Discarding columns...')
    valid_columns = discard_columns(x, nan_treshold)
    joblib.dump(valid_columns, valid_columns_file)
    x = x[:, valid_columns]

    print('---> Separating features...')
    categorical, continuous, incomplete = separate_features(x)

    print('---> Checking for monotonicity...')
    if check_for_monotonicity(x, incomplete):
        print('-----> Is monotonic!')
        print('-----> Peforming multiple imputation...')
        perform_multiple_imputation(x, incomplete, n_imputations)
    else:
        print('-----> Not monotonic. Skipping multiple imputation...')

    print('---> Normalizing data...')
    normalize_data(x, continuous)

    if apply_pca:
        print('---> Applying PCA...')
        x = reduce_components(x, continuous, categorical)

    if poly > 1:
        print('---> Applying polynomial interactions...')
        columns = continuous
        x = apply_polynomial_interaction(x, columns, poly)
        print('---> Normalizing again...')
        normalize_data(x, columns)

    print('!!! DONE !!!')

    return x, y

"""
Prepare testing data according to what was prepared on training
"""
def parse_testing(filename, valid_columns_file='valid_columns.pkl', n_imputations=10, 
        apply_pca=True, poly=1):
    x, y = read_csv(filename)
    valid_columns = joblib.load(valid_columns_file)
    x = x[:, valid_columns]

    categorical, continuous, incomplete = separate_features(x)
    if check_for_monotonicity(x, incomplete):
        perform_multiple_imputation(x, incomplete, n_imputations)

    normalize_data(x, continuous)

    if apply_pca:
        x = reduce_components(x, continuous, categorical)

    if poly > 1:
        columns = continuous
        x = apply_polynomial_interaction(x, columns, poly)
        normalize_data(x, columns)

    return x, y

if __name__ == '__main__':
    x, y = parse_training('data.csv', n_imputations=10, poly=1, apply_pca=True)
    x, y = parse_testing('data.csv', n_imputations=10, poly=1, apply_pca=True)
