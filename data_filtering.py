import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import cross_validation, datasets, preprocessing, svm, linear_model
from sklearn.grid_search import GridSearchCV
import re
import os
import string


def NumericalizeData(X, attrTypes, doNorm=True, enc_nom=None, enc_num=None, removeMiss=False, doImpute=True):
    cols = X.shape[1]
    if (cols != len(attrTypes)):
        print "mismatch in attrTypes and number of attributes in X"
        raise
    rows = X.shape[0]

    # create empty vectors
    x_numeric = np.array([])
    x_nominal = np.array([])
    for i in range(1, rows):
        x_numeric = np.vstack((x_numeric, np.array([])))
        x_nominal = np.vstack((x_nominal, np.array([])))

    # separate nominal and numeric attributes for further processing
    for i in range(len(attrTypes)):
        if attrTypes[i] is 'c':
            x_nominal = np.hstack((x_nominal, X[:, [i]]))
        elif attrTypes[i] is 'n':
            x_numeric = np.hstack((x_numeric, X[:, [i]]))
        else:
            print("Value error. Check {0}".format(attrTypes[i]))
            raise
    if doImpute is True:
        print "Imputing dataset..."
        nomDF = pd.DataFrame(x_nominal)  # Convert np.array to pd.DataFrame
        nomDF = nomDF.apply(
            lambda x: x.fillna(x.value_counts().index[0]))  # replace NaN with most frequent in each column
        x_nominal = nomDF.values  # convert back pd.DataFrame to np.array
        imp = preprocessing.Imputer(strategy='mean')
        x_numeric = imp.fit_transform(x_numeric)

    elif removeMiss is True:
        print "Removing instances having missing values..."
        isNomMissing = np.array([np.isnan(x_nominal[row, :]).any() for row in range(x_nominal.shape[0])])
        x_nominal = x_nominal[~isNomMissing, :]
        isNumMissing = np.array([np.isnan(x_numeric[row, :]).any() for row in range(x_numeric.shape[0])])
        x_numeric = x_numeric[~isNumMissing, :]

    # OneHotEncode nominal
    if enc_nom is None:
        enc_nom = preprocessing.MultiLabelBinarizer()
        if x_nominal.shape[1] is not 0:
            x_nominal = enc_nom.fit_transform(x_nominal)
    else:
        if x_nominal.shape[1] is not 0:
            x_nominal = enc_nom.transform(x_nominal)
    # normalize numerics
    if doNorm is True:
        if enc_num is None:
            enc_num = preprocessing.StandardScaler()
            if x_numeric.shape[1] is not 0:
                x_numeric = enc_num.fit_transform(x_numeric)
        else:
            if x_numeric.shape[1] is not 0:
                x_numeric = enc_num.transform(x_numeric)

    X_mod = np.hstack((x_nominal, x_numeric))
    return (X_mod, enc_nom, enc_num)