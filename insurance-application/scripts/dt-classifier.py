#!/usr/bin/python
# -*- coding: utf-8 -*-
##############################################################
# Nom : Mohamed AIT MANSOUR	 / BELGHARBI Meryem				 #
# Source : M2 ILSEN - Avignon University					 #
##############################################################

# librairies import
import pandas as pd
from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.preprocessing import Imputer
import seaborn as sns
from sklearn import tree

# read input text and put data inside a data frame
data = pd.read_csv('../data/base_prospect.csv', encoding='latin-1')

# data head
print("\nData Head \n")
print( data .head())

# We have to standrize our data, because there is a very large defirence between data (eg. 0.3 and -5 at tec00115 AT and CY)
scaler = StandardScaler()
scaled_X_norm = data.copy()

col_names = ["effectif","ca_total_FL","ca_export_FK","evo_risque","age","chgt_dir","rdv"]
X_norm = scaled_X_norm[col_names]
scaler = StandardScaler().fit(X_norm.values)
X_norm = scaler.transform(X_norm.values)




print("\nData Head Standrized\n")
scaled_X_norm[col_names] = X_norm
print(scaled_X_norm)

imp = Imputer(missing_values='NaN', strategy='mean', axis=1)
cleaned_data = imp.fit_transform(X_norm)


# Get Decisional Tree Classifier

