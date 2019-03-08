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
from graphviz import render
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score

def replace_missing_value(df, number_features):
    imputer = Imputer(strategy="median")
    df_num = df[number_features]
    imputer.fit(df_num)
    X = imputer.transform(df_num)
    res_def = pd.DataFrame(X, columns=df_num.columns)
    return res_def


# Read input text and put data inside a data frame
data = pd.read_csv('../data/base_prospect.csv', encoding='latin-1')

# Equilibrate rdv (0,1) data
rdv_1=data.loc[data['rdv'] == 1]
rdv_0_raw=data.loc[data['rdv'] == 0]
# Random sampling - Random n rows
rdv_0 = rdv_0_raw.sample(n=rdv_1.shape[0])

# Clean data with len(rdv0)=len(rdv1)
data = pd.concat([rdv_1, rdv_0],ignore_index=True)

# Show data dead
print(data.head())

