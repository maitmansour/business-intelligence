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
import seaborn as sns
from sklearn import tree
from graphviz import render
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer
import logging
logging.basicConfig(filename='../logs/decision-tree.log',level=logging.DEBUG,format='%(asctime)s %(message)s')

def replace_missing_value(df, number_features):
    logging.info('Replace missing values') 
    imputer = SimpleImputer(strategy="median")
    df_num = df[number_features]
    imputer.fit(df_num)
    X = imputer.transform(df_num)
    res_def = pd.DataFrame(X, columns=df_num.columns)
    return res_def


# Read input text and put data inside a data frame
logging.info('Read data from file') 
data = pd.read_csv('../data/base_prospect.csv', encoding='latin-1')

# Equilibrate rdv (0,1) data
logging.info('Equilibrate rdv data') 
rdv_1=data.loc[data['rdv'] == 1]
rdv_0_raw=data.loc[data['rdv'] == 0]

# Random sampling - Random n rows
rdv_0 = rdv_0_raw.sample(n=rdv_1.shape[0])

# Get clean data with len(rdv0)=len(rdv1)
data = pd.concat([rdv_1, rdv_0],ignore_index=True)

# Show data dead
print(data.head())

# Replace missing values
data=replace_missing_value(data,["effectif","ca_total_FL","ca_export_FK","evo_risque","age","chgt_dir","rdv"])

# Prepare classifier data
logging.info('Prepare classifier data') 
feature_names=["effectif","ca_total_FL","ca_export_FK","evo_risque","age","chgt_dir"]
X=data[feature_names]
Y=data.rdv 

# Separate the data into train and test set
logging.info('Separate the data into train and test set') 
X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size = 0.3)

# Fill NA values with MEAN
X_test.fillna(X_train.mean(), inplace=True)

# Building and fitting a classification tree
logging.info('Building and fitting a classification tree') 
tree_model = tree.DecisionTreeClassifier(max_depth=7)
tree_model.fit(X_train, y_train)

# Save the decision tree.
logging.info('Save the decision tree') 
tree.export_graphviz(tree_model,out_file="../plot/decision-tree.gv") 
render('dot', 'png', "../plot/decision-tree.gv")  