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


# Get Principal Components
acp = PCA(n_components=4)
principal_components=acp.fit_transform(cleaned_data)
print(principal_components)


y=['Principal Component 1', 'Principal Component 2', 'Principal Component 3', 'Principal Component 4']
acpDf = pd.DataFrame(data = principal_components, columns =y )
finalDf = pd.concat([acpDf, data[['rdv']]], axis = 1)
Df=acpDf.astype(float)

# Save Principal Components
g=sns.lmplot("Principal Component 1","Principal Component 2",hue='rdv',data=finalDf,fit_reg=False,scatter=True,size=7)
plt.savefig('../plot/principal_component_1_and2.png')

# Save Principal Components
g=sns.lmplot("Principal Component 3","Principal Component 4",hue='rdv',data=finalDf,fit_reg=False,scatter=True,size=7)
plt.savefig('../plot/principal_component_3_and4.png')