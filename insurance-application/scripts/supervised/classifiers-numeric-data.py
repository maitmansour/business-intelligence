#!/usr/bin/python
# -*- coding: utf-8 -*-
##############################################################
# Nom : Mohamed AIT MANSOUR	 / BELGHARBI Meryem				 #
# Source : M2 ILSEN - Avignon University					 #
##############################################################

# librairies import
import pandas as pd
import numpy as np
from sklearn import tree
from graphviz import render
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer
import logging
logging.basicConfig(filename='logs/classifiers-numeric-data.log',level=logging.DEBUG,format='%(asctime)s %(message)s')
from sklearn.dummy import DummyClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import LabelEncoder
from sklearn import svm
from sklearn.preprocessing import StandardScaler

dummycl = DummyClassifier(strategy="most_frequent")
gmb = GaussianNB()
dectree = tree.DecisionTreeClassifier()
logreg = LogisticRegression(solver="liblinear")
svc = svm.SVC(gamma='scale')

lst_classif = [dummycl, gmb, dectree, logreg, svc]
lst_classif_names = ['Dummy', 'Naive Bayes', 'Decision tree', 'Logistic regression']
#lst_classif_names = ['Dummy', 'Naive Bayes', 'Decision tree', 'Logistic regression', 'SVM']

def accuracy_score(lst_classif,lst_classif_names,X,y):
    for clf,name_clf in zip(lst_classif,lst_classif_names):
        scores = cross_val_score(clf, X, y, cv=5)
        print("Accuracy of "+name_clf+" classifier on cross-validation: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
        logging.info("Accuracy of "+name_clf+" classifier on cross-validation: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2)) 

def equilibrate_data(df):
    # Equilibrate rdv (0,1) data
    logging.info('Equilibrate rdv data') 
    rdv_1=df.loc[df['rdv'] == 1]
    rdv_0_raw=df.loc[df['rdv'] == 0]
    # Random sampling - Random n rows
    n=int(rdv_1.shape[0]*1)
    rdv_0 = rdv_0_raw.sample(n=n)
    # Get clean data with len(rdv0)=len(rdv1)
    data = pd.concat([rdv_1, rdv_0],ignore_index=True)
    return data

def fix_risque_data(risque_data):
    # Fix risque values (instead of 10-13, we will replace this value into mean([10,13]))
    logging.info('Fix risque values (instead of 10-13, we will replace this value into mean([10,13]))') 
    new_risque_values=[]
    for index, row in enumerate(risque_data):
        try:
         if "-" in row: 
            splited_risque=row.split('-')
            splited_risque=np.array(splited_risque).astype(np.float)
            mean_risque=np.mean(splited_risque, axis = 0)
            new_risque_values = np.append(new_risque_values, mean_risque)
         else:
            new_risque_values = np.append(new_risque_values, row)
        except TypeError:
         new_risque_values = np.append(new_risque_values, 7)
         continue # skips to next iteration
    return new_risque_values
    
# Read input text and put data inside a data frame
logging.info('Read data from file') 
data = pd.read_csv('data/base_prospect.csv', encoding='latin-1')
data=equilibrate_data(df=data)

# Prepare numeric data
numeric_attributes_names=["risque","effectif","ca_total_FL","ca_export_FK","evo_risque","age","evo_benefice","ratio_benef","evo_effectif"]
numeric_data=data[numeric_attributes_names]
risque_data=numeric_data['risque']
new_risque_values=fix_risque_data(risque_data=risque_data)
numeric_data['risque'] = new_risque_values

# Normlize numerique data
standard_scaler = StandardScaler();
standard_scaler.fit(numeric_data) 
numeric_data = standard_scaler.transform(numeric_data)

# Replace nan data (MEAN)
simple_imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
simple_imputer.fit(numeric_data)
numeric_data = simple_imputer.transform(numeric_data)

# Prepare RDV attribute
label_encoder = LabelEncoder()
Y = label_encoder.fit_transform(data["rdv"])

# Calcul et affichage des scores en appelant la methode accuracy_score fournit dans le tp
print('\n Accuracy Score for Numeric data : \n')
accuracy_score(lst_classif,lst_classif_names,numeric_data,Y)