#!/usr/bin/python
# -*- coding: utf-8 -*-
##############################################################
# Nom : Mohamed AIT MANSOUR  / BELGHARBI Meryem              #
# Source : M2 ILSEN - Avignon University                     #
##############################################################

#librairies import
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
import logging
logging.basicConfig(filename='logs/analyse-acp.log',level=logging.DEBUG,format='%(asctime)s %(message)s')
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

def equilibrate_data(df):
    logging.info('Equilibrate rdv (0,1) data')
    logging.info('Equilibrate rdv data') 
    rdv_1=df.loc[df['rdv'] == 1] 
    rdv_0_raw=df.loc[df['rdv'] == 0]
    logging.info('Random sampling - Random n rows')
    rdv_0 = rdv_0_raw.sample(n=rdv_1.shape[0])
    logging.info('Get clean data with len(rdv0)=len(rdv1)')
    data = pd.concat([rdv_1, rdv_0],ignore_index=True)
    return data

def fix_risque_data(risque_data):
    #Fix risque values (instead of 10-13, we will replace this value into mean([10,13]))
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
         continue
    return new_risque_values
    
logging.info('Read input text and put data inside a data frame')
logging.info('Read data from file') 
data = pd.read_csv('data/base_prospect.csv', encoding='latin-1')
data=equilibrate_data(df=data)

logging.info('Prepare numeric data')
numeric_attributes_names=["risque","effectif","ca_total_FL","ca_export_FK","evo_risque","age","evo_benefice","ratio_benef","evo_effectif"]
numeric_data=data[numeric_attributes_names]
risque_data=numeric_data['risque']
new_risque_values=fix_risque_data(risque_data=risque_data)
numeric_data['risque'] = new_risque_values

logging.info('Normlize numerique data')
standard_scaler = StandardScaler();
standard_scaler.fit(numeric_data) 
numeric_data = standard_scaler.transform(numeric_data)

logging.info('Replace nan data (MEAN)')
simple_imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
simple_imputer.fit(numeric_data)
numeric_data = simple_imputer.transform(numeric_data)

logging.info('Prepare categorial data')
categorial_attributes_names=["chgt_dir","type_com","activite","actionnaire","forme_jur_simpl"]
categorial_data=data[categorial_attributes_names]

logging.info('Replace none values with most frequent values')
simple_imputer= SimpleImputer(missing_values="'none'", strategy='most_frequent')
categorial_data = simple_imputer.fit_transform(categorial_data)

logging.info('Replace ? values with most frequent values')
simple_imputer = SimpleImputer(missing_values='?', strategy='most_frequent')
categorial_data = simple_imputer.fit_transform(categorial_data)

logging.info('Data descritization')
categorial_data = pd.DataFrame(categorial_data, columns=categorial_attributes_names);
categorial_data = pd.get_dummies(categorial_data)

logging.info('Prepare RDV attribute')
label_encoder = LabelEncoder()
Y = label_encoder.fit_transform(data["rdv"])

cleaned_data = np.concatenate((numeric_data, categorial_data), axis=1)


logging.info('Get Principal Components')
acp = PCA(n_components=4)
principal_components=acp.fit_transform(cleaned_data)
print(principal_components)

logging.info('Save the principal Components')
y=['Principal Component 1', 'Principal Component 2', 'Principal Component 3', 'Principal Component 4']
acpDf = pd.DataFrame(data = principal_components, columns =y )
finalDf = pd.concat([acpDf, data[['rdv']]], axis = 1)
Df=acpDf.astype(float)

logging.info('Save Principal Components')
g=sns.lmplot("Principal Component 1","Principal Component 2",hue='rdv',data=finalDf,fit_reg=False,scatter=True,size=7)
plt.savefig('plot/PCA/principal_component_1_and2.png')

logging.info('Save Principal Components')
g=sns.lmplot("Principal Component 3","Principal Component 4",hue='rdv',data=finalDf,fit_reg=False,scatter=True,size=7)
plt.savefig('plot/PCA/principal_component_3_and4.png')