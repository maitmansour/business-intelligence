#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Import Libs
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import logging
logging.basicConfig(filename='logs/kmeans-unequilibrated-data.log',level=logging.DEBUG,format='%(asctime)s %(message)s')
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans

def fix_risque_data(risque_data):
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
data = pd.read_csv('data/base_prospect.csv', encoding='latin-1')

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
categorial_data['chgt_dir'].fillna(2, inplace=True)
categorial_data['type_com'].fillna("autre", inplace=True)

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

full_data = np.concatenate((numeric_data, categorial_data), axis=1)

logging.info('Kmeans method')
array_res =  {}
for k in range (1, 20):
  km_mdl = KMeans(n_clusters=k, random_state=1).fit(full_data)
  labels = km_mdl.labels_
  array_res[k] = km_mdl.inertia_
  logging.info('Iterate over Kmeans classes '+str(k)) 

logging.info('Save figure')
plt.figure()
plt.plot(list(array_res.keys()), list(array_res.values()))
plt.xlabel("Nombre des clusters")
plt.ylabel("RSQ")
plt.savefig('plot/KMEANS/clusters-unequilibrated-data.png')
plt.close()