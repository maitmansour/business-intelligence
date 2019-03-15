#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from scipy. cluster .hierarchy import dendrogram, linkage , fcluster
from sklearn.impute import SimpleImputer
import logging
logging.basicConfig(filename='logs/kmeans.log',level=logging.DEBUG,format='%(asctime)s %(message)s')
from sklearn.preprocessing import LabelEncoder
import scipy.cluster.hierarchy as sch
from sklearn import cluster
from mpl_toolkits.mplot3d import Axes3D

def replace_missing_value(df, number_features):
    logging.info('Replace missing values') 
    imputer = SimpleImputer(strategy="median")
    df_num = df[number_features]
    imputer.fit(df_num)
    X = imputer.transform(df_num)
    res_def = pd.DataFrame(X, columns=df_num.columns)
    return res_def


def fix_risque_data(risque_data):
    # Fix risque values (instead of 10-13, we will replace this value into mean([10,13]))
    logging.info('Fix risque values (instead of 10-13, we will replace this value into mean([10,13]))') 
    new_risque_values={}
    for index, row in enumerate(risque_data):
        try:
         if "-" in row: 
            splited_risque=row.split('-')
            splited_risque=np.array(splited_risque).astype(np.float)
            mean_risque=np.mean(splited_risque, axis = 0)
            new_risque_values[index]=mean_risque
         else:
            new_risque_values[index]=row
        except TypeError:
         new_risque_values[index]=np.nan
         continue # skips to next iteration
    return new_risque_values


# Read file
data = pd.read_csv('../data/base_prospect.csv')
# print(data.head())


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

# Prepare categorial data
categorial_attributes_names=["type_com","activite","actionnaire","forme_jur_simpl"]
categorial_data=data[categorial_attributes_names]

# Replace none values with most frequent values
simple_imputer= SimpleImputer(missing_values="'none'", strategy='most_frequent')
categorial_data = simple_imputer.fit_transform(categorial_data)

# Replace ? values with most frequent values
simple_imputer = SimpleImputer(missing_values='?', strategy='most_frequent')
categorial_data = simple_imputer.fit_transform(categorial_data)

# Data descritization
categorial_data = pd.DataFrame(categorial_data, columns=categorial_attributes_names);
categorial_data = pd.get_dummies(categorial_data)

# Merge numeric data and categorial data
full_data = np.concatenate((numeric_data, categorial_data), axis=1)
print (full_data)


#kmeans method
array_res =  {}
for kmeans in range (1, 20):
  km_mdl = KMeans(n_clusters=kmeans, random_state=1).fit(full_data)
  # df.loc[row_indexer,column_indexer]
  labels = km_mdl.labels_
  array_res[kmeans] = km_mdl.inertia_
  #print (kmeans)


plt.figure()
plt.plot(list(array_res.keys()), list(array_res.values()))
plt.xlabel("Nombre des clusters")
plt.ylabel("RSQ")
plt.savefig('../plot/clusters.png')
plt.close()


kmeanss = KMeans(n_clusters=8, random_state=111)
kmeanss.fit(full_data)
pca = PCA(n_components=3).fit(full_data)
pca_2d = pca.transform(full_data)
fig = plt.figure('K-means with 8 clusters')
ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)
plt.scatter(pca_2d[:, 0], pca_2d[:, 1], c=kmeanss.labels_)
plt.savefig('../plot/classes.png')
print (pca_2d)

#correlation_circle
def correlation_circle(df,nb_var,x_axis,y_axis):
    fig, axes = plt.subplots(figsize=(8,8))
    axes.set_xlim(-1,1)
    axes.set_ylim(-1,1)
    # label with variable names
    for j in range(nb_var):
        # ignore two first columns of df: Nom and Code^Z
        plt.annotate(df.columns[j+2],(corvar[j,x_axis],corvar[j,y_axis]))
    # axes
    plt.plot([-1,1],[0,0],color='silver',linestyle='-',linewidth=1)
    plt.plot([0,0],[-1,1],color='silver',linestyle='-',linewidth=1)
    # add a circle
    cercle = plt.Circle((0,0),1,color='blue',fill=False)
    axes.add_artist(cercle)
    plt.savefig('../plot/acp_correlation_circle_axes_'+str(x_axis)+'_'+str(y_axis))
    plt.close(fig)

#la variance
vrc= pca.explained_variance_ratio_
square = np.sqrt(4/(4*vrc))
corvar = np.zeros((37,37))
for k in range(4):
    corvar[:,k] = pca.components_[k,:] * square[k]
print(corvar)

correlation_circle(df=full_data, nb_var=4, x_axis=0, y_axis=1)

'''
# bining numeric data
res_values = np.random.random(100)
bins = np.linspace(0, 1, 10)
digitized = np.digitize(res_values, bins)
bin_means = [res_values[digitized == i].mean() for i in range(1, len(bins))]
print (bin_means)'''

'''
# Normalize numerique data
standard_scaler = StandardScaler();
standard_scaler.fit(bin_means) 
bin_means = standard_scaler.transform(bin_means)'''

# print(res_values)

'''
# méthode hiérarchique ascendante
link = linkage(full_data, method='ward')
plt.figure(figsize=(25,18))
dendrogram(link,color_threshold=0,labels=data.index, no_labels=True)
plt.title ("Hierarchical Clustering Dendrogram (Ward)")
plt.xlabel("entreprises")
plt.ylabel("distance")
plt.savefig('../plot/classification-ascendante-hiérarchique.png')

groupes_cah = sch.fcluster(link,t=100,criterion='distance') 
# print(np.unique(g roupes_cah).size, "groupes constitues")

# idg = np.argsort(groupes_cah) 
print(np.unique(groupes_cah).size, "groupes constitués")'''

'''
# Affichage des groupes
idg = np.argsort(groupes_cah) 
df = pd.DataFrame(pd.concat([f_instances1.nom[idg],f_instances1.date[idg]],ignore_index=True),groupes_cah[idg]) 
gdf=f_instances1.iloc[idg]
gdf['groupe']=groupes_cah[idg]
gdf.groupby(['groupe','nom'])['id'].count()'''
