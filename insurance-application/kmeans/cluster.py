from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

import pylab as pl
from sklearn.decomposition import PCA


variables = pd.read_csv('../data/base_prospect.csv')
X = variables[['rdv']]
Y = variables[['ca_total_FL']]
Nc = range(1, 20)
kmeans = [KMeans(n_clusters=i) for i in Nc]
kmeans

score = [kmeans[i].fit(Y).score(Y) for i in range(len(kmeans))]
score
'''
pl.plot(Nc,score)
pl.xlabel('Number of Clusters')
pl.ylabel('Score')
pl.show()'''

pca = PCA(n_components=1).fit(Y)
pca_d = pca.transform(Y)
pca_c = pca.transform(X)
kmeans=KMeans(n_clusters=5)
kmeansoutput=kmeans.fit(Y)
kmeansoutput
pl.figure('5 Cluster K-Means')
pl.scatter(pca_c[:, 0], pca_d[:, 0], c=kmeansoutput.labels_)
pl.xlabel('Dividend Yield')
pl.ylabel('Returns')
pl.title('5 Cluster K-Means')
pl.show()

