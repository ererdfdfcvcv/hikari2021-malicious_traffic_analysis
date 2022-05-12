import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
import plotly.express as px
import matplotlib.pyplot as plt

# read dataset
dt = pd.read_csv("ALLFLOWMETER_HIKARI2021.csv")
dt  # debugging table plotting

# 0. Step. Get the behavior distribution
traffic_category_list = dt['traffic_category']
counter = {}

for elem in traffic_category_list:
    if elem not in counter:
        counter[elem] = 1
    else:
        counter[elem] += 1

# print(counter)

# 1. Step - Replace the categorization labels (traffic_category) with numeric values

traffic_category_replacement_matrix = {
    'Bruteforce-XML': 0,
    'Bruteforce': 1,
    'XMRIGCC CryptoMiner': 2,
    'Probing': 3,
    'Background': 4,
    'Benign': 5
}
# new dataset with replaced 'traffic_category' value
dt.traffic_category = dt.traffic_category.replace(traffic_category_replacement_matrix)
labels = dt.traffic_category
# dt  # debugging table plotting

# 2. Stop - Drop the unnecessary features

dt = dt.drop(columns=['Unnamed: 0.1', 'Unnamed: 0'])
# dt  # debugging table plotting

# 3. Step - Split the tables into 2 tables
# This step should separate the IP and its identifiers from the data, which is used to be analyze a webserver


dt_identification = dt[['uid', 'originh', 'originp', 'responh', 'responp']].copy()
dt_values = dt.drop(columns=['uid', 'originh', 'originp', 'responh', 'responp'])
# dt_identification  # debugging table plotting
# dt_values  # debugging table plotting

# 4. Step - Scaling and Normalization of the data

scaler = preprocessing.StandardScaler().fit(dt_values)
dt_scaled = pd.DataFrame(scaler.transform(dt_values))
dt_normalized_l2 = pd.DataFrame(preprocessing.normalize(dt_scaled, norm='l2'))
# dt_normalized_l2

# 5. Step - PCA Analysis with together with a plot of the values

pca = PCA(n_components=3)
dt_pca = pca.fit_transform(dt_normalized_l2)
dt_pca

total_var = pca.explained_variance_ratio_.sum() * 100
fig = px.scatter_3d(
    dt_pca, x=0, y=1, z=2, color=(labels),
    title=f'Total Explained Variance: {total_var:.2f}%',
    labels={'0': 'PC 1', '1': 'PC 2', '2': 'PC 3', 'color': 'Labels'}
)
fig.update_traces(marker=dict(size=1))
fig.write_html("url_analysis_normalized_pca.html")

# 6. Step: Clustering of the dataset together with a plotting of the Clustering results

dbscan = DBSCAN(eps=0.35, min_samples=10)
dt_clustered = dbscan.fit_predict(dt_normalized_l2)
dt_clustered = pd.DataFrame(dt_clustered)
dt_clustered

total_var = pca.explained_variance_ratio_.sum() * 100
fig = px.scatter_3d(
    dt_clustered, x=0, y=1, z=2, color=(labels),
    title=f'Total Explained Variance: {total_var:.2f}%',
    labels={'0': 'PC 1', '1': 'PC 2', '2': 'PC 3', 'color': 'Labels'}
)
fig.update_traces(marker=dict(size=1))
fig.write_html("url_analysis_dbscan.html")
