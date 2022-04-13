#%%
# Loading, cleaning up and converting 
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.decomposition import PCA
import plotly.express as px
import matplotlib.pyplot as plt

fp = "remaining_behavior_ext.csv"

DF = pd.read_csv(fp)

DF = DF.drop(['Unnamed: 0', '_id', 'behavior', 'source'], axis=1)

DF.ip_type = DF.ip_type.map({'default':0, 'private_ip':1, 'datacenter':2, 'google_bot':3})
DF.behavior_type = DF.behavior_type.map({'outlier':0, 'normal':1, 'bot':2, 'attack':3})
labels = DF.behavior_type
DF = DF.drop('behavior_type', axis=1)

DF = DF.astype(np.float32)
#print(DF.columns.values.tolist())
#print(DF['ip_type'].unique())
#print(DF.head())
#print(DF.describe())
DF = DF.fillna(0)
# %%
# Scale everything with standard scaler

scaler = preprocessing.StandardScaler().fit(DF)

df_scaled = pd.DataFrame(scaler.transform(DF))

print(df_scaled.describe())
print(df_scaled.isnull().sum())
#%%
# Do PCA
pca = PCA(n_components=3)
x = pca.fit_transform(df_scaled)

print(df_scaled[:2])

# %%
# Plot PCA

total_var = pca.explained_variance_ratio_.sum() * 100
fig = px.scatter_3d(
    x, x=0, y=1, z=2, color=(labels),
	title=f'Total Explained Variance: {total_var:.2f}%',
    labels={'0': 'PC 1', '1': 'PC 2', '2': 'PC 3', 'color': 'Labels'}
)
fig.update_traces(marker=dict(size=1))
fig.write_html("pca.html")
# %%
