import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import pickle

# read dataset
df = pd.read_csv("ALLFLOWMETER_HIKARI2021.csv")
df  # debugging table plotting
bening_entries = df[df['traffic_category'] == "Benign"]
background_entries = df[df['traffic_category'] == "Background"]


benign_downsampling = resample(bening_entries, n_samples=20000, random_state=42)
background_downsampling = resample(background_entries, n_samples=20000, random_state=42)

dropped_df = df.drop(df.index[df['traffic_category'] == "Benign"], inplace = True)
dropped_df = df.drop(df.index[df['traffic_category'] == "Background"], inplace = True)

df = pd.concat([df, benign_downsampling, background_downsampling])
traffic_category_replacement_matrix = {
    'Bruteforce-XML': 0,
    'Bruteforce': 1,
    'XMRIGCC CryptoMiner': 2,
    'Probing': 3,
    'Background': 4,
    'Benign': 5
}
df.traffic_category = df.traffic_category.replace(traffic_category_replacement_matrix)

df = df.drop(columns=['Unnamed: 0.1', 'Unnamed: 0'])

# Extract labels and test, train split 
coder = OneHotEncoder(sparse=False)

labels = coder.fit_transform(df.traffic_category.to_numpy().reshape(len(df.traffic_category), 1))
#labels = df.traffic_category.to_numpy()

df_values_no_traffic = df.drop(columns=['uid', 'originh', 'originp', 'responh', 'responp','traffic_category', 'Label'])
df_values_no_traffic = df_values_no_traffic.astype(np.double)
x_train, x_test, y_train, y_test = train_test_split(df_values_no_traffic, labels, test_size=0.2)

training_data = [x_train, x_test, y_train, y_test]

with open("pytorch/training_data.pickle", 'wb') as handle:
    pickle.dump(training_data, handle)