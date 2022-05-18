import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import pickle

dataset_path = "remaining_behavior_ext.csv"
show_tables = True  # debugging option to control the displaying of the new datasets/tables
dt = pd.read_csv(dataset_path)
ip_type_values = list(dt['ip_type'].unique())
ip_type_replacement_matrix = {
    'default': 0,
    'private_ip': 1,
    'datacenter': 2,
    'google_bot': 3
}
# new dataset with replaced 'ip_type' value
dt.ip_type = dt.ip_type.replace(ip_type_replacement_matrix)

ip_type_values = list(dt['behavior_type'].unique())
behavior_type_replacement_matrix = {
    'outlier': 0,
    'normal': 1,
    'bot': 2,
    'attack': 3
}
# new dataset with replaced 'behavior_type' value
dt.behavior_type = dt.behavior_type.replace(behavior_type_replacement_matrix)
labels = dt.behavior_type
dt = dt.drop(columns=['Unnamed: 0', '_id', 'source', 'behavior'])
scaler = preprocessing.StandardScaler().fit(dt)
dt_scaled = pd.DataFrame(scaler.transform(dt))
dt_scaled = pd.DataFrame(dt_scaled).fillna(0)
dt_normalized = preprocessing.normalize(dt_scaled, norm='l2')

dt_normalized_l2 = pd.DataFrame(preprocessing.normalize(dt_scaled, norm='l2'))

X_train, X_test, Y_train, Y_test = train_test_split(dt_normalized_l2, labels, test_size=0.33)

clf = SVC(gamma='auto')

print('now training')
clf.fit(X_train, Y_train)
print('training finnished')

pred_labels = clf.predict(X_test)
target_names = [
    'outlier',
    'normal',
    'bot',
    'attack'
]
print(classification_report(Y_test, pred_labels, target_names=target_names))

with open('modelweight.pickle', 'wb') as handle:
    pickle.dump(clf, handle)
