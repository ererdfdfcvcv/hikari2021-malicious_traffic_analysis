import torch
from network import *
from dataset import *
from torch.utils.data import DataLoader
from pathlib import Path
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
import numpy as np



model_path = Path('pytorch\\models\\20220520104430_10.model')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
### loading data
print("Loading data")
cstData = CustomDataLoader_testing_Labels()
train_dataloader = DataLoader(cstData, batch_size=64, shuffle=True)
print("Loading finnished")

#net = Net().double().to(device)
net = DenseNet_Label().double().to(device)
net.load_state_dict(torch.load(model_path))
net.eval()

pred_list = list()
Y_list = list()
with torch.no_grad():
    for i, data in enumerate(train_dataloader):
        X, Y = data
        Y_list.append(Y.tolist())
        X = X.to(device)
        pred = net.forward(X)

        pred = pred.detach().cpu()
        pred_list.append(pred.tolist())


ex_pred_list = [i for j in pred_list for i in j]
ex_Y_list = [i for j in Y_list for i in j]


ex_pred_list = np.array(ex_pred_list).squeeze().argmax(axis=1)
ex_Y_list = np.array(ex_Y_list).argmax(axis=1)



#traffic_category_replacement_matrix = ['Bruteforce-XML','Bruteforce','XMRIGCC CryptoMiner','Probing','Background','Benign']
print(classification_report(ex_Y_list, ex_pred_list))#, target_names=traffic_category_replacement_matrix))
print(accuracy_score(ex_Y_list, ex_pred_list))


