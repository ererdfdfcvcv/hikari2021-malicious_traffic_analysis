import pickle
from sklearn.model_selection import train_test_split
import torch.optim as optim
import torch.nn as nn
import torch
from network import Net
from network import DenseNet
from network import DenseNet_Label
from dataset import CustomDataLoader
from dataset import CustomDataLoader_Labels
from torch.utils.data import DataLoader
from pathlib import Path
from datetime import datetime
import re

### options
EPOCHS = 100
model_fd = Path('pytorch\models')
loss_fd = Path('pytorch\loss')
now = re.sub("[ :.-]", "", str(datetime.now()).split('.')[0])
loss_list = list()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main():
    ### loading data
    print("Loading data")
    cstData = CustomDataLoader_Labels()
    train_dataloader = DataLoader(cstData, batch_size=64, shuffle=True)
    print("Loading finnished")

    #net = Net().double().to(device)
    #net = DenseNet_Label().double().to(device)
    net = DenseNet().double().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(EPOCHS):
        print(f"Now in Epoch {epoch}")
        track_loss = 0.0
        for i, data in enumerate(train_dataloader):
            X, Y = data
            X = X.to(device)
            Y = Y.to(device).type(torch.float)
            optimizer.zero_grad()

            out = net(X)
            out = out.squeeze()
            loss = criterion(out, Y)
            loss.backward()
            optimizer.step()

            track_loss += loss.item()

            if i % 200 == 0 and i != 0:
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {track_loss / 1000:.3f}')
                loss_list.append(track_loss)
                track_loss = 0.0


        model_fp = model_fd / (now + '_' + str(epoch+1) + '.model')
        loss_fp = loss_fd / (now + '.loss')
        loss_fp.touch(exist_ok=True)
        if (epoch + 1) % 5 == 0:
            torch.save(net.state_dict(), model_fp)
            with open(loss_fp, 'wb') as handle:
                pickle.dump(loss_list, handle)


if __name__ == "__main__":
    main()
