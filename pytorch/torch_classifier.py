import pandas as pd
from sklearn.model_selection import train_test_split
import torch.optim as optim
import torch.nn as nn
import torch
from network import net
from dataset import CustomDataLoader
from torch.utils.data import DataLoader
from pathlib import Path
from datetime import datetime
import re

### options

EPOCHS = 10
save_fd = Path('pytorch\models')
now = re.sub("[ :.-]", "", str(datetime.now()))

### loading data
print("Loading data")
cstData = CustomDataLoader()
train_dataloader = DataLoader(cstData, batch_size=64, shuffle=True)

x_test, y_test = cstData.returnTestSet() 
print("Loading finnished")


criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

for epoch in range(EPOCHS):
    print(f"Now in Epoch {epoch}")
    track_loss = 0.0
    for i, data in enumerate(train_dataloader):
        X, Y = data

        optimizer.zero_grad()

        out = net(X)

        loss = criterion(out, Y)
        loss.backward()
        optimizer.step()

        track_loss += loss.item()

        if i % 1000 == 0:
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {track_loss / 1000:.3f}')
            track_loss = 0.0
    save_fp = save_fd / now + '_' + epoch + '.model'
    torch.save(net.state_dict(), save_fp)

