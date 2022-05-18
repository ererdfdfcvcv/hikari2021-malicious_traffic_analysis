import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin1 = nn.Linear(79, 200)
        self.lin2 = nn.Linear(200, 70)
        self.lin3 = nn.Linear(70, 6)
        self.bn2 = nn.BatchNorm1d(1) # 200
        self.bn3 = nn.BatchNorm1d(1) # 70
        self.emb_drop = nn.Dropout(0.6)
        self.drops = nn.Dropout(0.3)
        

    def forward(self, x):
        x = F.relu(self.lin1(x))
        x = self.drops(x)
        x = self.bn2(x)
        x = F.relu(self.lin2(x))
        x = self.drops(x)
        x = self.bn3(x)
        x = self.lin3(x)
        return x


net = Net().double()