from turtle import forward
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


class DenseNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(1,20, 3, stride=2)
        self.conv2 = nn.Conv1d(20, 50, kernel_size=5)
        self.conv3 = nn.Conv1d(50, 20, kernel_size=3)
        self.bn1 = nn.BatchNorm1d(20)
        self.bn2 = nn.BatchNorm1d(50)
        self.drop1 = nn.Dropout(0.6)
        self.drop2 = nn.Dropout(0.3)
        self.flat1 = nn.Flatten(1,2)
        self.lin1 = nn.Linear(660, 6)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.drop1(x)
        x = self.bn1(x)
        x = F.relu(self.conv2(x))
        x = self.drop2(x)
        x = self.bn2(x)
        x = self.conv3(x)
        x = self.flat1(x)
        x = self.lin1(x)
        return x

class DenseNet_Label(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(1,20, 3, stride=2)
        self.conv2 = nn.Conv1d(20, 50, kernel_size=5)
        self.conv25 = nn.Conv1d(50, 100, kernel_size=3)
        self.conv3 = nn.Conv1d(100, 20, kernel_size=3)
        self.bn1 = nn.BatchNorm1d(20)
        self.bn2 = nn.BatchNorm1d(50)
        self.drop1 = nn.Dropout(0.3) # lowered dropout
        self.drop2 = nn.Dropout(0.3)
        self.flat1 = nn.Flatten(1,2)
        self.lin1 = nn.Linear(620, 2)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.drop1(x)
        x = self.bn1(x)
        x = F.relu(self.conv2(x))
        x = self.drop2(x)
        x = self.bn2(x)
        x = self.conv25(x)
        x = self.conv3(x)
        x = self.flat1(x)
        x = self.lin1(x)
        return x