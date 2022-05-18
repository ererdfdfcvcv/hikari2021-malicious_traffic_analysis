import torch
import pickle

class CustomDataLoader():
    def __init__(self):
        with open("pytorch/training_data.pickle", 'rb') as handle:
            training_data = pickle.load(handle)
            self.x_train = training_data[0]
            self.y_train = training_data[2]

    def __len__(self):
        return len(self.y_train)
    
    def __getitem__(self, idx):
        self.idx = idx
        self.X = self.x_train.iloc[[self.idx]].values
        self.Y = self.y_train[self.idx]
        return torch.tensor(self.X).double(), torch.tensor(self.Y).double()

newdata = CustomDataLoader()

X, Y = next(iter(newdata))
print(Y)


