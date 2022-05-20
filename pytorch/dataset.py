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

class CustomDataLoader_Labels():
    def __init__(self):
        with open("pytorch/training_data_by_labels.pickle", 'rb') as handle:
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


class CustomDataLoader_testing():
    def __init__(self):
        with open("pytorch/training_data.pickle", 'rb') as handle:
            training_data = pickle.load(handle)
            self.x_test = training_data[1]
            self.y_test = training_data[3]

    def __len__(self):
        return len(self.y_test)
    
    def __getitem__(self, idx):
        self.idx = idx
        self.X = self.x_test.iloc[[self.idx]].values
        self.Y = self.y_test[self.idx]
        return torch.tensor(self.X).double(), torch.tensor(self.Y).double()

class CustomDataLoader_testing_Labels():
    def __init__(self):
        with open("pytorch/training_data_by_labels.pickle", 'rb') as handle:
            training_data = pickle.load(handle)
            self.x_test = training_data[1]
            self.y_test = training_data[3]

    def __len__(self):
        return len(self.y_test)
    
    def __getitem__(self, idx):
        self.idx = idx
        self.X = self.x_test.iloc[[self.idx]].values
        self.Y = self.y_test[self.idx]
        return torch.tensor(self.X).double(), torch.tensor(self.Y).double()

