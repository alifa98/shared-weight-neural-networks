import torch.nn as nn

# Define a Simple MLP
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, hidden_size)
        self.fc5 = nn.Linear(hidden_size, num_classes)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.relu(self.fc4(x))
        x = self.fc5(x)
        return x
    
    
# Define MLP B (Weight sharing between layer 2 and 4)
class MLPSharedWeights(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(MLPSharedWeights, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.shared_fc = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc5 = nn.Linear(hidden_size, num_classes)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.shared_fc(x))
        x = self.relu(self.fc3(x))
        x = self.relu(self.shared_fc(x))
        x = self.fc5(x)
        return x