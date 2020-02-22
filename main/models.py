import torch
import torch.functional as F


class SimpleNet(torch.nn.Module):
    # TODO: Implement more sophisticated architectures
    def __init__(self, in_dim, hidden_dim, out_dim):
        super(SimpleNet, self).__init__()
        self.fc1 = torch.nn.Linear(in_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, out_dim)

    def forward(self, predictors):
        hidden = F.relu((self.fc1(predictors)))
        return self.fc2(hidden)
 
