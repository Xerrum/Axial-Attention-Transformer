import torch

class Time2Vec(torch.nn.Module):
    """
    Provides positional encoding functionality in the form of a PyTorch module.
    This file was contained in the original project and was never moved.
    """
    def __init__(self, input_dim, output_dim):
        super(Time2Vec, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)
        self.periodic = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return torch.cat((self.linear(x), torch.sin(self.periodic(x))), -1)
        #return torch.cat((self.linear(x), torch.tanh(self.periodic(x))), -1)
        #return torch.cat((self.linear(x), torch.cos(self.periodic(x))), -1)
        #return torch.cat((self.linear(x), torch.exp(self.periodic(x))), -1)