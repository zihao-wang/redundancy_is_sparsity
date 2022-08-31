import torch
from torch import nn
from torch import functional as F

class LinearRegression(torch.nn.Module):
    def __init__(self, predictor_dim, respond_dim=1):
        super(LinearRegression, self).__init__()
        self.weight = torch.nn.Parameter(torch.randn(predictor_dim, respond_dim))

    def forward(self, x):
        return x.mm(self.weight)

    def get_weights(self):
        return self.weight.cpu().detach().numpy()

    def L1_reg(self):
        return torch.norm(self.weight, p=1)


class RSLinearRegression(torch.nn.Module):
    def __init__(self, predictor_dim, respond_dim=1):
        super(RSLinearRegression, self).__init__()
        self.weight = torch.nn.Parameter(torch.randn(predictor_dim, respond_dim))
        self.shadow_weight = torch.nn.Parameter(torch.randn(predictor_dim, respond_dim))

    def forward(self, x):
        w = self.weight * self.shadow_weight
        return x.mm(w)

    def get_weights(self):
        return self.weight.cpu().detach().numpy() * self.shadow_weight.cpu().detach().numpy()

    def get_redundant_weights(self):
        return self.weight.cpu().detach().numpy(), self.shadow_weight.cpu().detach().numpy()

    def L1_reg(self):
        return torch.norm(self.weight * self.shadow_weight, p=1)



class RedLinear(nn.Module):
    def __init__(self,in_dim,out_dim):
        super(RedLinear, self).__init__()
        self.weight = nn.Parameter(torch.zeros([in_dim, out_dim]).normal_(0, 1/out_dim ** 0.5))
        self.weight2 = nn.Parameter(torch.zeros([in_dim, out_dim]).normal_(0, 1/out_dim ** 0.5))
        self.bias = nn.Parameter(torch.zeros([1, out_dim]).normal_(0, 1))


    def forward(self, X, **kwargs):
        X = (X).mm((self.weight * self.weight2)) + self.bias
        return X

    def get_weights(self):
        return {'weights': self.weight * self.weight2}


class SparseWeightNet(nn.Module):
    def __init__(
            self,
            input_dim,
            output_dim,
            hidden_dim,
            dropout=0.5,
    ):
        super(SparseWeightNet, self).__init__()
        self.dropout = nn.Dropout(dropout)

        self.hidden = RedLinear(input_dim, hidden_dim)
        self.output = RedLinear(hidden_dim, output_dim)

    def forward(self, X, **kwargs):
        X = F.relu(self.hidden(X))
        #X = self.dropout(X)
        X = F.softmax(self.output(X), dim=-1)
        return X

    def get_weights(self):
        return {'hidden_weights': self.hidden.get_weights()['weights'],
                'output_weights': self.output.get_weights()['weights']}


class SparseFeatureNet(nn.Module):
    def __init__(
            self,
            input_dim,
            output_dim,
            hidden_dim=200,
    ):
        super(SparseFeatureNet, self).__init__()
        self.input_mask = nn.Parameter(torch.zeros([1, input_dim]).normal_(0, 1))
        self.hidden = nn.Linear(input_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, output_dim)

    def forward(self, X, **kwargs):
        X = F.relu(self.hidden(X * self.input_mask))
        X = F.softmax(self.output(X), dim=-1)
        return X

    def get_weights(self):
        return {'hidden_weights': self.hidden.weight * self.input_mask.T,
                'output_weights': self.output.weight}

class SparseFeatureLinear(nn.Module):
    def __init__(
            self,
            input_dim,
            output_dim):
        super(SparseFeatureLinear, self).__init__()
        self.input_mask = nn.Parameter(torch.zeros([1, input_dim]).normal_(0, 1))
        self.output = nn.Linear(input_dim, output_dim)

    def forward(self, X, **kwargs):
        X = (X * self.input_mask)
        #X = self.dropout(X)
        X = F.softmax(self.output(X), dim=-1)
        return X

    def get_weights(self):
        return {
            "output_weights": self.output.weight * self.input_mask
        }

class SparseLinear(nn.Module):
    def __init__(
            self,
            input_dim,
            output_dim):
        super(SparseLinear, self).__init__()
        self.output = RedLinear(input_dim, output_dim)

    def forward(self, X, **kwargs):
        #X = self.dropout(X)
        X = F.softmax(self.output(X), dim=-1)
        return X

    def get_weights(self):
        return self.output.get_weights()

class FNN(nn.Module):
    def __init__(
            self,
            input_dim,
            output_dim,
            hidden_dim=200,
            dropout=0.5,
    ):
        super(FNN, self).__init__()
        self.dropout = nn.Dropout(dropout)

        self.hidden = nn.Linear(input_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, output_dim)

    def forward(self, X, **kwargs):
        X = F.relu(self.hidden(X))
        #X = self.dropout(X)
        X = F.softmax(self.output(X), dim=-1)
        return X

    def get_weights(self):
        return {
            "output_weights": self.output.weight,
            "hidden_weights": self.hidden.weight,
        }