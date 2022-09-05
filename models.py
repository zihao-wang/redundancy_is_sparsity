from abc import abstractmethod
import math
from typing import Dict
import torch
from torch import nn
from torch import functional as F

class MyModelMixin:
    @abstractmethod
    def get_weights(self) -> Dict:
        pass

class Linear(nn.Linear, MyModelMixin):
    def get_weights(self):
        if self.bias:
            return {
                'weight': self.weight,
                'bias': self.bias
            }
        else:
            return {'weight': self.weight}

class LinearRegression(nn.Module, MyModelMixin):
    def __init__(
            self,
            input_dim,
            output_dim,
            bias=True):
        super(LinearRegression, self).__init__()
        self.linear = Linear(input_dim, output_dim, bias)

    def forward(self, X, **kwargs):
        return self.linear(X)

    def get_weights(self):
        return self.linear.get_weights()


class LogisticRegression(nn.Module, MyModelMixin):
    def __init__(
            self,
            input_dim,
            output_dim,
            bias=True):
        super(LogisticRegression, self).__init__()
        self.linear = Linear(input_dim, output_dim, bias)

    def forward(self, X, **kwargs):
        X = torch.softmax(self.linear(X), dim=-1)
        return X

    def get_weights(self):
        return self.linear.get_weights()

class SpaRedLinear(nn.Module, MyModelMixin):
    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(SpaRedLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(
            (out_features, in_features), **factory_kwargs))
        self.weight2 = nn.Parameter(torch.empty(
            (out_features, in_features), **factory_kwargs))
        self.use_bias = bias
        if bias:
            self.bias = nn.Parameter(torch.empty(
                out_features, **factory_kwargs))
            self.bias2 = nn.Parameter(
                torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, X, **kwargs):
        weight = self.weight * self.weight2
        if self.use_bias:
            bias = self.bias * self.bias2
            X = nn.functional.linear(X, weight, bias)
        else:
            X = nn.functional.linear(X, weight)
        return X

    def get_weights(self):
        if self.use_bias:
            return {'weight': self.weight * self.weight2,
                    'bias': self.bias * self.bias2}
        else:
            return {'weight': self.weight * self.weight2}


class SparedLinearRegression(nn.Module, MyModelMixin):
    def __init__(
            self,
            input_dim,
            output_dim,
            bias=True):
        super(SparedLinearRegression, self).__init__()
        self.linear = SpaRedLinear(input_dim, output_dim, bias)

    def forward(self, x):
        return self.linear(x)

    def get_weights(self):
        return self.linear.get_weights()


class SparseLogisticRegression(nn.Module, MyModelMixin):
    def __init__(
            self,
            input_dim,
            output_dim,
            bias=True):
        super(SparseLogisticRegression, self).__init__()
        self.output = SpaRedLinear(input_dim, output_dim, bias)

    def forward(self, X, **kwargs):
        X = torch.softmax(self.output(X), dim=-1)
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
        X = torch.relu(self.hidden(X))
        #X = self.dropout(X)
        X = torch.softmax(self.output(X), dim=-1)
        return X

    def get_weights(self):
        return {
            "output_weights": self.output.weight,
            "hidden_weights": self.hidden.weight,
        }


class SparseFeatureLinear(nn.Module):
    def __init__(
            self,
            input_dim,
            output_dim):
        super(SparseFeatureLinear, self).__init__()
        self.input_mask = nn.Parameter(
            torch.zeros([1, input_dim]).normal_(0, 1))
        self.output = nn.Linear(input_dim, output_dim)

    def forward(self, X, **kwargs):
        X = (X * self.input_mask)
        #X = self.dropout(X)
        X = torch.softmax(self.output(X), dim=-1)
        return X

    def get_weights(self):
        return {
            "output_weight": self.output.weight * self.input_mask
        }

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

        self.hidden = SpaRedLinear(input_dim, hidden_dim)
        self.output = SpaRedLinear(hidden_dim, output_dim)

    def forward(self, X, **kwargs):
        X = torch.relu(self.hidden(X))
        #X = self.dropout(X)
        X = torch.softmax(self.output(X), dim=-1)
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
        self.input_mask = nn.Parameter(
            torch.zeros([1, input_dim]).normal_(0, 1))
        self.hidden = nn.Linear(input_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, output_dim)

    def forward(self, X, **kwargs):
        X = torch.relu(self.hidden(X * self.input_mask))
        X = torch.softmax(self.output(X), dim=-1)
        return X

    def get_weights(self):
        return {'hidden_weights': self.hidden.weight * self.input_mask.T,
                'output_weights': self.output.weight}
