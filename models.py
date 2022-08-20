import torch

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