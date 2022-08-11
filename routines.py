import time

import torch
from tqdm import trange
from sklearn.linear_model import Lasso

from models import RSLinearRegression, LinearRegression
from utils import eval_over_datasets

def sgd_duplication_regression(alpha, x, y, lr, batch_size, epoch, device):
    _, predictor_dim = x.shape
    _, respond_dim = y.shape
    x_tensor = torch.tensor(x, dtype=torch.float32, device=device)
    y_tensor = torch.tensor(y, dtype=torch.float32, device=device)
    dataset = torch.utils.data.TensorDataset(x_tensor, y_tensor)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
    model = RSLinearRegression(predictor_dim=predictor_dim,
                               respond_dim=respond_dim)

    # using weight decay for L2 regularization
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=alpha)

    metric_list = []
    t = trange(epoch, desc="redundancy-sparse regression")
    for e in t:
        total_loss = 0
        for x_batch, y_batch in dataloader:
            y_pred = model(x_batch)
            loss = torch.sum((y_batch - y_pred)** 2) / 2 / y_pred.size(0)
            # reg = torch.norm(model.weight)**2 + torch.norm(model.shadow_weight)**2
            # loss += reg * alpha
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        metric = eval_over_datasets(x, y, model.get_weights(), alpha)
        metric['epoch'] = e+1
        metric_list.append(metric)
        t.set_postfix(metric)
    return model, metric_list

def sgd_L1_regression(alpha, x, y, lr, batch_size, epoch, device):
    _, predictor_dim = x.shape
    _, respond_dim = y.shape
    x_tensor = torch.tensor(x, dtype=torch.float32, device=device)
    x_tensor = torch.tensor(y, dtype=torch.float32, device=device)
    dataset = torch.utils.data.TensorDataset(x_tensor, x_tensor)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
    model = LinearRegression(predictor_dim=predictor_dim,
                             respond_dim=respond_dim)

    # using L1 regularization
    model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    metric_list = []
    t = trange(epoch, desc="L1 reg regression")
    for e in t:
        total_loss = 0
        for x_batch, y_batch in dataloader:
            y_pred = model(x_batch)
            loss = torch.sum((y_batch - y_pred)** 2) / 2 / y_pred.size(0)
            reg = torch.norm(model.weight, p=1)
            loss = loss + reg * alpha
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        metric = eval_over_datasets(x, y, model.get_weights(), alpha)
        metric['epoch'] = e+1
        metric_list.append(metric)
        t.set_postfix(metric)
    return model, metric_list


def run_lasso(alpha, x, y):
  _, predictor_dim = x.shape
  _, respond_dim = y.shape

  lasso_regressor = Lasso(alpha=alpha)

  t = time.time()
  lasso_regressor.fit(x, y)
  t = time.time() - t

  _coef = lasso_regressor.coef_
  weights = _coef.reshape((predictor_dim, respond_dim))

  return {'time': t,
          'weights': weights}

def run_redundent_sparse_regression(alpha, x, y,
                          batch_size=512, lr=1e-4, epoch=200, device='cuda:0'):
  num_samples, predictor_dim = x.shape
  _, respond_dim = y.shape

  t = time.time()
  rs_regressor, metric_list = sgd_duplication_regression(alpha, x, y,
                                            batch_size=batch_size,
                                            lr=lr,
                                            epoch=epoch,
                                            device=device)
  t = time.time() - t
  weights = rs_regressor.get_weights()

  return {'time': t,
          'weights': weights,
          'metric_list': metric_list}

def run_l1_regression(alpha, x, y,
                      batch_size=512, lr=1e-4, epoch=200, device='cuda:0'):
  num_samples, predictor_dim = x.shape
  _, respond_dim = y.shape

  t = time.time()
  rs_regressor, metric_list = sgd_L1_regression(alpha, x, y,
                                   batch_size=batch_size,
                                   lr=lr,
                                   epoch=epoch,
                                   device=device)
  t = time.time() - t

  weights = rs_regressor.get_weights()
  weights = weights.reshape((predictor_dim, respond_dim))

  return {'time': t,
          'weights': weights,
          'metric_list': metric_list}