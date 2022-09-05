import os

import torch
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.svm import SVC

from data import get_cancer_GDS
from models import SparseWeightNet
from routines import run_rs_regression


def _logistic_regression(X_train, y_train, X_test, y_test, **kwargs):
    clf = LogisticRegression(penalty='l1', solver='saga', C=10000, max_iter=10000)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_pred, y_test)
    return acc

def _hsic_lasso(X_train, y_train, X_test, y_test, num_feat, **kwargs):
    from pyHSICLasso import HSICLasso
    hsic_lasso = HSICLasso()
    hsic_lasso.input(X_train, y_train)
    hsic_lasso.classification(num_feat)
    feat_index = hsic_lasso.get_index()
    X_train_selected = X_train[:, feat_index]
    X_test_selected = X_test[:, feat_index]
    # first pick the variables
    clf = SVC()
    clf.fit(X_train_selected, y_train)
    y_pred = clf.predict(X_test_selected)
    acc = accuracy_score(y_pred, y_test)
    return acc

def _spared_net(X_train, y_train, X_test, y_test, alpha, device, **kwargs):
    input_dim = X_train.shape[1]
    print(X_train)
    output_dim = max(np.max(y_train), np.max(y_test)) + 1
    net = SparseWeightNet(input_dim, output_dim, hidden_dim=4096).to(device)
    run_rs_regression(alpha, X_train, y_train, net, device=device, **kwargs)
    X_test_ten = torch.tensor(X_test, device=device)
    y_test_ten = torch.tensor(y_test, device=device)
    y_pred = net(X_test_ten).argmax(-1)
    acc = (y_pred == y_test_ten).float().item()
    return acc

def evaluate_model(X_train, y_train, X_test, y_test, model_name, **kwargs):
    if model_name.lower() == 'logisticregression':
        acc = _logistic_regression(X_train, y_train, X_test, y_test, **kwargs)
    elif model_name.lower() == 'sparednet':
        acc = _spared_net(X_train, y_train, X_test, y_test, **kwargs)
    elif model_name.lower() == 'hsiclasso':
        acc = _hsic_lasso(X_train, y_train, X_test, y_test, **kwargs)

    return acc


def cross_validate_model(X, y, model_name, **kwargs):
    kf = KFold(n_splits=10)
    total_acc = 0
    for i, (train_index, test_index) in enumerate(kf.split(X)):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        acc = evaluate_model(X_train, y_train, X_test, y_test, model_name, **kwargs)
        print("cross validation fold", i, acc)
        total_acc += acc
    print(model_name, acc / 10)


if __name__ == "__main__":
    model_name_dist = {
        'hsiclasso': {
            "num_feat": [1, 5, 10, 100]
        },
        'logisticregression': {},
        'sparednet': {}}

    np.random.seed(111)

    for f in os.listdir('data'):
        if f.endswith('soft.gz'):
            filepath = os.path.join("data", f)
            X, y = get_cancer_GDS(filepath)
            perm = np.random.permutation(X.shape[0])
            X = X[perm, :]
            y = y[perm]
            model_name = 'hsiclasso'
            model_name = 'sparednet'
            cross_validate_model(X, y, model_name, num_feat=50, alpha=1, device='cpu')
