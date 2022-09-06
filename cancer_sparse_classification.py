import argparse
import os
import logging
from secrets import choice
import time

import torch
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.svm import SVC

from data import get_cancer_GDS
from models import FNN, SparseWeightNet
from routines import run_classification


parser = argparse.ArgumentParser()

parser.add_argument("--model_name", type=str, default='sparednet', choices=['sparednet', 'mlpnet', 'hsiclasso'])
parser.add_argument("--dataset_name", type=str, default="")
parser.add_argument("--num_trials", type=int, default=20)
parser.add_argument("--log_dir", type=str, default='log')
# hsic lasso arguments
parser.add_argument("--num_feat", type=int, default=50)
parser.add_argument("--B", type=int, default=20)
# spared arguments
parser.add_argument("--alpha", type=float, default=1e-3)
parser.add_argument("--lr", type=float, default=1e-5)
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--epochs", type=int, default=200)
parser.add_argument("--device", type=str, default="cuda:0")
parser.add_argument("--optname", type=str, default="Adam")


def _logistic_regression(X_train, y_train, X_test, y_test, **kwargs):
    clf = LogisticRegression(penalty='l1', solver='saga', C=10000, max_iter=10000)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_pred, y_test)
    return acc

def _hsic_lasso(X_train, y_train, X_test, y_test, num_feat, B, **kwargs):
    from pyHSICLasso import HSICLasso
    hsic_lasso = HSICLasso()
    hsic_lasso.input(X_train, y_train)
    hsic_lasso.classification(num_feat=num_feat, B=B)
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
    output_dim = max(np.max(y_train), np.max(y_test)) + 1
    net = SparseWeightNet(input_dim, output_dim, hidden_dim=4096).to(device)
    acc, metric_list = run_classification(
        alpha, X_train, y_train, X_test, y_test, net,
        device=device, eval_every_epoch=5, **kwargs)
    return acc

def _mlp_net(X_train, y_train, X_test, y_test, alpha, device, **kwargs):
    input_dim = X_train.shape[1]
    output_dim = max(np.max(y_train), np.max(y_test)) + 1
    net = FNN(input_dim, output_dim, hidden_dim=4096).to(device)
    acc, metric_list = run_classification(
        alpha, X_train, y_train, X_test, y_test, net,
        device=device, eval_every_epoch=5, **kwargs)
    return acc

def evaluate_model(X_train, y_train, X_test, y_test, model_name, **kwargs):
    if model_name.lower() == 'logisticregression':
        acc = _logistic_regression(X_train, y_train, X_test, y_test, **kwargs)
    elif model_name.lower() == 'mlpnet':
        acc = _mlp_net(X_train, y_train, X_test, y_test, **kwargs)
    elif model_name.lower() == 'sparednet':
        acc = _spared_net(X_train, y_train, X_test, y_test, **kwargs)
    elif model_name.lower() == 'hsiclasso':
        acc = _hsic_lasso(X_train, y_train, X_test, y_test, **kwargs)
    else:
        raise NotImplementedError

    return acc


def cross_validate_model(X, y, model_name, **kwargs):
    kf = KFold(n_splits=10)
    total_acc = 0
    for i, (train_index, test_index) in enumerate(kf.split(X)):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        t0 = time.time()
        acc = evaluate_model(X_train, y_train, X_test, y_test, model_name, **kwargs)
        dt = time.time() - t0
        metric = {
            'acc': acc,
            'dt': dt
        }
        logging.info(f"cross validation fold:{metric}")
        total_acc += acc
    print(model_name, acc / 10)
    return acc

def multi_trials_model(X, y, model_name, num_trials=100, **kwargs):
    acc_list = []
    for i in range(num_trials):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)
        t0 = time.time()
        acc = evaluate_model(X_train, y_train, X_test, y_test, model_name, **kwargs)
        dt = time.time() - t0
        metric = {
            'acc': acc,
            'dt': dt
        }
        logging.info(f"random trail {i+1} of {num_trials}:{metric}")
        acc_list.append(acc)
    ret = {
        'model_name': model_name,
        'mean_acc': np.mean(acc_list),
        'var_acc': np.std(acc_list),
        }
    return ret

if __name__ == "__main__":
    args = parser.parse_args()

    logging.basicConfig(filename=args.log_dir,
                        format='%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        filemode='wt',
                        level=logging.INFO)

    logging.info(args)

    model_name_dist = {
        'hsiclasso': {
            "num_feat": [1, 5, 10, 100]
        },
        'logisticregression': {},
        'sparednet': {}}

    np.random.seed(111)

    for f in os.listdir('data'):
        if f.endswith('soft.gz') and args.dataset_name in f:
            filepath = os.path.join("data", f)
            X, y = get_cancer_GDS(filepath)
            perm = np.random.permutation(X.shape[0])
            X = X[perm, :]
            y = y[perm]
            metric = multi_trials_model(X, y, **vars(args))
            logging.info("---- Begin of Report ----")
            logging.info(args)
            logging.info(f)
            logging.info(metric)
            logging.info("---- End of Report ----")
