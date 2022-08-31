import argparse
import json
import logging
import os

import torch
from sklearn.linear_model import LogisticRegression
from skorch import NeuralNetClassifier
from torch import nn


import data
from models import RedLinear

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='mnist')
parser.add_argument('--lr', type=float, default=4e-3)
parser.add_argument('--epoch', type=int, default=20)
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--alpha', type=float, default=1e-5)
parser.add_argument('--logging_path', type=str, default="output/SparseLogisticRegression/log")



class SparseLinear(nn.Module):
    def __init__(
            self,
            input_dim=130107,
            output_dim=20):
        super(SparseLinear, self).__init__()
        self.output = RedLinear(input_dim, output_dim)

    def forward(self, X, *args, **kwargs):
        #X = self.dropout(X)
        X = torch.softmax(self.output(X), dim=-1)
        return X

    def get_weights(self):
        return self.output.get_weights()


def neural_train_and_compress_ratio(dataset_callback,
                                    max_epochs=2,
                                    lr=0.004,
                                    weight_decay=1e-5,
                                    thr=1e-10,
                                    device="cuda:0"):
    X_train, X_test, y_train, y_test = dataset_callback()
    net = NeuralNetClassifier(
        SparseLinear,
        max_epochs=max_epochs,
        lr=lr,
        batch_size=1024,
        device=device,
        optimizer=torch.optim.Adam,
        optimizer__weight_decay=weight_decay
    )
    net.fit(X_train, y_train)

    y_pred = net.predict(X_test)
    accuracy = (y_pred == y_test).mean()

    net_weights = net.module_.get_weights()
    total_num = 0
    non_zero_num = 0
    for _, w in net_weights.items():
        total_num += w.numel()
        non_zero_num += (w > thr).sum()
    compress_ratio = total_num / non_zero_num
    print(accuracy, compress_ratio)
    return {'accuracy': accuracy,
            'compress_ratio': compress_ratio.item()}


def classic_train_and_compress_ratio(dataset_callback,
                                     solver,
                                     multi_class,
                                     alpha,
                                     thr=1e-10):
    X_train, X_test, y_train, y_test = dataset_callback()

    clf = LogisticRegression(
        penalty='l1',
        C=1/alpha,
        solver=solver,
        multi_class=multi_class
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    accuracy = (y_pred == y_test).mean()

    net_weights = clf.module_.get_weights()
    total_num = 0
    non_zero_num = 0
    for _, w in net_weights.items():
        total_num += w.numel()
        non_zero_num += (w > thr).sum()
    compress_ratio = total_num / non_zero_num
    print(accuracy, compress_ratio)
    return {'accuracy': accuracy.item(),
            'compress_ratio': compress_ratio.item()}


if __name__ == "__main__":
    args = parser.parse_args()

    os.makedirs(
        os.path.dirname(args.logging_path),
        exist_ok=True
    )
    logging.basicConfig(
        filename=args.logging_path,
        level=logging.INFO,
        filemode='at'
    )

    dataset_cbk = getattr(data, 'get_' + args.dataset)

    fetch = neural_train_and_compress_ratio(
        dataset_cbk,
        max_epochs=args.epoch,
        lr=args.lr,
        weight_decay=args.alpha,
        device=args.device
    )

    fetch['alpha'] = args.alpha
    logging.info(fetch)

    # classic_train_and_compress_ratio(
    #     dataset_cbk,
    #     solver='saga',
    #     multi_class='multinomial',
    #     alpha=args.alpha
    # )
