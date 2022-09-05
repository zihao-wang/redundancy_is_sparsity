from data import get_cancer_GDS
import os
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold

def _logistic_regression(X_train, y_train, X_test, y_test):
    clf = LogisticRegression(penalty='l1', solver='saga', C=10000, max_iter=10000)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_pred, y_test)
    return acc


def evaluate_model(X_train, y_train, X_test, y_test, model_name):
    if model_name.lower() == 'logisticregression':
        acc = _logistic_regression(X_train, y_train, X_test, y_test)

    return acc

if __name__ == "__main__":
    model_name = 'logisticregression'
    num_folds = 5

    for f in os.listdir('data'):
        if f.endswith('soft.gz'):
            filepath = os.path.join("data", f)
            X, y = get_cancer_GDS(filepath)
            kf = KFold(n_splits=num_folds)
            total_acc = 0
            for i, (train_index, test_index) in enumerate(kf.split(X)):
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]

                acc = evaluate_model(X_train, y_train, X_test, y_test, model_name)
                print("cross validation fold", i, acc)
                total_acc += acc
            print(model_name, acc / num_folds)