import numpy as np
from sklearn.model_selection import train_test_split


def isotropic_predictor_data(num_samples, predictor_dim, respond_dim, noisy_variance, sparse=0, seed=666):
    np.random.seed(seed)
    x = np.random.randn(num_samples, predictor_dim)

    trans = np.random.randn(predictor_dim, respond_dim)
    if sparse > 0:
        sparse_mask = np.random.rand(0, 2, trans.shape)
        trans = (sparse_mask > sparse) * trans

    y = x.dot(trans) + np.random.randn(num_samples,
                                       respond_dim) * noisy_variance

    return (x, y), trans


def get_mnist():
    from sklearn.datasets import fetch_openml
    print("loading mnist")
    mnist = fetch_openml('mnist_784', cache=True, as_frame=False)
    X = mnist.data.astype('float32')
    y = mnist.target.astype('int64')
    X /= 255.0
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42)
    print("mnist loaded")
    return X_train, X_test, y_train, y_test


def get_20news():
    from sklearn.datasets import fetch_20newsgroups_vectorized
    print("loading 20news")
    X, y = fetch_20newsgroups_vectorized(subset="all", return_X_y=True)
    X = X.astype('float32')
    y = y.astype('int64')
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, random_state=42, stratify=y, test_size=0.1
    )
    print("20news loaded")
    return X_train, X_test, y_train, y_test
