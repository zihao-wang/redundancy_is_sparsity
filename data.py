import GEOparse
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
    return (X_train, X_test, y_train, y_test), (X.shape[1], 20)


def get_20news():
    from sklearn.datasets import fetch_20newsgroups_vectorized
    print("loading 20news")
    X, y = fetch_20newsgroups_vectorized(subset="all", return_X_y=True, as_frame=False)
    print(X)
    X = np.asarray(X.todense(), dtype='float32')
    y = y.astype('int64')
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, random_state=42, stratify=y, test_size=0.1
    )
    print("20news loaded")
    return (X_train, X_test, y_train, y_test), (X.shape[1], 20)


def get_cancer_GDS(filepath):
    gds = GEOparse.get_GEO(filepath=filepath)
    X = []
    y = []
    subset_keys = list(gds.subsets.keys())
    for i, k in enumerate(subset_keys):
        sample_ids = gds.subsets[k].metadata['sample_id'][0].split(',')
        for sample_id in sample_ids:
            _x = gds.table.loc[:, sample_id].to_numpy().reshape((1, -1))
            _y = i
            X.append(_x)
            y.append(_y)

    X_arr = np.concatenate(X, axis=0).astype('float32')
    X_arr[np.isnan(X_arr)] = 0
    y_arr = np.asarray(y).astype('int64')

    print(f"GDS dataset {filepath} loaded")
    print(f"#features {X_arr.shape[1]}, #labels {np.max(y)+1}, #samples {X_arr.shape[0]}")
    return X_arr, y_arr