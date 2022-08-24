import numpy as np

def eval_over_datasets(x, y, trans, alpha):
    _, predictor_dim = x.shape
    _, respond_dim = y.shape
    trans = trans.reshape((predictor_dim, respond_dim))
    diff = y - x.dot(trans)

    mse = np.linalg.norm(diff ** 2, axis=-1, ord=2).mean()/2
    l1 = np.linalg.norm(trans, ord=1)

    zero_rate3 = (np.abs(trans) < 1e-3).mean()
    zero_rate6 = (np.abs(trans) < 1e-6).mean()
    zero_rate9 = (np.abs(trans) < 1e-9).mean()
    zero_rate12 = (np.abs(trans) < 1e-12).mean()
    ret = {'mse': mse, 'l1': l1, 'total': mse + l1 * alpha,
            'zero_rate3': zero_rate3,
            'zero_rate6': zero_rate6,
            'zero_rate9': zero_rate9,
            'zero_rate12': zero_rate12}
    return {k: float(v) for k, v in ret.items()}
