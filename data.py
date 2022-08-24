import numpy as np

def isotropic_predictor_data(num_samples, predictor_dim, respond_dim, noisy_variance, sparse=0, seed=666):
    np.random.seed(seed)
    x = np.random.randn(num_samples, predictor_dim)

    trans = np.random.randn(predictor_dim, respond_dim) * 2
    if sparse > 0:
      sparse_mask = np.random.rand(0, 2, trans.shape)
      trans = (sparse_mask > sparse) * trans

    y = x.dot(trans) + np.random.randn(num_samples, respond_dim) * noisy_variance

    return (x, y), trans