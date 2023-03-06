import numpy as np
import scipy.sparse as sp

def kmeans(X, k=5, max_iter=100, tol=1e-6, verbose=False, seed=None, init_par=None):
    # K-means clustering algorithm

    n = X.shape[0]
    if init_par is None:
        prng = np.random.RandomState(seed)
        par = prng.randint(k, size=n)
    else:
        par = init_par

    # Initialisation of the classification matrix Z
    Z = sp.lil_matrix((n, k))
    Z[np.arange(n), par] = 1

    change = True
    l_init = -1e1000
    l = []
    iter_ = 0
    while change and iter_ < max_iter:
        change = False
        # Update centroids
        MU = Z.T * X

        # Object Assignements
        Z1 = X * MU.T
        par = Z1.argmax(1).A1  # The object partition in k clusters
        # update the classification matrix
        Z = sp.lil_matrix((n, k))
        Z[np.arange(len(par)), par] = 1

        # Kmeans criteria (likelihood)
        l_t = Z1.multiply(Z).sum()

        if np.abs(l_t - l_init) > tol:
            if verbose:
                print("Iter %i, likelihood: %f" % (iter_ + 1, l_t))
            l_init = l_t
            change = True
            l.append(l_t)
            iter_ += 1

    return {"centroids": MU, "partition": par}
