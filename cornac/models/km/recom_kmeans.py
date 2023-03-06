import numpy as np
import scipy.sparse as sp

from ..recommender import Recommender
from ...exception import ScoreException


class KMeans(Recommender):
    """K-means based recommender.

    Parameters
    ----------
    k: int, optional, default: 5
        Number of clusters.

    max_iter: int, optional, default: 100
        The maximum number of iterations.

    name: string, optional, default: 'Kmeans'
        The name of the recommender model.

    trainable: boolean, optional, default: True
        When False, the model is not trained and Cornac assumes that the model is already \
        trained. 
        
    tol : float, optional, default: 1e-6
        Convergence threshold
        
    verbose: boolean, optional, default: False
        When True, some running logs are displayed.

    seed: int, optional, default: None
        Random seed for parameters initialization.

    init_par: numpy 1d array, optional, default: None
        The initial object parition, 1d array contaning the cluster label (int type starting from 0) \
        of each object (user). If par = None, then kmeans is initialized randomly.
    """

    def __init__(
            self,
            k=5,
            max_iter=100,
            name="Kmeans",
            trainable=True,
            tol=1e-6,
            verbose=False,
            seed=None,
            init_par=None,
    ):
        Recommender.__init__(self, name=name, trainable=trainable, verbose=verbose)
        self.k = k
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose
        self.seed = seed
        self.init_par = init_par
        self.centroids = None  # matrix of cluster centroids

    def fit(self, train_set, val_set=None):
        """Fit the model to observations.

        Parameters
        ----------
        train_set: :obj:`cornac.data.Dataset`, required
            User-Item preference data as well as additional modalities.

        val_set: :obj:`cornac.data.Dataset`, optional, default: None
            User-Item preference data for model selection purposes (e.g., early stopping).

        Returns
        -------
        self : object
        """
        Recommender.fit(self, train_set, val_set)

        X = self.train_set.matrix
        X = sp.csr_matrix(X)

        # Kmeans requires rows of X to have a unit L2 norm. We therefore need to make a copy of X as we should not modify the latter.
        X1 = X.copy()
        X1 = X1.multiply(
            sp.csc_matrix(1.0 / (np.sqrt(X1.multiply(X1).sum(1).A1) + 1e-20)).T
        )

        if self.trainable:
            from .kmeans import kmeans

            res = kmeans(
                X1,
                k=self.k,
                max_iter=self.max_iter,
                tol=self.tol,
                verbose=self.verbose,
                seed=self.seed,
                init_par=getattr(self, "final_par", self.init_par),
            )
            self.centroids = res["centroids"]
            self.final_par = res["partition"]
        else:
            print("%s is trained already (trainable = False)" % (self.name))

        self.user_center_sim = (
            X1 * self.centroids.T
        )  # user-centroid cosine similarity matrix
        del X1

        return self

    def score(self, user_idx, item_idx=None):
        """Predict the scores/ratings of a user for an item.

        Parameters
        ----------
        user_idx: int, required
            The index of the user for whom to perform score prediction.

        item_idx: int, optional, default: None
            The index of the item for which to perform score prediction.
            If None, scores for all known items will be returned.

        Returns
        -------
        res : A scalar or a Numpy array
            Relative scores that the user gives to the item or to all known items

        """
        if item_idx is None:
            if self.train_set.is_unk_user(user_idx):
                raise ScoreException(
                    "Can't make score prediction for (user_id=%d)" % user_idx
                )

            known_item_scores = self.centroids.multiply(
                self.user_center_sim[user_idx, :].T
            )
            known_item_scores = known_item_scores.sum(0).A1 / (
                    self.user_center_sim[user_idx, :].sum() + 1e-20
            )  # weighted average of cluster centroids
            return known_item_scores
        else:
            if self.train_set.is_unk_user(user_idx) or self.train_set.is_unk_item(
                    item_idx
            ):
                raise ScoreException(
                    "Can't make score prediction for (user_id=%d, item_id=%d)"
                    % (user_idx, item_idx)
                )

            user_pred = self.centroids[:, item_idx].multiply(
                self.user_center_sim[user_idx, :].T
            )
            # transform user_pred to a flatten array
            user_pred = user_pred.sum(0).A1 / (
                    self.user_center_sim[user_idx, :].sum() + 1e-20
            )  # weighted average of cluster centroids

            return user_pred
