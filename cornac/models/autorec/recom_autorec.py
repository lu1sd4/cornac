import numpy as np

from ..recommender import Recommender

class AutoRec(Recommender):
  """AutoRec: Autoencoders Meet Collaborative Filtering
  
  Parameters
  ----------
  k: int, optional, default: 600
      The dimension of the hidden layer. Default: 600

  n_epochs: int, optional, default: 100
      The number of epochs for SGD.

  batch_size: int, optional, default: 256
      The batch size.

  learning_rate: float, optional, default: 0.001
      The learning rate for Adam.

  lambda_reg: float, optional, default: 0.0005

  name: string, optional, default: 'VAECF'
      The name of the recommender model.

  trainable: boolean, optional, default: True
      When False, the model is not trained and Cornac assumes that the model is already \
      pre-trained.

  verbose: boolean, optional, default: False
      When True, some running logs are displayed.

  seed: int, optional, default: None
      Random seed for parameters initialization.

  use_gpu: boolean, optional, default: False
      If True and your system supports CUDA then training is performed on GPUs.

  References
  ----------
  * Liang, Dawen, Rahul G. Krishnan, Matthew D. Hoffman, and Tony Jebara. "Variational autoencoders for collaborative filtering." \
  In Proceedings of the 2018 World Wide Web Conference on World Wide Web, pp. 689-698.
  """

  def __init__(
    self,
    name="AutoRec",
    k=600,
    n_epochs=100,
    batch_size=256,
    learning_rate=0.001,
    lambda_reg=0.0005,
    trainable=True,
    verbose=False,
    seed=None,
    use_gpu=False):
    Recommender.__init__(self, name=name, trainable=trainable, verbose=verbose)

    self.learning_rate = learning_rate
    self.n_epochs = n_epochs
    self.hidden_dim = k
    self.lambda_reg = lambda_reg
    self.batch_size = batch_size
    self.seed = seed
    self.use_gpu = use_gpu

  def fit(self, train_set, val_set=None):
    """Fit the model to observations.

    Parameters
    ----------
    train_set: :obj:`cornac.data.MultimodalTrainSet`, required
        User-Item preference data as well as additional modalities.

    val_set: :obj:`cornac.data.MultimodalTestSet`, optional, default: None
        User-Item preference data for model selection purposes (e.g., early stopping).

    Returns
    -------
    self : object
    """
    Recommender.fit(self, train_set, val_set)

    import torch
    from .autorec import Autorec, learn

    self.device = (
        torch.device("cuda:0")
        if (self.use_gpu and torch.cuda.is_available())
        else torch.device("cpu")
    )

    if self.trainable:
        if self.seed is not None:
            torch.manual_seed(self.seed)
            torch.cuda.manual_seed(self.seed)

        if not hasattr(self, "autorec"):
            data_dim = train_set.matrix.shape[1]
            self.autorec = Autorec(
                self.hidden_dim,
                data_dim
            ).to(self.device)

        learn(
            self.autorec,
            self.train_set,
            n_epochs=self.n_epochs,
            batch_size=self.batch_size,
            learn_rate=self.learning_rate,
            lambda_reg=self.lambda_reg,
            verbose=self.verbose,
            device=self.device,
        )

    elif self.verbose:
        print("%s is trained already (trainable = False)" % (self.name))

    return self

  def score(self, user_idx, item_idx=None):
    """Predict the scores/ratings of a user for an item.

    Parameters
    ----------
    user_id: int, required
        The index of the user for whom to perform score prediction.

    item_id: int, optional, default: None
        The index of the item for that to perform score prediction.
        If None, scores for all known items will be returned.

    Returns
    -------
    res : A scalar or a Numpy array
        Relative scores that the user gives to the item or to all known items
    """
    import torch

    if item_idx is None:
        if self.train_set.is_unk_user(user_idx):
            raise ScoreException(
                "Can't make score prediction for (user_id=%d)" % user_idx
            )

        x_u = self.train_set.matrix[user_idx].copy()
        z_u = self.autorec.encode(
          torch.tensor(x_u.A, dtype=torch.float32, device=self.device)
        )
        known_item_scores = self.autorec.decode(z_u).data.cpu().numpy().flatten()

        return known_item_scores
    else:
        if self.train_set.is_unk_user(user_idx) or self.train_set.is_unk_item(
            item_idx
        ):
            raise ScoreException(
                "Can't make score prediction for (user_id=%d, item_id=%d)"
                % (user_idx, item_idx)
            )

        x_u = self.train_set.matrix[user_idx].copy()
        z_u = self.autorec.encode(
            torch.tensor(x_u.A, dtype=torch.float32, device=self.device)
        )
        user_pred = (
            self.autorec.decode(z_u).data.cpu().numpy().flatten()[item_idx]
        )

        return user_pred