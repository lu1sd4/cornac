import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import trange

from ...utils import estimate_batches

torch.set_default_dtype(torch.float32)

EPS = 1e-10

class Autorec(nn.Module):
    def __init__(self, hidden_dim, num_items):
        super(Autorec, self).__init__()

        # encoder
        self.encoder = nn.Sequential(
          nn.Linear(num_items, hidden_dim),
          nn.Sigmoid()
        )

        # decoder
        self.decoder = nn.Sequential(
          nn.Linear(hidden_dim, num_items),
          nn.Identity()
        )

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        h = self.decoder(z)
        return h

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def loss(self, y_true, y_pred, mask_input):
        rating = torch.mul(y_true, mask_input)
        preds = torch.mul(y_pred, mask_input)
        mse = F.mse_loss(preds, rating)
        return mse

def learn(
    autorec,
    train_set,
    n_epochs,
    batch_size,
    learn_rate,
    lambda_reg,
    verbose,
    device=torch.device("cpu"),
):
    optimizer = torch.optim.Adam(
      params=autorec.parameters(),
      lr=learn_rate,
      weight_decay=lambda_reg # L2 regularization
    )
    num_steps = estimate_batches(train_set.num_users, batch_size)

    progress_bar = trange(1, n_epochs + 1, disable=not verbose)
    for _ in progress_bar:
        sum_loss = 0.0
        count = 0
        for batch_id, u_ids in enumerate(
            train_set.user_iter(batch_size, shuffle=False)
        ):
            u_batch = train_set.matrix[u_ids, :]
            u_batch = u_batch.A
            u_batch = torch.tensor(u_batch, dtype=torch.float32, device=device)
            mask = torch.where(u_batch > 0, 1, 0)

            # Reconstructed batch
            batch_pred = autorec(u_batch)

            loss = autorec.loss(u_batch, batch_pred, mask)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            sum_loss += loss.data.item()
            count += len(u_batch)

            if batch_id % 10 == 0:
                progress_bar.set_postfix(loss=(sum_loss/count))

    return autorec
