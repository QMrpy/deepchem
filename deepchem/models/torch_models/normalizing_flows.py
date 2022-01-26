"""
Normalizing flows for transforming probability distributions in PyTorch.
"""

import numpy as np
import logging
from typing import Sequence

import torch
import torch.nn as nn
import torch.distributions as distribution
import torch.distributions.transforms as transform

from deepchem.data import Dataset
from deepchem.models.models import Model
from deepchem.models.optimizers import Adam
from deepchem.utils.typing import OneOrMany
from deepchem.utils.data_utils import load_from_disk, save_to_disk

logger = logging.getLogger(__name__)


class NormalizingFlow(nn.Module):
  """Base class for normalizing flow.

  The purpose of a normalizing flow is to map a simple distribution (that is
  easy to sample from and evaluate probability densities for) to a more
  complex distribution that is learned from data. The base distribution 
  p(x) is transformed by the associated normalizing flow y=g(x) to model the
  distribution p(y).

  Normalizing flows combine the advantages of autoregressive models
  (which provide likelihood estimation but do not learn features) and
  variational autoencoders (which learn feature representations but
  do not provide marginal likelihoods).

  """

  def __init__(self, base_distribution: torch.distributions, 
              flow_layers: Sequence, **kwargs) -> None:
    """Create a new NormalizingFlow.

    Parameters
    ----------
    base_distribution: torch.distributions
      Probability distribution to be transformed.
      Typically an N dimensional multivariate Gaussian.
    flow_layers: Sequence[torch.distributions.transforms]
      An iterable of transforms that comprise the flow.
    **kwargs

    """

    super().__init__()
    self.base_distribution = base_distribution
    self.flow_layers = flow_layers

    self.bijectors = nn.ModuleList(self.flow_layers)
    self.transforms = transform.composeTransform(self.flow_layers)

    self.flow = distribution.TransformedDistribution(
        base_distribution=self.base_distribution, transforms=self.transforms)

  def forward(self, z: OneOrMany[torch.Tensor]):
    """Calls the NormalizingFlow Model.

    Parameters
    ----------
    z: OneOrMany[torch.Tensor]
      A batch of data.

    Returns
    -------
    The transformed input,
    The final density after applying the normalizing flow.

    """

    for bijector in self.bijectors:
      z = bijector(z)

    return z

  
class NormalizingFlowModel(Model):
  """A base distribution and normalizing flow for applying transformations.

  Normalizing flows are effective for any application requiring 
  a probabilistic model that can both sample from a distribution and
  compute marginal likelihoods, e.g. generative modeling,
  unsupervised learning, or probabilistic inference. For a thorough review
  of normalizing flows, see [1]_.

  A distribution implements two main operations:
    1. Sampling from the transformed distribution
    2. Calculating log probabilities

  A normalizing flow implements three main operations:
    1. Forward transformation 
    2. Inverse transformation 
    3. Calculating the Jacobian

  Deep Normalizing Flow models require normalizing flow layers where
  input and output dimensions are the same, the transformation is invertible,
  and the determinant of the Jacobian is efficient to compute and
  differentiable. The determinant of the Jacobian of the transformation 
  gives a factor that preserves the probability volume to 1 when transforming
  between probability densities of different random variables.

  References
  ----------
  [1] Papamakarios, George et al. "Normalizing Flows for Probabilistic Modeling and Inference." (2019). https://arxiv.org/abs/1912.02762.

  """

  def __init__(self, model: NormalizingFlow, **kwargs) -> None:
    """Creates a new NormalizingFlowModel.

    Parameters
    ----------
    model: NormalizingFlow
      An instance of NormalizingFlow.    

    """

    self.model = model
    self.flow = self.model.flow

  def loss(self, flow: torch.distributions, 
          output: OneOrMany[torch.Tensor]) -> torch.Tensor:
    """Creates the negative log likelihood loss function.

    The default implementation is appropriate for most cases. Subclasses can
    override this if there is a need to customize it.

    Parameters
    ----------
    flow: torch.distributions
      Final distribution of normalizing flow.
    output: OneOrMany[torch.Tensor]
      Transformed output after applying normalizing flow to input.

    Returns
    -------
    A Tensor equal to the loss function to use for optimization.

    """

    return -1.0 * torch.mean(torch.sum(self.flow.log_prob(output)))

  def save(self):
    """Saves model to disk using joblib."""

    save_to_disk(self.model, self.get_model_filename(self.model_dir))

  def reload(self):
    """Loads model from joblib file on disk."""

    self.model = load_from_disk(self.get_model_filename(self.model_dir))

  def fit(self, dataset: Dataset):
    inputs = dataset.X
    optimizer = Adam(learning_rate=0.001)._create_pytorch_optimizer(
        self.model.parameters)
    
    running_loss = 0.0
    for i, input in enumerate(inputs):
      optimizer.zero_grad()

      z = self.model(input)
      loss = self.loss(z)
      loss.backward()
      optimizer.step()

      running_loss += loss

      if (i + 1) % 500 == 0:
        print(f'i = {i + 1}, loss = {running_loss / 500}')
        running_loss = 0.0


  