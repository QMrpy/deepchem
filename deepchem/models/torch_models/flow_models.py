"""
Implentation of different normalizing flow layers in PyTorch.
"""

import logging

import torch
import torch.nn as nn
import torch.distributions.transforms as transform


class NormalizingFlowLayer(nn.Module, transform.Transform):
  """Base class for normalizing flow layers.

  This is an abstract base class for implementing new normalizing flow
  layers. It should not be called directly.

  A normalizing flow transforms random variables into new random variables.
  Each learnable layer is a bijection, an invertible
  transformation between two probability distributions. A simple initial
  density is pushed through the normalizing flow to produce a richer, 
  more multi-modal distribution. Normalizing flows have three main operations:

  1. Forward
    Transform a distribution. Useful for generating new samples.
  2. Inverse
    Reverse a transformation, useful for computing conditional probabilities.
  3. Log(|det(Jacobian)|) [LDJ]
    Compute the determinant of the Jacobian of the transformation, 
    which is a scaling that conserves the probability "volume" to equal 1. 

  For examples of customized normalizing flows applied to toy problems,
  see [1]_.

  References
  ----------
  .. [1] Saund, Brad. "Normalizing Flows." (2020). https://github.com/bsaund/normalizing_flows.

  Notes
  -----
  - A sequence of normalizing flows is a normalizing flow.
  - The Jacobian is the matrix of first-order derivatives of the transform.

  """

  def __init__(self, **kwargs):
    """Create a new NormalizingFlowLayer."""

    pass

  def _forward(self, x: torch.Tensor) -> torch.Tensor:
    """Forward transformation.

    x = g(y)

    Parameters
    ----------
    x: torch.Tensor
      Input tensor.

    Returns
    -------
    fwd_x: torch.Tensor
      Transformed tensor.

    """

    raise NotImplementedError("Forward transform must be defined.")

  def _inverse(self, y: torch.Tensor) -> torch.Tensor:
    """Inverse transformation.

    x = g^{-1}(y)
    
    Parameters
    ----------
    y: torch.Tensor
      Input tensor.

    Returns
    -------
    inv_y: torch.Tensor
      Inverted tensor.

    """

    raise NotImplementedError("Inverse transform must be defined.")

  def _forward_log_det_jacobian(self, x: torch.Tensor) -> torch.Tensor:
    """Log |Determinant(Jacobian(x)|

    Note x = g^{-1}(y)

    Parameters
    ----------
    x: torch.Tensor
      Input tensor.

    Returns
    -------
    ldj: torch.Tensor
      Log of absolute value of determinant of Jacobian of x.

    """

    raise NotImplementedError("LDJ must be defined.")

  def _inverse_log_det_jacobian(self, y: torch.Tensor) -> torch.Tensor:
    """Inverse LDJ.

    The ILDJ = -LDJ.

    Note x = g^{-1}(y)

    Parameters
    ----------
    y: torch.Tensor
      Input tensor.

    Returns
    -------
    ildj: torch.Tensor
      Log of absolute value of determinant of Jacobian of y.

    """

    return -1.0 * self._forward_log_det_jacobian(self._inverse(y))

  def __hash__(self):
    return nn.Module.__hash__(self)


class PlanarFlow(NormalizingFlowLayer):

  def __init__(self, dim, **kwargs):
    super().__init__(**kwargs)

    self.layer = nn.Linear((dim, 1), bias=True)
    self.scale = nn.Linear((1, dim))
    self.activation = nn.Tanh()

  def _forward(self, z: torch.Tensor) -> torch.Tensor:
    return z + self.scale * self.activation(self.layer(z))

  def _forward_log_det_jacobian(self, z: torch.Tensor) -> torch.Tensor:
    psi = (1 - self.activation(self.layer(z)) ** 2) * self.layer.weight
    det_grad = 1 + torch.mm(psi, torch.transpose(self.scale))

    return torch.log(det_grad.abs() + 1e-9)
