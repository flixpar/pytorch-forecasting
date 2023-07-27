import torch
import torch.nn as nn


class _ResidualBlock(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_size: int,
        dropout: float,
        use_layer_norm: bool,
    ):
        """Pytorch module implementing the Residual Block from the TiDE paper."""
        super().__init__()

        # dense layer with ReLU activation with dropout
        self.dense = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_dim),
            nn.Dropout(dropout),
        )

        # linear skip connection from input to output of self.dense
        self.skip = nn.Linear(input_dim, output_dim)

        # layer normalization as output
        if use_layer_norm:
            self.layer_norm = nn.LayerNorm(output_dim)
        else:
            self.layer_norm = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        # residual connection
        x = self.dense(x) + self.skip(x)

        # layer normalization
        if self.layer_norm is not None:
            x = self.layer_norm(x)

        return x
    

class RINorm(nn.Module):
    def __init__(self, input_dim: int, eps=1e-5, affine=True):
        """Reversible Instance Normalization based on [1]

        Parameters
        ----------
        input_dim
            The dimension of the input axis being normalized
        eps
            The epsilon value for numerical stability
        affine
            Whether to apply an affine transformation after normalization

        References
        ----------
        .. [1] Kim et al. "Reversible Instance Normalization for Accurate Time-Series Forecasting against
                Distribution Shift" International Conference on Learning Representations (2022)
        """

        super().__init__()
        self.input_dim = input_dim
        self.eps = eps
        self.affine = affine

        if self.affine:
            self.affine_weight = nn.Parameter(torch.ones(self.input_dim))
            self.affine_bias = nn.Parameter(torch.zeros(self.input_dim))

    def forward(self, x: torch.Tensor):
        # at the beginning of `PLForecastingModule.forward()`, `x` has shape
        # (batch_size, input_chunk_length, n_targets).
        # select all dimensions except batch and input_dim (0, -1)
        # TL;DR: calculate mean and variance over all dimensions except batch and input_dim
        calc_dims = tuple(range(1, x.ndim - 1))

        self.mean = torch.mean(x, dim=calc_dims, keepdim=True).detach()
        self.stdev = torch.sqrt(
            torch.var(x, dim=calc_dims, keepdim=True, unbiased=False) + self.eps
        ).detach()

        x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias

        return x

    def inverse(self, x: torch.Tensor):
        # x is assumed to be the output of PLForecastingModule.forward(), and has shape
        # (batch_size, output_chunk_length, n_targets, nr_params). we ha
        if self.affine:
            x = x - self.affine_bias.view(self.affine_bias.shape + (1,))
            x = x / (
                self.affine_weight.view(self.affine_weight.shape + (1,))
                + self.eps * self.eps
            )
        x = x * self.stdev.view(self.stdev.shape + (1,))
        x = x + self.mean.view(self.mean.shape + (1,))
        return x
