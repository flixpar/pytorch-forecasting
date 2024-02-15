from typing import Optional, Tuple, Dict, List, Union

import torch
import torch.nn as nn

from pytorch_forecasting.data import TimeSeriesDataSet
from pytorch_forecasting.metrics import MAE, MAPE, MASE, RMSE, SMAPE, MultiHorizonMetric, MultiLoss, QuantileLoss
from pytorch_forecasting.models.base_model import BaseModelWithCovariates
from pytorch_forecasting.models.tide.sub_modules import _ResidualBlock, RINorm
from pytorch_forecasting.models.nn.embeddings import MultiEmbedding

import numpy as np
from copy import copy



class TiDE(BaseModelWithCovariates):
    def __init__(
        self,
        output_size: Union[int, List[int]] = 1,
        static_categoricals: List[str] = [],
        static_reals: List[str] = [],
        time_varying_categoricals_encoder: List[str] = [],
        time_varying_categoricals_decoder: List[str] = [],
        categorical_groups: Dict[str, List[str]] = {},
        time_varying_reals_encoder: List[str] = [],
        time_varying_reals_decoder: List[str] = [],
        embedding_sizes: Dict[str, Tuple[int, int]] = {},
        embedding_paddings: List[str] = [],
        embedding_labels: Dict[str, np.ndarray] = {},
        x_reals: List[str] = [],
        x_categoricals: List[str] = [],
        context_length: int = 1,
        prediction_length: int = 1,

        num_encoder_layers: int = 1,
        num_decoder_layers: int = 1,
        decoder_output_dim: int = 16,
        hidden_size: int = 128,
        temporal_width: int = 4,
        temporal_decoder_hidden: int = 32,
        use_layer_norm: bool = False,
        use_reversible_instance_norm: bool = False,

        dropout: float = 0.0,
        learning_rate: float = 1e-2,
        log_interval: int = -1,
        log_gradient_flow: bool = False,
        log_val_interval: int = None,
        weight_decay: float = 1e-3,
        loss: MultiHorizonMetric = None,
        reduce_on_plateau_patience: int = 1000,
        logging_metrics: nn.ModuleList = None,
        **kwargs,
    ):
        """
        Initialize TiDE Model - use its :py:meth:`~from_dataset` method if possible.

        Based on the article
        `Long-term Forecasting with TiDE: Time-series Dense Encoder <https://arxiv.org/abs/2304.08424>`_.

        Args:
            loss: loss to optimize. Defaults to MASE(). QuantileLoss is also supported
            output_size: number of outputs (typically number of quantiles for QuantileLoss and one target or list
                of output sizes but currently only point-forecasts allowed). Set automatically.
            static_categoricals: names of static categorical variables
            static_reals: names of static continuous variables
            time_varying_categoricals_encoder: names of categorical variables for encoder
            time_varying_categoricals_decoder: names of categorical variables for decoder
            time_varying_reals_encoder: names of continuous variables for encoder
            time_varying_reals_decoder: names of continuous variables for decoder
            categorical_groups: dictionary where values
                are list of categorical variables that are forming together a new categorical
                variable which is the key in the dictionary
            x_reals: order of continuous variables in tensor passed to forward function
            x_categoricals: order of categorical variables in tensor passed to forward function
            hidden_continuous_size: default for hidden size for processing continous variables (similar to categorical
                embedding size)
            hidden_continuous_sizes: dictionary mapping continuous input indices to sizes for variable selection
                (fallback to hidden_continuous_size if index is not in dictionary)
            embedding_sizes: dictionary mapping (string) indices to tuple of number of categorical classes and
                embedding size
            embedding_paddings: list of indices for embeddings which transform the zero's embedding to a zero vector
            embedding_labels: dictionary mapping (string) indices to list of categorical labels
            learning_rate: learning rate
            log_interval: log predictions every x batches, do not log if 0 or less, log interpretation if > 0. If < 1.0
                , will log multiple entries per batch. Defaults to -1.
            log_val_interval: frequency with which to log validation set metrics, defaults to log_interval
            log_gradient_flow: if to log gradient flow, this takes time and should be only done to diagnose training
                failures
            prediction_length: Length of the prediction. Also known as 'horizon'.
            context_length: Number of time units that condition the predictions. Also known as 'lookback period'.
                Should be between 1-10 times the prediction length.
            backcast_loss_ratio: weight of backcast in comparison to forecast when calculating the loss.
                A weight of 1.0 means that forecast and backcast loss is weighted the same (regardless of backcast and
                forecast lengths). Defaults to 0.0, i.e. no weight.
            log_gradient_flow: if to log gradient flow, this takes time and should be only done to diagnose training
                failures
            reduce_on_plateau_patience (int): patience after which learning rate is reduced by a factor of 10
            logging_metrics (nn.ModuleList[MultiHorizonMetric]): list of metrics that are logged during training.
                Defaults to nn.ModuleList([SMAPE(), MAE(), RMSE(), MAPE(), MASE()])
            **kwargs: additional arguments to :py:class:`~BaseModel`.
        """
        if logging_metrics is None:
            logging_metrics = nn.ModuleList([SMAPE(), MAE(), RMSE(), MAPE(), MASE()])
        if loss is None:
            loss = MASE()

        self.save_hyperparameters()
        super().__init__(loss=loss, logging_metrics=logging_metrics, **kwargs)

        self.embeddings = MultiEmbedding(
            embedding_sizes=self.hparams.embedding_sizes,
            categorical_groups=self.hparams.categorical_groups,
            embedding_paddings=self.hparams.embedding_paddings,
            x_categoricals=self.hparams.x_categoricals,
        )

        input_dim_enc_cat = sum(self.embeddings.output_size[name] for name in self.hparams.time_varying_categoricals_encoder)
        input_dim_dec_cat = sum(self.embeddings.output_size[name] for name in self.hparams.time_varying_categoricals_decoder)
        input_dim_enc_real = len(self.hparams.time_varying_reals_encoder)
        input_dim_dec_real = len(self.hparams.time_varying_reals_decoder)
        input_dim_enc = input_dim_enc_cat + input_dim_enc_real
        input_dim_dec = input_dim_dec_cat + input_dim_dec_real

        input_dim_static_real = len(self.hparams.static_reals)
        input_dim_static_cat = sum(self.embeddings.output_size[name] for name in self.hparams.static_categoricals)
        input_dim_static = input_dim_static_real + input_dim_static_cat

        n_targets = len(self.target_names)

        if isinstance(output_size, list):
            assert all(output_size[i] == output_size[0] for i in range(len(output_size))), "output_size must be a list of equal values"
            output_size = output_size[0]

        self.model = _TideModule(
            input_chunk_length=self.hparams.context_length,
            output_chunk_length=self.hparams.prediction_length,

            input_dim=input_dim_enc,
            output_dim=n_targets,
            future_cov_dim=input_dim_dec,
            static_cov_dim=input_dim_static,
            nr_params=output_size,

            num_encoder_layers=self.hparams.num_encoder_layers,
            num_decoder_layers=self.hparams.num_decoder_layers,
            decoder_output_dim=self.hparams.decoder_output_dim,
            hidden_size=self.hparams.hidden_size,
            temporal_width=self.hparams.temporal_width,
            temporal_decoder_hidden=self.hparams.temporal_decoder_hidden,
            use_layer_norm=self.hparams.use_layer_norm,
            use_reversible_instance_norm=self.hparams.use_reversible_instance_norm,
            dropout=self.hparams.dropout,
        )

    @property
    def decoder_covariate_size(self) -> int:
        """Decoder covariates size.

        Returns:
            int: size of time-dependent covariates used by the decoder
        """
        return len(self.hparams.time_varying_reals_decoder) + sum(
            self.embeddings.output_size[name] for name in self.hparams.time_varying_categoricals_decoder
        )

    @property
    def encoder_covariate_size(self) -> int:
        """Encoder covariate size.

        Returns:
            int: size of time-dependent covariates used by the encoder
        """
        return len(self.hparams.time_varying_reals_encoder) - len(self.target_names) + sum(
            self.embeddings.output_size[name] for name in self.hparams.time_varying_categoricals_encoder
        )

    @property
    def static_size(self) -> int:
        """Static covariate size.

        Returns:
            int: size of static covariates
        """
        return len(self.hparams.static_reals) + sum(
            self.embeddings.output_size[name] for name in self.hparams.static_categoricals
        )

    def forward(self, x: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Pass forward of network.

        Args:
            x (Dict[str, torch.Tensor]): input from dataloader generated from
                :py:class:`~pytorch_forecasting.data.timeseries.TimeSeriesDataSet`.

        Returns:
            Dict[str, torch.Tensor]: output of model
        """
        # covariates
        if self.encoder_covariate_size > 0:
            encoder_features = self.extract_features(x, self.embeddings, period="encoder")
            encoder_only_variables = [name for name in self.encoder_variables if name not in self.decoder_variables and name not in self.target_names]
            encoder_x_t = torch.concat(
                [encoder_features[name] for name in self.target_names]
                + [encoder_features[name] for name in encoder_only_variables]
                + [encoder_features[name] for name in self.decoder_variables],
                dim=2,
            )
        else:
            encoder_x_t = None

        if self.decoder_covariate_size > 0:
            decoder_features = self.extract_features(x, self.embeddings, period="decoder")
            decoder_x_t = torch.concat([decoder_features[name] for name in self.decoder_variables], dim=2)
        else:
            decoder_x_t = None

        # statics
        if self.static_size > 0:
            encoder_features = self.extract_features(x, self.embeddings, period="all")
            x_s = torch.concat([encoder_features[name][:, 0] for name in self.static_variables], dim=1)
        else:
            x_s = None

        # run model
        forecast = self.model((encoder_x_t, decoder_x_t, x_s)) # n_samples x n_timesteps x n_outputs x output_size

        forecast = forecast.permute(2, 0, 1, 3)  # n_outputs x n_samples x n_timesteps x output_size
        if forecast.shape[0] == 1:
            forecast = forecast.squeeze(0)  # n_samples x n_timesteps x output_size
        else:
            forecast = [f.squeeze(0) for f in forecast.split(1, dim=0)]  # n_outputs * (n_samples x n_timesteps x output_size)

        return self.to_network_output(
            prediction=self.transform_output(
                forecast, target_scale=x["target_scale"]
            ),  # (n_outputs x) n_samples x n_timesteps x output_size
        )

    @classmethod
    def from_dataset(
        cls,
        dataset: TimeSeriesDataSet,
        **kwargs,
    ):
        """
        Create model from dataset.

        Args:
            dataset: timeseries dataset
            **kwargs: additional arguments such as hyperparameters for model (see ``__init__()``)

        Returns:
            TiDE
        """
        # add maximum encoder length
        # update defaults
        new_kwargs = copy(kwargs)
        new_kwargs.update(
            {"prediction_length": dataset.max_prediction_length, "context_length": dataset.max_encoder_length}
        )
        new_kwargs.update(cls.deduce_default_output_parameters(dataset, kwargs, QuantileLoss()))

        # create class and return
        return super().from_dataset(
            dataset, **new_kwargs
        )

class _TideModule(nn.Module):
    def __init__(
        self,
        input_chunk_length: int,
        output_chunk_length: int,
        input_dim: int,
        output_dim: int,
        future_cov_dim: int,
        static_cov_dim: int,
        nr_params: int,
        num_encoder_layers: int,
        num_decoder_layers: int,
        decoder_output_dim: int,
        hidden_size: int,
        temporal_decoder_hidden: int,
        temporal_width: int,
        use_layer_norm: bool,
        use_reversible_instance_norm: bool,
        dropout: float,
        **kwargs,
    ):
        """Pytorch module implementing the TiDE architecture.

        Parameters
        ----------
        input_dim
            The number of input components (target + optional past covariates + optional future covariates).
        output_dim
            Number of output components in the target.
        future_cov_dim
            Number of future covariates.
        static_cov_dim
            Number of static covariates.
        nr_params
            The number of parameters of the likelihood (or 1 if no likelihood is used).
        num_encoder_layers
            Number of stacked Residual Blocks in the encoder.
        num_decoder_layers
            Number of stacked Residual Blocks in the decoder.
        decoder_output_dim
            The number of output components of the decoder.
        hidden_size
            The width of the hidden layers in the encoder/decoder Residual Blocks.
        temporal_decoder_hidden
            The width of the hidden layers in the temporal decoder.
        temporal_width
            The width of the future covariate embedding space.
        use_layer_norm
            Whether to use layer normalization in the Residual Blocks.
        use_reversible_instance_norm
            Whether to use reversible instance normalization.
        dropout
            Dropout probability
        **kwargs
            all parameters required for :class:`darts.model.forecasting_models.PLForecastingModule` base class.

        Inputs
        ------
        x
            Tuple of Tensors `(x_past, x_future, x_static)` where `x_past` is the input/past chunk and
            `x_future`is the output/future chunk. Input dimensions are `(batch_size, time_steps, components)`
        Outputs
        -------
        y
            Tensor of shape `(batch_size, output_chunk_length, output_dim, nr_params)`

        """

        super().__init__(**kwargs)

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.future_cov_dim = future_cov_dim
        self.static_cov_dim = static_cov_dim
        self.nr_params = nr_params
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.decoder_output_dim = decoder_output_dim
        self.hidden_size = hidden_size
        self.temporal_decoder_hidden = temporal_decoder_hidden
        self.use_layer_norm = use_layer_norm
        self.use_reversible_instance_norm = use_reversible_instance_norm
        self.dropout = dropout
        self.temporal_width = temporal_width

        self.input_chunk_length = input_chunk_length
        self.output_chunk_length = output_chunk_length

        # residual block for input feature projection
        # this is only needed when covariates are used
        if future_cov_dim:
            self.feature_projection = _ResidualBlock(
                input_dim=future_cov_dim,
                output_dim=temporal_width,
                hidden_size=hidden_size,
                use_layer_norm=use_layer_norm,
                dropout=dropout,
            )
        else:
            self.feature_projection = None

        # original paper doesn't specify how to use past covariates
        # we assume that they pass them raw to the encoder
        historical_future_covariates_flat_dim = (
            self.input_chunk_length + self.output_chunk_length
        ) * (self.temporal_width if future_cov_dim > 0 else 0)
        encoder_dim = (
            self.input_chunk_length * (input_dim - future_cov_dim)
            + historical_future_covariates_flat_dim
            + static_cov_dim
        )

        self.encoders = nn.Sequential(
            _ResidualBlock(
                input_dim=encoder_dim,
                output_dim=hidden_size,
                hidden_size=hidden_size,
                use_layer_norm=use_layer_norm,
                dropout=dropout,
            ),
            *[
                _ResidualBlock(
                    input_dim=hidden_size,
                    output_dim=hidden_size,
                    hidden_size=hidden_size,
                    use_layer_norm=use_layer_norm,
                    dropout=dropout,
                )
                for _ in range(num_encoder_layers - 1)
            ],
        )

        self.decoders = nn.Sequential(
            *[
                _ResidualBlock(
                    input_dim=hidden_size,
                    output_dim=hidden_size,
                    hidden_size=hidden_size,
                    use_layer_norm=use_layer_norm,
                    dropout=dropout,
                )
                for _ in range(num_decoder_layers - 1)
            ],
            # add decoder output layer
            _ResidualBlock(
                input_dim=hidden_size,
                output_dim=decoder_output_dim
                * self.output_chunk_length
                * self.nr_params,
                hidden_size=hidden_size,
                use_layer_norm=use_layer_norm,
                dropout=dropout,
            ),
        )

        self.temporal_decoder = _ResidualBlock(
            input_dim=decoder_output_dim * self.nr_params
            + (temporal_width if future_cov_dim > 0 else 0),
            output_dim=output_dim * self.nr_params,
            hidden_size=temporal_decoder_hidden,
            use_layer_norm=use_layer_norm,
            dropout=dropout,
        )

        self.lookback_skip = nn.Linear(
            self.input_chunk_length, self.output_chunk_length * self.nr_params
        )

        if self.use_reversible_instance_norm:
            self.rin = RINorm(input_dim=output_dim)
        else:
            self.rin = None

    def forward(
        self, x_in: Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]
    ) -> torch.Tensor:
        """TiDE model forward pass.
        Parameters
        ----------
        x_in
            comes as tuple `(x_past, x_future, x_static)` where `x_past` is the input/past chunk and `x_future`
            is the output/future chunk. Input dimensions are `(batch_size, time_steps, components)`
        Returns
        -------
        torch.Tensor
            The output Tensor of shape `(batch_size, output_chunk_length, output_dim, nr_params)`
        """

        # x has shape (batch_size, input_chunk_length, input_dim)
        # x_future_covariates has shape (batch_size, input_chunk_length, future_cov_dim)
        # x_static_covariates has shape (batch_size, static_cov_dim)
        x, x_future_covariates, x_static_covariates = x_in

        if self.use_reversible_instance_norm:
            x[:, :, : self.output_dim] = self.rin(x[:, :, : self.output_dim])

        x_lookback = x[:, :, : self.output_dim]

        # future covariates need to be extracted from x and stacked with historical future covariates
        if self.future_cov_dim > 0:
            x_dynamic_covariates = torch.cat(
                [
                    x_future_covariates,
                    x[
                        :,
                        :,
                        None if self.future_cov_dim == 0 else -self.future_cov_dim :,
                    ],
                ],
                dim=1,
            )

            # project input features across all input time steps
            x_dynamic_covariates_proj = self.feature_projection(x_dynamic_covariates)

        else:
            x_dynamic_covariates = None
            x_dynamic_covariates_proj = None

        # extract past covariates, if they exist
        if self.input_dim - self.output_dim - self.future_cov_dim > 0:
            x_past_covariates = x[
                :,
                :,
                self.output_dim : (None if self.future_cov_dim == 0 else -self.future_cov_dim) :,
            ]
        else:
            x_past_covariates = None

        # setup input to encoder
        encoded = [
            x_lookback,
            x_past_covariates,
            x_dynamic_covariates_proj,
            x_static_covariates,
        ]
        encoded = [t.flatten(start_dim=1) for t in encoded if t is not None]
        encoded = torch.cat(encoded, dim=1)

        # encoder, decode, reshape
        encoded = self.encoders(encoded)
        decoded = self.decoders(encoded)

        # get view that is batch size x output chunk length x self.decoder_output_dim x nr params
        decoded = decoded.view(x.shape[0], self.output_chunk_length, -1)

        # stack and temporally decode with future covariate last output steps
        temporal_decoder_input = [
            decoded,
            x_dynamic_covariates_proj[:, -self.output_chunk_length :, :]
            if self.future_cov_dim > 0
            else None,
        ]
        temporal_decoder_input = [t for t in temporal_decoder_input if t is not None]

        temporal_decoder_input = torch.cat(temporal_decoder_input, dim=2)
        temporal_decoded = self.temporal_decoder(temporal_decoder_input)

        # pass x_lookback through self.lookback_skip but swap the last two dimensions
        # this is needed because the skip connection is applied across the input time steps
        # and not across the output time steps
        skip = self.lookback_skip(x_lookback.transpose(1, 2)).transpose(1, 2)

        # add skip connection
        y = temporal_decoded + skip.reshape_as(
            temporal_decoded
        )  # skip.view(temporal_decoded.shape)

        y = y.view(-1, self.output_chunk_length, self.output_dim, self.nr_params)

        if self.use_reversible_instance_norm:
            y = self.rin.inverse(y)

        return y
