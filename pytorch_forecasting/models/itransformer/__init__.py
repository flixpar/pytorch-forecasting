from typing import Optional, Tuple, Dict, List, Union

import torch
import torch.nn as nn

from pytorch_forecasting.data import TimeSeriesDataSet
from pytorch_forecasting.metrics import MAE, MAPE, MASE, RMSE, SMAPE, MultiHorizonMetric, MultiLoss, QuantileLoss
from pytorch_forecasting.models.base_model import BaseModelWithCovariates
from pytorch_forecasting.models.nn.embeddings import MultiEmbedding

import numpy as np
from copy import copy



class ITransformer(BaseModelWithCovariates):
	def __init__(
		self,

		context_length: int = 1,
		prediction_length: int = 1,
		output_size: Union[int, List[int]] = 1,

		hidden_size: int = 128,
		n_layers: int = 4,
		attention_heads: int = 2,

		dropout: float = 0.0,

		logging_metrics: nn.ModuleList = None,
		loss: MultiHorizonMetric = None,

		**kwargs,
	):
		"""
		Initialize iTransformer Model - use its :py:meth:`~from_dataset` method if possible.

		Based on the article
		`iTransformer: Inverted Transformers Are Effective for Time Series Forecasting <https://arxiv.org/abs/2310.06625>`_.

		Args:
			prediction_length: Length of the prediction. Also known as 'horizon'.
			context_length: Number of time units that condition the predictions. Also known as 'lookback period'.
				Should be between 1-10 times the prediction length.
			output_size: number of outputs (typically number of quantiles for QuantileLoss and one target or list
				of output sizes but currently only point-forecasts allowed). Set automatically.
			hidden_size: The dimension of the encoder hidden state.
			n_layers: Number of layers in the encoder.
			attention_heads: Number of attention heads per layer.
			dropout: Dropout probability
			loss: loss to optimize. Defaults to MASE(). QuantileLoss is also supported
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
		self.input_dim_enc = input_dim_enc_cat + input_dim_enc_real
		self.input_dim_dec = input_dim_dec_cat + input_dim_dec_real

		input_dim_static_real = len(self.hparams.static_reals)
		input_dim_static_cat = sum(self.embeddings.output_size[name] for name in self.hparams.static_categoricals)
		self.input_dim_static = input_dim_static_real + input_dim_static_cat

		n_targets = len(self.target_names)

		if isinstance(output_size, list):
			assert all(output_size[i] == output_size[0] for i in range(len(output_size))), "output_size must be a list of equal values"
			self.output_dim = output_size[0]
		else:
			self.output_dim = output_size

		self.model = ITransformerModel(
			context_length=self.hparams.context_length,
			prediction_length=self.hparams.prediction_length,

			input_dim_enc = self.input_dim_enc,
			input_dim_dec = self.input_dim_dec,
			input_dim_static = self.input_dim_static,
			output_dim = self.output_dim,
			n_targets = n_targets,

			hidden_size=self.hparams.hidden_size,
			n_layers=self.hparams.n_layers,
			n_head=self.hparams.attention_heads,
			dropout=self.hparams.dropout,
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
		if self.input_dim_enc > 0:
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

		if self.input_dim_dec > 0:
			decoder_features = self.extract_features(x, self.embeddings, period="decoder")
			decoder_x_t = torch.concat([decoder_features[name] for name in self.decoder_variables], dim=2)
		else:
			decoder_x_t = None

		# statics
		if self.input_dim_static > 0:
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
			iTransformer
		"""
		new_kwargs = copy(kwargs)
		new_kwargs.update(
			{"prediction_length": dataset.max_prediction_length, "context_length": dataset.max_encoder_length}
		)
		new_kwargs.update(cls.deduce_default_output_parameters(dataset, kwargs))
		return super().from_dataset(
			dataset, **new_kwargs
		)

class ITransformerModel(nn.Module):
	def __init__(
		self,
		context_length: int,
		prediction_length: int,
		input_dim_enc: int,
		input_dim_dec: int,
		input_dim_static: int,
		output_dim: int,
		n_targets: int,

		hidden_size: int,
		n_layers: int,
		n_head: int,
		dropout: float,

		**kwargs,
	):
		"""Pytorch module implementing the iTransformer architecture.

		Parameters
		----------
		context_length
			Number of time steps that condition the predictions.
		prediction_length
			Number of time steps that are predicted.
		input_dim_enc
			Number of input components in the encoder.
		input_dim_dec
			Number of input components in the decoder.
		input_dim_static
			Number of static components in the encoder.
		output_dim
			Number of output components in the target.
		n_targets
			Number of targets.
		hidden_size
			The dimension of the encoder hidden state.
		n_layers
			Number of layers in the encoder.
		n_head
			Number of attention heads per layer.
		dropout
			Dropout probability
		**kwargs
			Additional arguments.

		Inputs
		------
		x
			Tuple of Tensors `(x_past, x_future, x_static)` where `x_past` is the input/past chunk and
			`x_future`is the output/future chunk. Input dimensions are `(batch_size, time_steps, components)`
		Outputs
		-------
		y
			Tensor of shape `(batch_size, prediction_length, n_targets, output_dim)`

		"""

		super().__init__(**kwargs)

		self.context_length = context_length
		self.prediction_length = prediction_length

		self.input_dim_enc = input_dim_enc
		self.input_dim_dec = input_dim_dec
		self.input_dim_static = input_dim_static
		self.output_dim = output_dim
		self.n_targets = n_targets

		self.hidden_size = hidden_size
		self.n_layers = n_layers
		self.n_head = n_head
		self.dropout = dropout

		self.embedding_enc = nn.Linear(self.context_length, self.hidden_size) if self.input_dim_enc > 0 else None
		self.embedding_dec = nn.Linear(self.prediction_length, self.hidden_size) if self.input_dim_dec > 0 else None
		self.embedding_static = nn.Linear(self.input_dim_static, self.hidden_size) if self.input_dim_static > 0 else None

		self.encoder = nn.TransformerEncoder(
			nn.TransformerEncoderLayer(
				d_model=self.hidden_size,
				nhead=self.n_head,
				dim_feedforward=2*self.hidden_size,
				dropout=self.dropout,
				activation="relu",
				batch_first=True,
				norm_first=False,
			),
			num_layers=self.n_layers,
			enable_nested_tensor=False,
		)

		self.decoder_time = nn.Linear(self.hidden_size, self.prediction_length)
		self.decoder_targets = nn.Linear(self.input_dim_enc + self.input_dim_dec + int(self.input_dim_static > 0), self.output_dim * self.n_targets)

	def forward(
		self, x_in: Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]
	) -> torch.Tensor:
		"""iTransformer model forward pass.
		Parameters
		----------
		x_in
			comes as tuple `(x_past, x_future, x_static)` where `x_past` is the input/past chunk and `x_future`
			is the output/future chunk. Input dimensions are `(batch_size, time_steps, components)`
		Returns
		-------
		torch.Tensor
			The output Tensor of shape `(batch_size, prediction_length, n_targets, output_dim)`
		"""

		# x_enc has shape (batch_size, context_length, input_dim_enc)
		# x_dec has shape (batch_size, prediction_length, input_dim_dec)
		# x_static has shape (batch_size, input_dim_static)
		x_enc, x_dec, x_static = x_in

		if self.input_dim_enc > 0:
			x_enc = x_enc.transpose(1, 2)
			x_enc = self.embedding_enc(x_enc)
		else:
			x_enc = torch.zeros(x_dec.shape[0], 0, self.hidden_size, device=x_dec.device)

		if self.input_dim_dec > 0:
			x_dec = x_dec.transpose(1, 2)
			x_dec = self.embedding_dec(x_dec)
		else:
			x_dec = torch.zeros(x_enc.shape[0], 0, self.hidden_size, device=x_enc.device)

		if self.input_dim_static > 0:
			x_static = self.embedding_static(x_static).unsqueeze(1)
		else:
			x_static = torch.zeros(x_enc.shape[0], 0, self.hidden_size, device=x_enc.device)

		x = torch.cat([x_enc, x_dec, x_static], dim=1) # (batch_size, input_dim_enc + input_dim_dec + 1, hidden_size)
		x = self.encoder(x)

		y = x.transpose(1, 2) # (batch_size, hidden_size, input_dim_enc + input_dim_dec + 1)
		y = self.decoder_targets(y) # (batch_size, hidden_size, n_targets * output_dim)

		y = y.transpose(1, 2) # (batch_size, n_targets * output_dim, hidden_size)
		y = self.decoder_time(y) # (batch_size, n_targets * output_dim, prediction_length)

		y = y.transpose(1, 2) # (batch_size, prediction_length, n_targets * output_dim)
		y = y.reshape(y.shape[0], self.prediction_length, self.n_targets, self.output_dim)

		return y
