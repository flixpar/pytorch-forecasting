from typing import Optional, Tuple, Dict, List, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from copy import copy

from pytorch_forecasting.data import TimeSeriesDataSet
from pytorch_forecasting.metrics import MAE, MAPE, MASE, RMSE, SMAPE, MultiHorizonMetric
from pytorch_forecasting.models.base_model import BaseModelWithCovariates
from pytorch_forecasting.models.nn.embeddings import MultiEmbedding


class DLinear(BaseModelWithCovariates):
	def __init__(
		self,

		context_length: int = 1,
		prediction_length: int = 1,
		output_size: Union[int, List[int]] = 1,

		individual: bool = True,
		kernel_size: int = 24,

		logging_metrics: nn.ModuleList = None,
		loss: MultiHorizonMetric = None,

		**kwargs,
	):
		"""
		Initialize DLinear model - use its :py:meth:`~from_dataset` method if possible.

		Args:
			prediction_length: Length of the prediction. Also known as 'horizon'.
			context_length: Number of time units that condition the predictions. Also known as 'lookback period'.
				Should be between 1-10 times the prediction length.
			output_size: number of outputs (typically number of quantiles for QuantileLoss and one target or list
				of output sizes but currently only point-forecasts allowed). Set automatically.
			loss: loss to optimize. Defaults to MASE().
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

		input_dim_cat = sum(self.embeddings.output_size.get(name, 0) for name in self.target_names)
		input_dim_real = len([name for name in self.target_names if name not in self.embeddings.output_size])
		self.input_dim = input_dim_cat + input_dim_real

		if isinstance(output_size, list):
			assert all(output_size[i] == output_size[0] for i in range(len(output_size))), "output_size must be a list of equal values"
			self.output_dim = output_size[0]
		else:
			self.output_dim = output_size
		assert self.output_dim == 1, "only one output allowed for DLinear"

		self.model = DLinearModel(
			context_length=self.hparams.context_length,
			prediction_length=self.hparams.prediction_length,
			input_dim = self.input_dim,

			individual=self.hparams.individual,
			kernel_size=self.hparams.kernel_size,
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
		features = self.extract_features(x, self.embeddings, period="encoder")
		x_input = torch.concat([features[name] for name in self.target_names], dim=2)

		forecast = self.model(x_input) # n_samples x n_timesteps x n_outputs
		forecast = forecast.unsqueeze(-1)  # n_samples x n_timesteps x n_outputs x output_size

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
			DLinear
		"""
		new_kwargs = copy(kwargs)
		new_kwargs.update(
			{"prediction_length": dataset.max_prediction_length, "context_length": dataset.max_encoder_length}
		)
		new_kwargs.update(cls.deduce_default_output_parameters(dataset, kwargs))
		return super().from_dataset(
			dataset, **new_kwargs
		)


class DLinearModel(nn.Module):
	"""
	Decomposition-Linear
	"""
	def __init__(
			self,
			context_length,
			prediction_length,
			input_dim,

			individual=True,
			kernel_size=24,
		):
		super(DLinearModel, self).__init__()
		self.seq_len = context_length
		self.pred_len = prediction_length

		# Decompsition Kernel Size
		self.decompsition = SeriesDecompBlock(kernel_size)
		self.individual = individual
		self.channels = input_dim
		self.kernel_size = kernel_size

		if self.individual:
			self.Linear_Seasonal = nn.ModuleList()
			self.Linear_Trend = nn.ModuleList()

			for i in range(self.channels):
				self.Linear_Seasonal.append(nn.Linear(self.seq_len,self.pred_len))
				self.Linear_Trend.append(nn.Linear(self.seq_len,self.pred_len))

				# Use these two lines if you want to visualize the weights
				# self.Linear_Seasonal[i].weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
				# self.Linear_Trend[i].weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
		else:
			self.Linear_Seasonal = nn.Linear(self.seq_len,self.pred_len)
			self.Linear_Trend = nn.Linear(self.seq_len,self.pred_len)

			# Use these two lines if you want to visualize the weights
			# self.Linear_Seasonal.weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
			# self.Linear_Trend.weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))

	def forward(self, x):
		# x: [Batch, Input length, Channel]
		seasonal_init, trend_init = self.decompsition(x)
		seasonal_init, trend_init = seasonal_init.permute(0,2,1), trend_init.permute(0,2,1)
		if self.individual:
			seasonal_output = torch.zeros([seasonal_init.size(0),seasonal_init.size(1),self.pred_len],dtype=seasonal_init.dtype).to(seasonal_init.device)
			trend_output = torch.zeros([trend_init.size(0),trend_init.size(1),self.pred_len],dtype=trend_init.dtype).to(trend_init.device)
			for i in range(self.channels):
				seasonal_output[:,i,:] = self.Linear_Seasonal[i](seasonal_init[:,i,:])
				trend_output[:,i,:] = self.Linear_Trend[i](trend_init[:,i,:])
		else:
			seasonal_output = self.Linear_Seasonal(seasonal_init)
			trend_output = self.Linear_Trend(trend_init)

		x = seasonal_output + trend_output
		return x.permute(0,2,1) # to [Batch, Output length, Channel]

class MovingAvgBlock(nn.Module):
	"""
	Moving average block to highlight the trend of time series
	"""
	def __init__(self, kernel_size, stride):
		super(MovingAvgBlock, self).__init__()
		self.kernel_size = kernel_size if (kernel_size % 2) else (kernel_size + 1)
		self.avg = nn.AvgPool1d(kernel_size=self.kernel_size, stride=stride, padding=0)

	def forward(self, x):
		# padding on the both ends of time series
		front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
		end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
		x = torch.cat([front, x, end], dim=1)
		x = self.avg(x.permute(0, 2, 1))
		x = x.permute(0, 2, 1)
		return x

class SeriesDecompBlock(nn.Module):
	"""
	Series decomposition block
	"""
	def __init__(self, kernel_size):
		super(SeriesDecompBlock, self).__init__()
		self.moving_avg = MovingAvgBlock(kernel_size, stride=1)

	def forward(self, x):
		moving_mean = self.moving_avg(x)
		res = x - moving_mean
		return res, moving_mean
