import torch
from torch import nn

from pytorch_forecasting.models.base_model import BaseModelWithCovariates
from pytorch_forecasting.models.nn.embeddings import MultiEmbedding

from pytorch_forecasting.metrics import MAE, MAPE, MASE, RMSE, SMAPE


class DecoderTransformer(BaseModelWithCovariates):

	def __init__(
			self,

			d_model: int = 512,
			nhead: int = 8,
			dim_feedforward: int = 2048,
			num_encoder_layers: int = 6,
			activation: str = "relu",

			prediction_steps: int = 1,
			autoregressive: bool = False,

			output_size = 1,

			static_categoricals: list = [],
			static_reals: list = [],
			time_varying_categoricals_encoder: list = [],
			time_varying_categoricals_decoder: list = [],
			time_varying_reals_encoder: list = [],
			time_varying_reals_decoder: list = [],
			x_reals: list = [],
			x_categoricals: list = [],

			embedding_sizes: dict = {},
			embedding_labels: dict = None,
			embedding_paddings: list = [],
			categorical_groups: dict = {},

			dropout: float = 0.0,
			weight_decay: float = 1e-3,

			loss = None,
			learning_rate: float = 1e-2,
			reduce_on_plateau_patience: int = 1000,

			log_interval: int = -1,
			log_gradient_flow: bool = False,
			log_val_interval: int = None,
			logging_metrics: nn.ModuleList = None,

			**kwargs,
		):

		assert len(static_categoricals) == 0
		assert len(static_reals) == 0
		assert len(time_varying_categoricals_decoder) == 0
		assert len(time_varying_reals_decoder) == 0

		if loss is None:
			loss = MASE()

		if logging_metrics is None:
			logging_metrics = nn.ModuleList([SMAPE(), MAE(), RMSE(), MAPE(), MASE()])

		self.save_hyperparameters()
		super().__init__(loss=loss, logging_metrics=logging_metrics, **kwargs)

		assert not autoregressive
		assert self.n_targets == 1

		self.embeddings = MultiEmbedding(
			embedding_sizes = self.hparams.embedding_sizes,
			categorical_groups = self.hparams.categorical_groups,
			embedding_paddings = self.hparams.embedding_paddings,
			x_categoricals = self.hparams.x_categoricals,
		)

		input_dim_enc_cat = sum(self.embeddings.output_size[name] for name in self.hparams.time_varying_categoricals_encoder)
		input_dim_enc_real = len(self.hparams.time_varying_reals_encoder)
		self.input_size = input_dim_enc_cat + input_dim_enc_real

		self.output_size = len(self.target_names) * self.n_targets
		if not autoregressive:
			self.output_size = self.output_size * self.hparams.prediction_steps

		self.input_layer = nn.Linear(self.input_size, self.hparams.d_model)

		self.decoder = nn.TransformerDecoder(
			decoder_layer = nn.TransformerDecoderLayer(
				d_model = self.hparams.d_model,
				nhead = self.hparams.nhead,
				dim_feedforward = self.hparams.dim_feedforward,
				dropout = self.hparams.dropout,
				activation = self.hparams.activation,
				batch_first = True,
			),
			num_layers = self.hparams.num_encoder_layers,
			norm = nn.LayerNorm(self.hparams.d_model),
		)
		self.decoder = torch.compile(self.decoder)

		self.output_layer = nn.Linear(self.hparams.d_model, self.output_size)

	def forward(self, x_input):
		x = self.extract_features(x_input)

		x = self.input_layer(x)

		memory = torch.zeros_like(x)
		x = self.decoder(x, memory)

		x = self.output_layer(x)

		# x = x[:, -1, :]
		x = torch.mean(x, dim=1)

		return self.to_network_output(
			prediction=self.transform_output(
				x, target_scale=x_input["target_scale"]
			),
		)

	def extract_features(self, x):
		x_cont = x["encoder_cont"]
		x_cont = {
			name: x_cont[..., idx].unsqueeze(-1)
			for idx, name in enumerate(self.hparams.x_reals)
		}

		x_cat = x["encoder_cat"]
		x_cat = self.embeddings(x_cat)

		x = x_cont | x_cat

		x = torch.concat([x[f] for f in self.encoder_variables], dim=2)
		return x

	@classmethod
	def from_dataset(cls, dataset, **kwargs):
		new_kwargs = cls.deduce_default_output_parameters(dataset, kwargs, MASE())
		kwargs.update(new_kwargs)
		kwargs.update({
			"prediction_steps": dataset.max_prediction_length,
		})
		return super().from_dataset(dataset, **kwargs)
