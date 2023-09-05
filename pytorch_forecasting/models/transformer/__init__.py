import torch
from torch import nn
import torch.nn.functional as F

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
			target: list = [],

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

		self.target = target
		self.autoregressive = autoregressive
		if self.autoregressive:
			assert len(self.hparams.time_varying_categoricals_encoder) == 0
			assert len(self.hparams.time_varying_categoricals_decoder) == 0
			assert self.hparams.time_varying_reals_decoder == self.target

		self.embeddings = MultiEmbedding(
			embedding_sizes = self.hparams.embedding_sizes,
			categorical_groups = self.hparams.categorical_groups,
			embedding_paddings = self.hparams.embedding_paddings,
			x_categoricals = self.hparams.x_categoricals,
		)

		input_dim_enc_cat = sum(self.embeddings.output_size[name] for name in self.hparams.time_varying_categoricals_encoder)
		input_dim_enc_real = len(self.hparams.time_varying_reals_encoder)
		self.input_size = input_dim_enc_cat + input_dim_enc_real

		self.output_size = self.n_targets
		if not self.autoregressive:
			self.output_size = self.output_size * self.hparams.prediction_steps

		self.input_layer = nn.Linear(self.input_size, self.hparams.d_model)

		# self.decoder = nn.TransformerDecoder(
		# 	decoder_layer = nn.TransformerDecoderLayer(
		# 		d_model = self.hparams.d_model,
		# 		nhead = self.hparams.nhead,
		# 		dim_feedforward = self.hparams.dim_feedforward,
		# 		dropout = self.hparams.dropout,
		# 		activation = self.hparams.activation,
		# 		batch_first = True,
		# 	),
		# 	num_layers = self.hparams.num_encoder_layers,
		# 	norm = nn.LayerNorm(self.hparams.d_model),
		# )

		self.decoder = nn.Sequential(*[Block(d_model, dim_feedforward, nhead, bias=False, dropout=dropout) for _ in range(num_encoder_layers)])

		self.output_layer = nn.Linear(self.hparams.d_model, self.output_size)

		# self.embeddings = torch.compile(self.embeddings)
		# self.input_layer = torch.compile(self.input_layer)
		# self.decoder = torch.compile(self.decoder)
		# self.output_layer = torch.compile(self.output_layer)

	def forward(self, x_input):
		if self.autoregressive:
			return self.forward_autoregressive(x_input)
		else:
			return self.forward_horizon(x_input)

	def forward_horizon(self, x_input):
		x = self.extract_features(x_input)

		x = self.input_layer(x)
		x = self.decoder(x)
		x = x[:, -1, :]
		x = self.output_layer(x)

		y = self.transform_output(x, target_scale=x_input["target_scale"])
		return self.to_network_output(prediction=y)

	def forward_autoregressive(self, x_input):
		if self.training:
			return self.forward_autoregressive_train(x_input)
		else:
			return self.forward_autoregressive_predict(x_input)

	def forward_autoregressive_train(self, x_input):
		xs = self.extract_features_autoregressive(x_input)

		ys = []
		for i in range(xs.shape[0]):
			x = xs[i]
			x = self.input_layer(x)
			x = self.decoder(x)
			x = x[:, -1, :]
			x = self.output_layer(x)
			ys.append(x)

		ys = torch.stack(ys, dim=1)
		ys = ys.unsqueeze(-1)
		ys = [ys[:,:,i,:] for i in range(ys.shape[2])]
		ys = self.transform_output(ys, target_scale=x_input["target_scale"])

		return self.to_network_output(prediction = ys)

	def forward_autoregressive_predict(self, x_input):
		x = self.extract_features(x_input, cat=False)

		for i in range(self.hparams.prediction_steps):
			z = self.input_layer(x)
			z = self.decoder(z)
			z = z[:, -1, :]
			z = self.output_layer(z)

			x = torch.cat([x[:, 1:, :], z.unsqueeze(1)], dim=1)

		y = x[:, -self.hparams.prediction_steps:, :]
		y = y.unsqueeze(-1)
		y = [y[:,:,i,:] for i in range(y.shape[2])]
		y = self.transform_output(y, target_scale=x_input["target_scale"])

		return self.to_network_output(prediction=y)

	def extract_features(self, x, cat=True):
		x_cont = x["encoder_cont"]
		x_cont = {
			name: x_cont[..., idx].unsqueeze(-1)
			for idx, name in enumerate(self.hparams.x_reals)
		}

		if cat:
			x_cat = x["encoder_cat"]
			x_cat = self.embeddings(x_cat)
			x_all = x_cont | x_cat
			x = torch.concat([x_all[f] for f in self.encoder_variables], dim=2)
		else:
			x = torch.concat([x_cont[f] for f in self.hparams.x_reals], dim=2)

		return x

	def extract_features_autoregressive(self, x):
		n_steps = x["decoder_cont"].shape[1]

		x_cont = x["encoder_cont"]
		x_cont = {
			name: x_cont[..., idx].unsqueeze(-1)
			for idx, name in enumerate(self.hparams.x_reals)
		}

		y_cont = x["decoder_cont"]
		y_cont = {
			name: y_cont[..., idx].unsqueeze(-1)
			for idx, name in enumerate(self.hparams.x_reals)
		}

		x_prev = torch.concat([x_cont[f] for f in self.hparams.x_reals], dim=2)
		xs = [x_prev]
		for i in range(n_steps-1):
			y = torch.concat([y_cont[f][:, i:i+1, :] for f in self.hparams.x_reals], dim=2)
			x_prev = torch.concat([x_prev[:, 1:, :], y], dim=1)
			xs.append(x_prev)

		xs = torch.stack(xs, dim=0)
		return xs

	@classmethod
	def from_dataset(cls, dataset, **kwargs):
		new_kwargs = cls.deduce_default_output_parameters(dataset, kwargs, MASE())
		kwargs.update(new_kwargs)
		kwargs.update({"prediction_steps": dataset.max_prediction_length})
		return super().from_dataset(dataset, **kwargs)

class Block(nn.Module):

	def __init__(self, d_model, dim_feedforward, nhead, bias, dropout):
		super().__init__()
		self.ln_1 = nn.LayerNorm(d_model, bias)
		self.attn = CausalSelfAttention(d_model, nhead, bias, dropout)
		self.ln_2 = nn.LayerNorm(d_model, bias)
		self.mlp = MLP(d_model, dim_feedforward, bias, dropout)

	def forward(self, x):
		x = x + self.attn(self.ln_1(x))
		x = x + self.mlp(self.ln_2(x))
		return x

class MLP(nn.Module):

	def __init__(self, d_model, dim_feedforward, bias, dropout):
		super().__init__()
		self.c_fc    = nn.Linear(d_model, dim_feedforward, bias)
		self.gelu    = nn.GELU()
		self.c_proj  = nn.Linear(dim_feedforward, d_model, bias)
		self.dropout = nn.Dropout(dropout)

	def forward(self, x):
		x = self.c_fc(x)
		x = self.gelu(x)
		x = self.c_proj(x)
		x = self.dropout(x)
		return x

class LayerNorm(nn.Module):
	""" LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

	def __init__(self, ndim, bias):
		super().__init__()
		self.weight = nn.Parameter(torch.ones(ndim))
		self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

	def forward(self, input):
		return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)

class CausalSelfAttention(nn.Module):

	def __init__(self, d_model, nhead, bias, dropout):
		super().__init__()
		assert d_model % nhead == 0
		# key, query, value projections for all heads, but in a batch
		self.c_attn = nn.Linear(d_model, 3 * d_model, bias=bias)
		# output projection
		self.c_proj = nn.Linear(d_model, d_model, bias=bias)
		# regularization
		self.attn_dropout = nn.Dropout(dropout)
		self.resid_dropout = nn.Dropout(dropout)
		self.n_head = nhead
		self.n_embd = d_model
		self.dropout = dropout

	def forward(self, x):
		B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

		# calculate query, key, values for all heads in batch and move head forward to be the batch dim
		q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
		k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
		q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
		v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

		# causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
		# efficient attention using Flash Attention CUDA kernels
		y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
		y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

		# output projection
		y = self.resid_dropout(self.c_proj(y))
		return y
