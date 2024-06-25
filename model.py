from utils import generate_incidence

import torch.nn as nn
from topomodelx.nn.hypergraph.allset_transformer import AllSetTransformer
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
import torch.nn.functional as F
import torch
import math
import torch.nn.functional as F
import numpy as np
from ray.rllib.models.preprocessors import get_preprocessor
from ray.rllib.models.modelv2 import restore_original_dimensions


def convert_to_tensor(arr):
    tensor = torch.from_numpy(np.asarray(arr))
    if tensor.dtype == torch.double:
        tensor = tensor.float()
    return tensor


class PositionalEncoding(nn.Module):
    """
    Positional encoding for transformer models.

    Parameters:
    - d_model (int): The embedding dimension.
    - dropout (float): Dropout rate.
    - max_len (int): Maximum length of the input sequence.
    """
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 1000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class Model(TorchModelV2, nn.Module):
    """
    PyTorch model representing a transformer-based policy and value network.

    Parameters:
    - obs_space: Observation space.
    - action_space: Action space.
    - num_outputs: Number of output units.
    - model_config: Model configuration.
    - name: Name of the model.
    - in_dim (int): Dimension of the input embedding.
    - d_model (int): Dimension of the model.
    - nhead (int): Number of attention heads.
    - dim_feedforward (int): Dimension of the feedforward network.
    - n_layers (int): Number of layers in the transformer encoder.
    - dropout (float): Dropout rate.
    - max_len (int): Maximum length of the input sequence.
    """
    def __init__(self, obs_space, action_space, num_outputs, model_config, name, out_dim=3, in_dim=4, d_model=128, nhead=8, dim_feedforward=256, n_layers=4, dropout=0, max_len=64):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)
        self.embed = nn.Embedding(in_dim, d_model)
        # self.embed = nn.Linear(in_dim, d_model)
        encoder = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer=encoder, num_layers=n_layers)
        self.position_encoder = PositionalEncoding(d_model=d_model, dropout=dropout, max_len=max_len)
        self.policy_fc = nn.Linear(d_model, out_dim)
        self.value_fc = nn.Linear(d_model, 1)
        self.preprocessor = get_preprocessor(obs_space.original_space)(
            obs_space.original_space
        )
        self._value_out = None

    def forward(self, input_dict, state, seq_lens):
        x = input_dict["obs"]
        x = self.embed(x.long())
        x = x.permute((1, 0, 2))
        x = self.position_encoder(x)
        x = x.permute((1, 0, 2))
        x = self.encoder(x)
        pooled = torch.mean(x, dim=1)
        out = self.policy_fc(pooled)
        self._value_out = self.value_fc(pooled)
        return out, state

    def value_function(self):
        """
        Computes the value function.

        Returns:
        - torch.Tensor: The flattened value tensor.
        """
        return self._value_out.flatten()

    def compute_priors_and_value(self, obs):
        obs = convert_to_tensor([self.preprocessor.transform(obs)])
        input_dict = restore_original_dimensions(obs, self.obs_space, "torch")

        with torch.no_grad():
            model_out = self.forward(input_dict, None, [1])
            logits, _ = model_out
            value = self.value_function()
            logits, value = torch.squeeze(logits), torch.squeeze(value)
            priors = nn.Softmax(dim=-1)(logits)

            priors = priors.cpu().numpy()
            value = value.cpu().numpy()
            return priors, value



class HyperModel(TorchModelV2, nn.Module):
    """
    A PyTorch neural network model for a specific task.

    Args:
        in_channels (int): Number of input channels.
        hidden_channels (int): Number of hidden channels in the transformer model.
        out_dim (int): Dimension of the output.
        n_layers (int, optional): Number of layers in the transformer model. Defaults to 1.
        heads (int, optional): Number of attention heads in the transformer model. Defaults to 4.
        dropout (float, optional): Dropout probability in the transformer model. Defaults to 0.
        task_level (str, optional): Task level for the model, either "graph" or another task level. Defaults to "graph".

    Attributes:
        model (AllSetTransformer): The underlying transformer model.
        task_level (str): Task level for the model.
        fc (nn.Linear): Fully connected layer for final output.

    Methods:
        forward(x_source, adjacency_matrix): Forward pass of the model.

    """

    def __init__(
        self, obs_space, action_space, num_outputs, model_config, name, out_dim=3, in_dim=4, d_model=128, nhead=8, dim_feedforward=256, n_layers=4, dropout=0, max_len=64, max_per_partition=13
    ):
    
        """
        Initialize the Model with the specified parameters.

        Args:
            in_channels (int): Number of input channels.
            hidden_channels (int): Number of hidden channels in the transformer model.
            out_dim (int): Dimension of the output.
            n_layers (int, optional): Number of layers in the transformer model. Defaults to 1.
            heads (int, optional): Number of attention heads in the transformer model. Defaults to 4.
            dropout (float, optional): Dropout probability in the transformer model. Defaults to 0.
            task_level (str, optional): Task level for the model, either "graph" or another task level. Defaults to "graph".
        """
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)
        self.n_colors = out_dim
        self.embed = nn.Embedding(in_dim, d_model)
        self.position_encoder = PositionalEncoding(d_model=d_model, dropout=dropout, max_len=max_len)
        self.model = AllSetTransformer(
            in_channels=d_model,
            hidden_channels=dim_feedforward,
            n_layers=n_layers,
            heads=nhead,
            dropout=dropout
        )
        self.policy_fc = nn.Linear(dim_feedforward, out_dim)
        self.value_fc = nn.Linear(dim_feedforward, 1)
        self._value_out = None
        self.preprocessor = get_preprocessor(obs_space.original_space)(
            obs_space.original_space
        )
        #self.adjacency_matrix = nn.Parameter(generate_incidence(max_per_partition).to_sparse_coo())
        #self.adjacency_matrix.requires_grad = False
        adjacency_matrix = generate_incidence(max_per_partition)
        self.register_buffer('adjacency_matrix', adjacency_matrix)

    def forward(self, input_dict, state, seq_lens):
        """
        Forward pass of the model.

        Args:
            x_source (torch.Tensor): Input tensor.
            adjacency_matrix (torch.Tensor): Adjacency matrix for the input.

        Returns:
            torch.Tensor: Output tensor.
        """
        # obs = F.one_hot(input_dict["obs"], num_classes=self.n_colors + 1)
        x = input_dict["obs"].long()
        x = self.embed(x)
        x = x.permute((1, 0, 2))
        x = self.position_encoder(x)
        x = x.permute((1, 0, 2))
        x, _ = self.model(x, self.adjacency_matrix.to_sparse_coo())
        pooled = torch.mean(x, dim=1)
        # out = self.policy_fc(x[:, input_dict["max_n"].long(), :])
        out = self.policy_fc(pooled)
        self._value_out = self.value_fc(pooled)
        return out, state

    def value_function(self):
        """
        Computes the value function.

        Returns:
        - torch.Tensor: The flattened value tensor.
        """
        return self._value_out.flatten()

    def compute_priors_and_value(self, obs):
        obs = convert_to_tensor([self.preprocessor.transform(obs)])
        input_dict = restore_original_dimensions(obs, self.obs_space, "torch")

        with torch.no_grad():
            model_out = self.forward(input_dict, None, [1])
            logits, _ = model_out
            value = self.value_function()
            logits, value = torch.squeeze(logits), torch.squeeze(value)
            priors = nn.Softmax(dim=-1)(logits)

            priors = priors.cpu().numpy()
            value = value.cpu().numpy()

            return priors, value
