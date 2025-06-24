'''
Implement of FT-Transformer was based on https://github.com/yandex-research/tabular-dl-revisiting-models (https://arxiv.org/abs/2106.11959)
Implement of ResNet was based on https://github.com/csho33/bacteria-ID (https://www.nature.com/articles/s41467-019-12898-9)
'''

import math
import typing as ty

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as nn_init
from torch import Tensor

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms



def reglu(x: Tensor) -> Tensor:
    a, b = x.chunk(2, dim=-1)
    return a * F.relu(b)


def geglu(x: Tensor) -> Tensor:
    a, b = x.chunk(2, dim=-1)
    return a * F.gelu(b)



def get_activation_fn(name: str) -> ty.Callable[[Tensor], Tensor]:
    return (
        reglu
        if name == 'reglu'
        else geglu
        if name == 'geglu'
        else torch.sigmoid
        if name == 'sigmoid'
        else getattr(F, name)
    )


def get_nonglu_activation_fn(name: str) -> ty.Callable[[Tensor], Tensor]:
    return (
        F.relu
        if name == 'reglu'
        else F.gelu
        if name == 'geglu'
        else get_activation_fn(name)
    )




class Tokenizer(nn.Module):
    category_offsets: ty.Optional[Tensor]

    def __init__(
        self,
        d_numerical: int,
        categories: ty.Optional[ty.List[int]],
        d_token: int,
        bias: bool,
    ) -> None:
        super().__init__()
        if categories is None:
            d_bias = d_numerical
            self.category_offsets = None
            self.category_embeddings = None
        else:
            d_bias = d_numerical + len(categories)
            category_offsets = torch.tensor([0] + categories[:-1]).cumsum(0)
            self.register_buffer('category_offsets', category_offsets)
            self.category_embeddings = nn.Embedding(sum(categories), d_token)
            nn_init.kaiming_uniform_(self.category_embeddings.weight, a=math.sqrt(5))
            

        # take [CLS] token into account
        self.weight = nn.Parameter(Tensor(d_numerical + 1, d_token))
        self.bias = nn.Parameter(Tensor(d_bias, d_token)) if bias else None
        # The initialization is inspired by nn.Linear
        nn_init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            nn_init.kaiming_uniform_(self.bias, a=math.sqrt(5))

    @property
    def n_tokens(self) -> int:
        return len(self.weight) + (
            0 if self.category_offsets is None else len(self.category_offsets)
        )

    def forward(self, x_num: Tensor) -> Tensor:
        x_some = x_num 
        x_num = x_num.squeeze(1)
        assert x_some is not None
        #print(torch.ones(len(x_some), 1, device=x_some.device).shape)
        #print(x_num.shape)
        x_num = torch.cat(
            [torch.ones(len(x_some), 1, device=x_some.device)]  # [CLS]
            + [x_num],
            dim=1,
        )
        x = self.weight[None] * x_num[:, :, None]
        if self.bias is not None:
            bias = torch.cat(
                [
                    torch.zeros(1, self.bias.shape[1], device=x.device),
                    self.bias,
                ]
            )
            x = x + bias[None]
        return x



class MultiheadAttention(nn.Module):
    def __init__(
        self, d: int, n_heads: int, dropout: float, initialization: str
    ) -> None:
        if n_heads > 1:
            assert d % n_heads == 0
        assert initialization in ['xavier', 'kaiming']

        super().__init__()
        self.W_q = nn.Linear(d, d)
        self.W_k = nn.Linear(d, d)
        self.W_v = nn.Linear(d, d)
        self.W_out = nn.Linear(d, d) if n_heads > 1 else None
        self.n_heads = n_heads
        self.dropout = nn.Dropout(dropout) if dropout else None

        for m in [self.W_q, self.W_k, self.W_v]:
            if initialization == 'xavier' and (n_heads > 1 or m is not self.W_v):
                # gain is needed since W_qkv is represented with 3 separate layers
                nn_init.xavier_uniform_(m.weight, gain=1 / math.sqrt(2))
            nn_init.zeros_(m.bias)
        if self.W_out is not None:
            nn_init.zeros_(self.W_out.bias)

    def _reshape(self, x: Tensor) -> Tensor:
        batch_size, n_tokens, d = x.shape
        d_head = d // self.n_heads
        return (
            x.reshape(batch_size, n_tokens, self.n_heads, d_head)
            .transpose(1, 2)
            .reshape(batch_size * self.n_heads, n_tokens, d_head)
        )

    def forward(
        self,
        x_q: Tensor,
        x_kv: Tensor,
        key_compression: ty.Optional[nn.Linear],
        value_compression: ty.Optional[nn.Linear],
    ) -> Tensor:
        q, k, v = self.W_q(x_q), self.W_k(x_kv), self.W_v(x_kv)
        for tensor in [q, k, v]:
            assert tensor.shape[-1] % self.n_heads == 0
        if key_compression is not None:
            assert value_compression is not None
            k = key_compression(k.transpose(1, 2)).transpose(1, 2)
            v = value_compression(v.transpose(1, 2)).transpose(1, 2)
        else:
            assert value_compression is None

        batch_size = len(q)
        d_head_key = k.shape[-1] // self.n_heads
        d_head_value = v.shape[-1] // self.n_heads
        n_q_tokens = q.shape[1]

        q = self._reshape(q)
        k = self._reshape(k)
        attention = F.softmax(q @ k.transpose(1, 2) / math.sqrt(d_head_key), dim=-1)
        if self.dropout is not None:
            attention = self.dropout(attention)
        x = attention @ self._reshape(v)
        x = (
            x.reshape(batch_size, self.n_heads, n_q_tokens, d_head_value)
            .transpose(1, 2)
            .reshape(batch_size, n_q_tokens, self.n_heads * d_head_value)
        )
        if self.W_out is not None:
            x = self.W_out(x)
        return x





class SiameseFTT(nn.Module):
    """Feature Tokenizer Transformer, siamese version.

    References:
    - https://pytorch.org/docs/stable/generated/torch.nn.Transformer.html
    - https://github.com/facebookresearch/pytext/tree/master/pytext/models/representations/transformer
    - https://github.com/pytorch/fairseq/blob/1bba712622b8ae4efb3eb793a8a40da386fe11d0/examples/linformer/linformer_src/modules/multihead_linear_attention.py#L19
    """

    def __init__(
        self,
        *,
        # tokenizer
        d_numerical: int,
        categories: ty.Optional[ty.List[int]],
        token_bias: bool,
        # transformer
        n_layers: int,
        d_token: int,
        n_heads: int,
        d_ffn_factor: float,
        attention_dropout: float,
        ffn_dropout: float,
        residual_dropout: float,
        activation: str,
        prenormalization: bool,
        initialization: str,
        # linformer
        kv_compression: ty.Optional[float],
        kv_compression_sharing: ty.Optional[str],
        # embedding
        dim_before: int,
        embedding_size: int,

    ) -> None:
        assert (kv_compression is None) ^ (kv_compression_sharing is not None)

        super().__init__()
        self.tokenizer = Tokenizer(d_numerical, categories, d_token, token_bias)
        n_tokens = self.tokenizer.n_tokens

        def make_kv_compression():
            assert kv_compression
            compression = nn.Linear(
                n_tokens, int(n_tokens * kv_compression), bias=False
            )
            if initialization == 'xavier':
                nn_init.xavier_uniform_(compression.weight)
            return compression

        self.shared_kv_compression = (
            make_kv_compression()
            if kv_compression and kv_compression_sharing == 'layerwise'
            else None
        )

        def make_normalization():
            return nn.LayerNorm(d_token)

        d_hidden = int(d_token * d_ffn_factor)
        self.layers = nn.ModuleList([])
        for layer_idx in range(n_layers):
            layer = nn.ModuleDict(
                {
                    'attention': MultiheadAttention(
                        d_token, n_heads, attention_dropout, initialization
                    ),
                    'linear0': nn.Linear(
                        d_token, d_hidden * (2 if activation.endswith('glu') else 1)
                    ),
                    'linear1': nn.Linear(d_hidden, d_token),
                    'norm1': make_normalization(),
                }
            )
            if not prenormalization or layer_idx:
                layer['norm0'] = make_normalization()
            if kv_compression and self.shared_kv_compression is None:
                layer['key_compression'] = make_kv_compression()
                if kv_compression_sharing == 'headwise':
                    layer['value_compression'] = make_kv_compression()
                else:
                    assert kv_compression_sharing == 'key-value'
            self.layers.append(layer)

        self.activation = get_activation_fn(activation)
        self.last_activation = get_nonglu_activation_fn(activation)
        self.prenormalization = prenormalization
        self.last_normalization = make_normalization() if prenormalization else None
        self.ffn_dropout = ffn_dropout
        self.residual_dropout = residual_dropout
        #self.head = nn.Linear(d_token, d_out)
        #self.full_model = full_model
        self.to_embed = nn.Linear(dim_before, embedding_size) 

    def _get_kv_compressions(self, layer):
        return (
            (self.shared_kv_compression, self.shared_kv_compression)
            if self.shared_kv_compression is not None
            else (layer['key_compression'], layer['value_compression'])
            if 'key_compression' in layer and 'value_compression' in layer
            else (layer['key_compression'], layer['key_compression'])
            if 'key_compression' in layer
            else (None, None)
        )

    def _start_residual(self, x, layer, norm_idx):
        x_residual = x
        if self.prenormalization:
            norm_key = f'norm{norm_idx}'
            if norm_key in layer:
                x_residual = layer[norm_key](x_residual)
        return x_residual

    def _end_residual(self, x, x_residual, layer, norm_idx):
        if self.residual_dropout:
            x_residual = F.dropout(x_residual, self.residual_dropout, self.training)
        x = x + x_residual
        if not self.prenormalization:
            x = layer[f'norm{norm_idx}'](x)
        return x

    def forward_once(self, x_num: Tensor) -> Tensor:
        x = self.tokenizer(x_num)
        #full_model = self.full_model

        for layer_idx, layer in enumerate(self.layers):
            is_last_layer = layer_idx + 1 == len(self.layers)
            layer = ty.cast(ty.Dict[str, nn.Module], layer)

            x_residual = self._start_residual(x, layer, 0)
            x_residual = layer['attention'](
                # for the last attention, it is enough to process only [CLS]
                (x_residual[:, :1] if is_last_layer else x_residual),
                x_residual,
                *self._get_kv_compressions(layer),
            )
            if is_last_layer:
                x = x[:, : x_residual.shape[1]]
            x = self._end_residual(x, x_residual, layer, 0)

            x_residual = self._start_residual(x, layer, 1)
            x_residual = layer['linear0'](x_residual)
            x_residual = self.activation(x_residual)
            if self.ffn_dropout:
                x_residual = F.dropout(x_residual, self.ffn_dropout, self.training)
            x_residual = layer['linear1'](x_residual)
            x = self._end_residual(x, x_residual, layer, 1)

        assert x.shape[1] == 1
        x = x[:, 0]
        if self.last_normalization is not None:
            x = self.last_normalization(x)
        x = self.last_activation(x)
        #if self.full_model:
        #    x = self.head(x)
        #x = x.squeeze(-1)
        x = x.view(x.size(0), -1)
        #print(x.shape)
        x = self.to_embed(x)
        return x

    def forward(self, x_qry, x_ref):
        x_qry = self.forward_once(x_qry)
        x_ref = self.forward_once(x_ref)
        return x_qry, x_ref

    def forward_create_database(self, x_ref):
        feature_ref = self.forward_once(x_ref)
        return feature_ref

    def forward_matching_twins(self, x_qry, feat_db):
        feature_qry = self.forward_once(x_qry)
        dist_list = []
        for feature_ref in feat_db:
            euclidean_distance = F.pairwise_distance(feature_qry, feature_ref)
            dist_list.append(euclidean_distance)
        dist_list = torch.stack(dist_list)
        dist_list = torch.transpose(dist_list, 0, 1)
        return dist_list


        

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()

        # Layers
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=5,
            stride=stride, padding=2, dilation=1, bias=False)
        self.bn1 = nn.BatchNorm1d(num_features=out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=5,
            stride=1, padding=2, dilation=1, bias=False)
        self.bn2 = nn.BatchNorm1d(num_features=out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1,
                    stride=stride, bias=False),
                nn.BatchNorm1d(out_channels))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out




class SiameseResNet(nn.Module):
    def __init__(self, hidden_sizes, num_blocks,
        in_channels=64, embedding_size=128, use_dropout=False, dropout_rate=0.1, dim_before=7600):
        super(SiameseResNet, self).__init__()
        assert len(num_blocks) == len(hidden_sizes)
        self.in_channels = in_channels
        
        self.conv1 = nn.Conv1d(1, self.in_channels, kernel_size=5, stride=1,
            padding=2, bias=False)
        self.bn1 = nn.BatchNorm1d(self.in_channels)

        self.use_dropout = use_dropout
        self.dropout_rate = dropout_rate
        if self.use_dropout:
            self.dropout = nn.Dropout(self.dropout_rate)
        
        # Flexible number of residual encoding layers
        layers = []
        strides = [1] + [2] * (len(hidden_sizes) - 1)
        for idx, hidden_size in enumerate(hidden_sizes):
            layers.append(self._make_layer(hidden_size, num_blocks[idx],
                stride=strides[idx]))
        self.encoder = nn.Sequential(*layers)  # feature map extractor

        self.embedding_size = embedding_size
        self.dim_before = dim_before
        self.linear = nn.Linear(self.dim_before, self.embedding_size)  

    def forward_once(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x

    def forward(self, x_qry, x_ref):
        x_qry = self.forward_once(x_qry)
        x_ref = self.forward_once(x_ref)
        return x_qry, x_ref

    def forward_create_database(self, x_ref):
        feature_ref = self.forward_once(x_ref)
        return feature_ref

    def forward_matching_twins(self, x_qry, feat_db):
        feature_qry = self.forward_once(x_qry)
        dist_list = []
        for feature_ref in feat_db:
            euclidean_distance = F.pairwise_distance(feature_qry, feature_ref)
            dist_list.append(euclidean_distance)
        dist_list = torch.stack(dist_list)
        dist_list = torch.transpose(dist_list, 0, 1)
        return dist_list

    def _make_layer(self, out_channels, num_blocks, stride=1):
        strides = [stride] + [1] * (num_blocks - 1)
        blocks = []
        for stride in strides:
            blocks.append(ResidualBlock(self.in_channels, out_channels,
                stride=stride))
            self.in_channels = out_channels
        return nn.Sequential(*blocks)



class SiameseHybrid(nn.Module):
    def __init__(self, num_morphol=10, embed_size_morphol=128, embed_size_spectra=128, 
        dim_before_morphol=200, dim_before_spectra=7600):
        super(SiameseHybrid, self).__init__()
        self.model_spectra = SiameseResNet(hidden_sizes=[100]*6, num_blocks=[2]*6, 
            in_channels=64, embedding_size=embed_size_spectra, dim_before=dim_before_spectra)
        self.model_morphol = SiameseFTT(d_numerical=num_morphol, categories=None, token_bias=True, 
            n_layers=1, d_token=200, n_heads=8, d_ffn_factor=2, 
            attention_dropout=0.0, ffn_dropout=0.0, residual_dropout=0.0, 
            activation='reglu', prenormalization=True, initialization='kaiming', kv_compression=None, 
            kv_compression_sharing=None, dim_before=dim_before_morphol, embedding_size=embed_size_morphol)
        self.num_morphol = num_morphol

    def forward_once(self, x):
        x_morphol = x.narrow(2, 0, self.num_morphol)
        x_spectra = x.narrow(2, self.num_morphol, x.size()[2]-self.num_morphol)
        x_morphol = self.model_morphol.forward_once(x_morphol)
        x_spectra = self.model_spectra.forward_once(x_spectra)
        x = torch.cat((x_morphol, x_spectra), dim=1)
        return x

    def forward(self, x_qry, x_ref):
        x_qry = self.forward_once(x_qry)
        x_ref = self.forward_once(x_ref)
        return x_qry, x_ref

    def forward_create_database(self, x_ref):
        feature_ref = self.forward_once(x_ref)
        return feature_ref

    def forward_matching_twins(self, x_qry, feat_db):
        feature_qry = self.forward_once(x_qry)
        dist_list = []
        for feature_ref in feat_db:
            euclidean_distance = F.pairwise_distance(feature_qry, feature_ref)
            dist_list.append(euclidean_distance)
        dist_list = torch.stack(dist_list)
        dist_list = torch.transpose(dist_list, 0, 1)
        return dist_list



