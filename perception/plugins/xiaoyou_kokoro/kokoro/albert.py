"""Inference-only ALBERT for this Kokoro configuration (Transformers Apache-2.0).

Equations and state names follow Transformers 4.53.2 modeling_albert.py.
Only unpadded, absolute-position, single shared-group inference is supported.
"""
import math
from types import SimpleNamespace
import torch
from torch import nn
from torch.nn import functional as F


def AlbertConfig(**kwargs):
    values = dict(embedding_size=128, hidden_size=768, num_attention_heads=12,
                  intermediate_size=2048, num_hidden_layers=12, num_hidden_groups=1,
                  inner_group_num=1, hidden_act='gelu_new', max_position_embeddings=512,
                  type_vocab_size=2, layer_norm_eps=1e-12, pad_token_id=0,
                  position_embedding_type='absolute', hidden_dropout_prob=0,
                  attention_probs_dropout_prob=0)
    values.update(kwargs)
    assert values['num_hidden_groups'] == values['inner_group_num'] == 1
    assert values['position_embedding_type'] == 'absolute' and values['hidden_act'] == 'gelu_new'
    assert values['hidden_size'] % values['num_attention_heads'] == 0
    return SimpleNamespace(**values)


class Embeddings(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.word_embeddings = nn.Embedding(c.vocab_size, c.embedding_size, padding_idx=c.pad_token_id)
        self.position_embeddings = nn.Embedding(c.max_position_embeddings, c.embedding_size)
        self.token_type_embeddings = nn.Embedding(c.type_vocab_size, c.embedding_size)
        self.LayerNorm = nn.LayerNorm(c.embedding_size, eps=c.layer_norm_eps)
        self.dropout = nn.Dropout(c.hidden_dropout_prob)
        self.register_buffer('position_ids', torch.arange(c.max_position_embeddings)[None], persistent=False)

    def forward(self, tokens):
        h = self.word_embeddings(tokens) + self.token_type_embeddings(torch.zeros_like(tokens))
        h += self.position_embeddings(self.position_ids[:, :tokens.shape[1]])
        return self.dropout(self.LayerNorm(h))


class Attention(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.heads, self.width = c.num_attention_heads, c.hidden_size // c.num_attention_heads
        self.query, self.key, self.value = [nn.Linear(c.hidden_size, c.hidden_size) for _ in range(3)]
        self.dense = nn.Linear(c.hidden_size, c.hidden_size)
        self.LayerNorm = nn.LayerNorm(c.hidden_size, eps=c.layer_norm_eps)

    def forward(self, h):
        shape = (*h.shape[:2], self.heads, self.width)
        q, k, v = [layer(h).view(shape).permute(0, 2, 1, 3) for layer in [self.query, self.key, self.value]]
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=0., is_causal=False)
        out = out.transpose(1, 2).reshape(h.shape)
        return self.LayerNorm(h + self.dense(out))


class Layer(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.full_layer_layer_norm = nn.LayerNorm(c.hidden_size, eps=c.layer_norm_eps)
        self.attention = Attention(c)
        self.ffn = nn.Linear(c.hidden_size, c.intermediate_size)
        self.ffn_output = nn.Linear(c.intermediate_size, c.hidden_size)

    def forward(self, h):
        h = self.attention(h)
        x = self.ffn(h)
        x = .5 * x * (1. + torch.tanh(math.sqrt(2. / math.pi) * (x + .044715 * torch.pow(x, 3))))
        return self.full_layer_layer_norm(self.ffn_output(x) + h)


class Group(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.albert_layers = nn.ModuleList([Layer(c)])

    def forward(self, h):
        return self.albert_layers[0](h)


class Encoder(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.embedding_hidden_mapping_in = nn.Linear(c.embedding_size, c.hidden_size)
        self.albert_layer_groups = nn.ModuleList([Group(c)])
        self.depth = c.num_hidden_layers

    def forward(self, h):
        h = self.embedding_hidden_mapping_in(h)
        for _ in range(self.depth):
            h = self.albert_layer_groups[0](h)
        return h


class CustomAlbert(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embeddings = Embeddings(config)
        self.encoder = Encoder(config)
        self.pooler = nn.Linear(config.hidden_size, config.hidden_size)

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, input_ids, attention_mask=None):
        assert not self.training, 'Inference-only ALBERT'
        if attention_mask is not None:
            assert bool(torch.all(attention_mask)), 'Padded batches are unsupported'
        return self.encoder(self.embeddings(input_ids))
