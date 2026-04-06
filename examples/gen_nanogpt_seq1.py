#!/usr/bin/env python3
"""
Generate nanoGPT ONNX with seq_len=1 for fair comparison with zkTransformer.

Based on EZKL's examples/onnx/nanoGPT/gen.py but with seq_len=1 and configurable sizes.

Usage:
  python gen_nanogpt_seq1.py --size tiny     # ~209K params (EZKL default)
  python gen_nanogpt_seq1.py --size small    # ~1.5M params
  python gen_nanogpt_seq1.py --size medium   # ~10M params

Requirements:
  pip install torch onnx
"""

import argparse
import json
import math
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
import os


# Model size configurations
MODEL_CONFIGS = {
    'tiny': {   # EZKL's default nanoGPT (~209K params)
        'n_layer': 4,
        'n_head': 4,
        'n_embd': 64,
        'vocab_size': 65,
    },
    'small': {  # ~1.5M params
        'n_layer': 6,
        'n_head': 6,
        'n_embd': 192,
        'vocab_size': 65,
    },
    'medium': { # ~10M params
        'n_layer': 12,
        'n_head': 8,
        'n_embd': 384,
        'vocab_size': 65,
    },
    'gpt2': {   # GPT-2 small (~124M params)
        'n_layer': 12,
        'n_head': 12,
        'n_embd': 768,
        'vocab_size': 50257,
    },
}


def new_gelu(x):
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x * x * x)))


class LayerNorm(nn.Module):
    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                             .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float(-10))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = new_gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            wpe=nn.Embedding(config.block_size, config.n_embd),
            drop=nn.Dropout(config.dropout),
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f=LayerNorm(config.n_embd, bias=config.bias),
        ))

        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight
        self.block = Block(config)
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

    def get_num_params(self, non_embedding=True):
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size
        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0)

        idx = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        idx = self.transformer.drop(idx + pos_emb)

        for block in self.transformer.h:
            idx = block(idx)

        idx = self.transformer.ln_f(idx)
        idx = self.lm_head(idx)

        return idx


def main():
    parser = argparse.ArgumentParser(description='Generate nanoGPT ONNX with seq_len=1')
    parser.add_argument('--size', choices=['tiny', 'small', 'medium', 'gpt2'], default='tiny',
                        help='Model size (default: tiny)')
    parser.add_argument('--output-dir', default=None,
                        help='Output directory (default: nanogpt_{size}_seq1)')
    args = parser.parse_args()

    cfg = MODEL_CONFIGS[args.size]
    output_dir = args.output_dir or f'nanogpt_{args.size}_seq1'
    os.makedirs(output_dir, exist_ok=True)

    # Create GPT config
    gptconf = GPTConfig(
        block_size=64,   # Max seq len (for position embeddings)
        vocab_size=cfg['vocab_size'],
        n_layer=cfg['n_layer'],
        n_head=cfg['n_head'],
        n_embd=cfg['n_embd'],
        dropout=0.0,
        bias=False
    )

    print(f"Creating nanoGPT-{args.size} model...")
    model = GPT(gptconf)
    model.eval()

    num_params = model.get_num_params()
    print(f"Parameters: {num_params:,}")

    # Export with seq_len=1 for fair comparison
    seq_len = 1
    x = torch.randint(cfg['vocab_size'], (1, seq_len))

    print(f"\nInput shape: {x.shape} (batch=1, seq_len={seq_len})")

    torch_out = model(x)
    print(f"Output shape: {torch_out.shape}")

    # Export to ONNX
    model_path = os.path.join(output_dir, 'network.onnx')
    print(f"\nExporting to {model_path}...")

    torch.onnx.export(
        model,
        x,
        model_path,
        export_params=True,
        opset_version=10,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )

    onnx_size = os.path.getsize(model_path) / 1024
    print(f"ONNX size: {onnx_size:.2f} KB")

    # Save input data
    data_path = os.path.join(output_dir, 'input.json')
    d = x.detach().numpy().reshape([-1]).tolist()
    data = dict(
        input_shapes=[[1, seq_len]],
        input_data=[d],
        output_data=[torch_out.detach().numpy().reshape([-1]).tolist()]
    )
    json.dump(data, open(data_path, 'w'))
    print(f"Input saved to {data_path}")

    print("\n" + "=" * 50)
    print(f"nanoGPT-{args.size} model generated with seq_len=1")
    print("=" * 50)
    print(f"  Layers: {gptconf.n_layer}")
    print(f"  Embed dim: {gptconf.n_embd}")
    print(f"  Heads: {gptconf.n_head}")
    print(f"  Seq len: {seq_len}")
    print(f"  Vocab size: {gptconf.vocab_size}")
    print(f"  Parameters: {num_params:,}")
    print(f"  Output dir: {output_dir}")
    print("=" * 50)
    print(f"\nNow run: python prove_nanogpt_seq1.py --model-dir {output_dir}")


if __name__ == '__main__':
    main()
