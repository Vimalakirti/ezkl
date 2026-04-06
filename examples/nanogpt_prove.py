#!/usr/bin/env python3
"""
nanoGPT Proof Generation with EZKL

Based on EZKL's working little_transformer example.

Usage:
  python nanogpt_prove.py --size tiny --mock-only     # Quick test
  python nanogpt_prove.py --size tiny                 # Full proof
  python nanogpt_prove.py --size medium               # Larger model

Requirements:
  pip install ezkl torch numpy onnx
"""

import argparse
import json
import math
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Model Definition (matching EZKL's little_transformer exactly)
# ============================================================

def attention(queries, keys, values):
    """Simple attention without view/transpose ops for EZKL compatibility."""
    d = queries.shape[-1]
    scores = torch.matmul(queries, keys.transpose(-2, -1)) / math.sqrt(d)
    attention_weights = F.softmax(scores, dim=-1)
    return torch.matmul(attention_weights, values)


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        assert embed_dim % num_heads == 0
        self.projection_dim = embed_dim // num_heads

        self.W_q = nn.Linear(embed_dim, embed_dim)
        self.W_k = nn.Linear(embed_dim, embed_dim)
        self.W_v = nn.Linear(embed_dim, embed_dim)
        self.W_o = nn.Linear(embed_dim, embed_dim)

    def transpose(self, x):
        x = x.reshape(x.shape[0], x.shape[1], self.num_heads, self.projection_dim)
        return x.permute(0, 2, 1, 3)

    def transpose_output(self, x):
        x = x.permute(0, 2, 1, 3)
        return x.reshape(x.shape[0], x.shape[1], self.embed_dim)

    def forward(self, q, k, v):
        q = self.transpose(self.W_q(q))
        k = self.transpose(self.W_k(k))
        v = self.transpose(self.W_v(v))
        output = attention(q, k, v)
        return self.W_o(self.transpose_output(output))


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super().__init__()
        self.att = MultiHeadAttention(embed_dim, num_heads)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, embed_dim)
        )
        self.layernorm1 = nn.LayerNorm(embed_dim)
        self.layernorm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(rate)

    def forward(self, x):
        x = self.layernorm1(x + self.dropout(self.att(x, x, x)))
        x = self.layernorm2(x + self.dropout(self.ffn(x)))
        return x


class TokenAndPositionEmbedding(nn.Module):
    def __init__(self, maxlen, vocab_size, embed_dim):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, embed_dim)
        self.pos_emb = nn.Embedding(maxlen, embed_dim)

    def forward(self, x):
        pos = torch.arange(0, x.size(1), dtype=torch.long, device=x.device)
        return self.token_emb(x) + self.pos_emb(pos).view(1, x.size(1), -1)


class LittleTransformer(nn.Module):
    """
    Transformer model matching EZKL's little_transformer example.
    """

    CONFIGS = {
        'tiny': {
            'seq_len': 1,
            'max_value': 256,
            'layer_count': 4,
            'embed_dim': 128,
            'num_heads': 4,
            'ff_dim': 512,
        },
        'small': {
            'seq_len': 1,
            'max_value': 256,
            'layer_count': 8,
            'embed_dim': 256,
            'num_heads': 8,
            'ff_dim': 1024,
        },
        'medium': {
            'seq_len': 1,
            'max_value': 256,
            'layer_count': 12,
            'embed_dim': 512,
            'num_heads': 8,
            'ff_dim': 2048,
        },
    }

    def __init__(self, config_name='tiny'):
        super().__init__()
        cfg = self.CONFIGS[config_name]
        self.cfg = cfg
        self.max_value = cfg['max_value']
        self.seq_len = cfg['seq_len']

        self.model = nn.Sequential(
            TokenAndPositionEmbedding(cfg['seq_len'], cfg['max_value'], cfg['embed_dim']),
            *[TransformerBlock(cfg['embed_dim'], cfg['num_heads'], cfg['ff_dim'])
              for _ in range(cfg['layer_count'])],
            nn.Linear(cfg['embed_dim'], cfg['max_value']),
            nn.LogSoftmax(dim=-1)
        )

    def forward(self, x):
        return self.model(x)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters())


# ============================================================
# EZKL Pipeline
# ============================================================

def run_ezkl_pipeline(model_path, data_path, output_dir, mock_only=False):
    """Run the complete EZKL proving pipeline."""
    import ezkl

    timings = {}

    # File paths
    settings_path = os.path.join(output_dir, 'settings.json')
    compiled_path = os.path.join(output_dir, 'model.compiled')
    witness_path = os.path.join(output_dir, 'witness.json')
    cal_path = os.path.join(output_dir, 'calibration.json')
    pk_path = os.path.join(output_dir, 'proving.key')
    vk_path = os.path.join(output_dir, 'verifying.key')
    proof_path = os.path.join(output_dir, 'proof.json')

    # 1. Generate settings
    print("\n[1/9] Generating settings...")
    start = time.time()
    try:
        res = ezkl.gen_settings(model_path, settings_path)
        if not res:
            raise Exception("gen_settings returned False")
        timings['gen_settings'] = time.time() - start
        print(f"  Done in {timings['gen_settings']:.2f}s")
    except Exception as e:
        print(f"ERROR in gen_settings: {e}")
        timings['gen_settings'] = None
        return timings

    # 2. Calibrate settings
    print("\n[2/9] Calibrating settings...")
    start = time.time()
    try:
        res = ezkl.calibrate_settings(data_path, model_path, settings_path, "resources")
        if not res:
            raise Exception("calibrate_settings returned False")
        timings['calibrate'] = time.time() - start
        print(f"  Done in {timings['calibrate']:.2f}s")
    except Exception as e:
        print(f"ERROR in calibrate_settings: {e}")
        timings['calibrate'] = None
        return timings

    # 3. Compile circuit
    print("\n[3/9] Compiling circuit...")
    start = time.time()
    try:
        res = ezkl.compile_circuit(model_path, compiled_path, settings_path)
        if not res:
            raise Exception("compile_circuit returned False")
        timings['compile'] = time.time() - start
        print(f"  Done in {timings['compile']:.2f}s")
    except Exception as e:
        print(f"ERROR in compile_circuit: {e}")
        timings['compile'] = None
        return timings

    # 4. Get SRS
    print("\n[4/9] Getting SRS (structured reference string)...")
    start = time.time()
    try:
        import asyncio
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        res = loop.run_until_complete(ezkl.get_srs(settings_path))
        timings['get_srs'] = time.time() - start
        print(f"  Done in {timings['get_srs']:.2f}s")
    except Exception as e:
        print(f"ERROR in get_srs: {e}")
        timings['get_srs'] = None
        return timings

    # 5. Generate witness
    print("\n[5/9] Generating witness...")
    start = time.time()
    try:
        res = ezkl.gen_witness(data_path, compiled_path, witness_path)
        if not os.path.isfile(witness_path):
            raise Exception("Witness file not created")
        timings['gen_witness'] = time.time() - start
        print(f"  Done in {timings['gen_witness']:.2f}s")
    except Exception as e:
        print(f"ERROR in gen_witness: {e}")
        timings['gen_witness'] = None
        return timings

    # 6. Mock proof
    print("\n[6/9] Running mock proof...")
    start = time.time()
    try:
        res = ezkl.mock(witness_path, compiled_path)
        if not res:
            raise Exception("mock returned False")
        timings['mock'] = time.time() - start
        print(f"  Done in {timings['mock']:.2f}s")
    except Exception as e:
        print(f"ERROR in mock: {e}")
        timings['mock'] = None
        return timings

    if mock_only:
        print("\n[MOCK ONLY] Skipping setup, prove, and verify steps")
        timings['setup'] = None
        timings['prove'] = None
        timings['verify'] = None
        return timings

    # 7. Setup
    print("\n[7/9] Setting up proving/verifying keys...")
    start = time.time()
    try:
        res = ezkl.setup(compiled_path, vk_path, pk_path)
        if not res:
            raise Exception("setup returned False")
        timings['setup'] = time.time() - start
        print(f"  Done in {timings['setup']:.2f}s")
    except Exception as e:
        print(f"ERROR in setup: {e}")
        timings['setup'] = None
        return timings

    # 8. Prove
    print("\n[8/9] Generating proof...")
    start = time.time()
    try:
        res = ezkl.prove(witness_path, compiled_path, pk_path, proof_path)
        if not os.path.isfile(proof_path):
            raise Exception("Proof file not created")
        timings['prove'] = time.time() - start
        print(f"  Done in {timings['prove']:.2f}s")

        proof_size = os.path.getsize(proof_path)
        timings['proof_size_bytes'] = proof_size
        print(f"  Proof size: {proof_size / 1024:.2f} KB")
    except Exception as e:
        print(f"ERROR in prove: {e}")
        timings['prove'] = None
        return timings

    # 9. Verify
    print("\n[9/9] Verifying proof...")
    start = time.time()
    try:
        res = ezkl.verify(proof_path, settings_path, vk_path)
        if not res:
            raise Exception("verify returned False")
        timings['verify'] = time.time() - start
        print(f"  Done in {timings['verify']:.2f}s")
    except Exception as e:
        print(f"ERROR in verify: {e}")
        timings['verify'] = None

    return timings


def main():
    parser = argparse.ArgumentParser(description='nanoGPT Proof Generation with EZKL')
    parser.add_argument('--size', choices=['tiny', 'small', 'medium'], default='tiny',
                        help='Model size (default: tiny)')
    parser.add_argument('--output-dir', default='nanogpt_ezkl_output',
                        help='Output directory (default: nanogpt_ezkl_output)')
    parser.add_argument('--mock-only', action='store_true',
                        help='Only run mock proof (skip full proving)')
    args = parser.parse_args()

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("nanoGPT Proof Generation with EZKL")
    print("=" * 60)
    print(f"Model size: {args.size}")
    print(f"Output directory: {output_dir}")
    print(f"Mock only: {args.mock_only}")
    print("=" * 60)

    # Create model
    print(f"\nCreating LittleTransformer-{args.size} model...")
    model = LittleTransformer(args.size)
    model.eval()

    num_params = model.count_parameters()
    print(f"Model parameters: {num_params:,}")
    print(f"Config: {model.cfg}")

    # Create sample input
    seq_len = model.seq_len
    x = torch.zeros((1, seq_len), dtype=torch.long)

    # Export to ONNX (matching EZKL's little_transformer exactly)
    model_path = os.path.join(output_dir, 'network.onnx')
    print(f"\nExporting to ONNX...")
    start = time.time()

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

    export_time = time.time() - start
    onnx_size = os.path.getsize(model_path) / (1024 * 1024)
    print(f"ONNX export completed in {export_time:.2f}s, file size: {onnx_size:.2f} MB")

    # Save input data (matching EZKL's format)
    data_path = os.path.join(output_dir, 'input.json')
    data_array = x.numpy().reshape([-1]).tolist()
    data_json = {'input_data': [data_array]}
    with open(data_path, 'w') as f:
        json.dump(data_json, f)
    print(f"Input data saved to {data_path}")

    # Run EZKL pipeline
    total_start = time.time()
    timings = run_ezkl_pipeline(model_path, data_path, output_dir, args.mock_only)
    total_time = time.time() - total_start

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Model: LittleTransformer-{args.size} ({num_params:,} parameters)")

    for step, t in timings.items():
        if t is not None:
            if step == 'proof_size_bytes':
                print(f"  {step}: {t / 1024:.2f} KB")
            else:
                print(f"  {step}: {t:.2f}s")
        else:
            print(f"  {step}: FAILED")

    print(f"\n  TOTAL TIME: {total_time:.2f}s ({total_time/60:.2f} min)")
    print("=" * 60)

    # Save timings
    timings['total'] = total_time
    timings['model_size'] = args.size
    timings['num_params'] = num_params
    timings_path = os.path.join(output_dir, 'timings.json')
    with open(timings_path, 'w') as f:
        json.dump(timings, f, indent=2)
    print(f"\nTimings saved to {timings_path}")


if __name__ == '__main__':
    main()
