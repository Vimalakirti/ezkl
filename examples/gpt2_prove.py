"""
GPT-2 Small Proof Generation with EZKL

This script exports GPT-2 small to ONNX and generates a ZK proof using EZKL.
Used to compare EZKL's performance with zkTransformer.

Usage:
    python gpt2_prove.py [--seq-len 1] [--mock-only]
"""

import os
import sys
import json
import time
import argparse
import torch
import numpy as np

# Check if ezkl is installed
try:
    import ezkl
except ImportError:
    print("EZKL not found. Install with: pip install ezkl")
    sys.exit(1)

# Check if transformers is installed
try:
    from transformers import GPT2LMHeadModel, GPT2Config
except ImportError:
    print("transformers not found. Install with: pip install transformers")
    sys.exit(1)


def export_gpt2_to_onnx(model_path: str, seq_len: int = 1):
    """Export GPT-2 small to ONNX format."""
    print(f"Loading GPT-2 small model...")

    # Load GPT-2 small (124M parameters)
    # Use a simpler config for testing - the full model might be too large
    config = GPT2Config(
        vocab_size=50257,
        n_positions=1024,
        n_embd=768,
        n_layer=12,
        n_head=12,
        use_cache=False,  # Disable KV cache for ONNX export
    )
    model = GPT2LMHeadModel(config)
    model.eval()

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Create dummy input
    dummy_input = torch.randint(0, 50257, (1, seq_len), dtype=torch.long)

    print(f"Exporting to ONNX with seq_len={seq_len}...")
    start_time = time.time()

    torch.onnx.export(
        model,
        dummy_input,
        model_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=['input_ids'],
        output_names=['logits'],
        dynamic_axes={
            'input_ids': {0: 'batch_size'},
            'logits': {0: 'batch_size'}
        }
    )

    export_time = time.time() - start_time
    file_size = os.path.getsize(model_path) / (1024 * 1024)  # MB
    print(f"ONNX export completed in {export_time:.2f}s, file size: {file_size:.2f} MB")

    return dummy_input


def create_input_json(input_tensor: torch.Tensor, data_path: str):
    """Create input JSON file for EZKL."""
    data_array = input_tensor.detach().numpy().reshape([-1]).tolist()
    data_json = {"input_data": [data_array]}

    with open(data_path, 'w') as f:
        json.dump(data_json, f)
    print(f"Input data saved to {data_path}")


def run_ezkl_pipeline(model_path: str, data_path: str, output_dir: str, mock_only: bool = False):
    """Run the full EZKL proving pipeline."""

    # Define paths
    settings_path = os.path.join(output_dir, 'settings.json')
    compiled_model_path = os.path.join(output_dir, 'network.compiled')
    pk_path = os.path.join(output_dir, 'proving.key')
    vk_path = os.path.join(output_dir, 'verifying.key')
    witness_path = os.path.join(output_dir, 'witness.json')
    proof_path = os.path.join(output_dir, 'proof.json')
    cal_path = os.path.join(output_dir, 'calibration.json')

    timings = {}

    # Step 1: Generate settings
    print("\n[1/8] Generating settings...")
    start = time.time()
    try:
        res = ezkl.gen_settings(model_path, settings_path)
        assert res == True, "gen_settings failed"
    except Exception as e:
        print(f"ERROR in gen_settings: {e}")
        return None
    timings['gen_settings'] = time.time() - start
    print(f"    Done in {timings['gen_settings']:.2f}s")

    # Step 2: Calibrate settings
    print("\n[2/8] Calibrating settings...")
    start = time.time()
    try:
        # Create calibration data
        with open(data_path, 'r') as f:
            input_data = json.load(f)
        cal_data = {"input_data": input_data["input_data"] * 10}  # Replicate for calibration
        with open(cal_path, 'w') as f:
            json.dump(cal_data, f)

        res = ezkl.calibrate_settings(cal_path, model_path, settings_path, "resources")
        assert res == True, "calibrate_settings failed"
    except Exception as e:
        print(f"ERROR in calibrate_settings: {e}")
        return None
    timings['calibrate'] = time.time() - start
    print(f"    Done in {timings['calibrate']:.2f}s")

    # Step 3: Compile circuit
    print("\n[3/8] Compiling circuit...")
    start = time.time()
    try:
        res = ezkl.compile_circuit(model_path, compiled_model_path, settings_path)
        assert res == True, "compile_circuit failed"
    except Exception as e:
        print(f"ERROR in compile_circuit: {e}")
        return None
    timings['compile'] = time.time() - start
    print(f"    Done in {timings['compile']:.2f}s")

    # Step 4: Get SRS
    print("\n[4/8] Downloading/generating SRS...")
    start = time.time()
    try:
        import asyncio
        res = asyncio.get_event_loop().run_until_complete(ezkl.get_srs(settings_path))
    except Exception as e:
        print(f"ERROR in get_srs: {e}")
        return None
    timings['get_srs'] = time.time() - start
    print(f"    Done in {timings['get_srs']:.2f}s")

    # Step 5: Generate witness
    print("\n[5/8] Generating witness...")
    start = time.time()
    try:
        res = ezkl.gen_witness(data_path, compiled_model_path, witness_path)
        assert os.path.isfile(witness_path), "witness file not created"
    except Exception as e:
        print(f"ERROR in gen_witness: {e}")
        return None
    timings['gen_witness'] = time.time() - start
    print(f"    Done in {timings['gen_witness']:.2f}s")

    # Step 6: Mock proof (quick check)
    print("\n[6/8] Running mock proof...")
    start = time.time()
    try:
        res = ezkl.mock(witness_path, compiled_model_path)
        assert res == True, "mock failed"
    except Exception as e:
        print(f"ERROR in mock: {e}")
        return None
    timings['mock'] = time.time() - start
    print(f"    Done in {timings['mock']:.2f}s")

    if mock_only:
        print("\n--mock-only flag set, skipping actual proof generation")
        return timings

    # Step 7: Setup (generate proving/verifying keys)
    print("\n[7/8] Generating proving/verifying keys...")
    start = time.time()
    try:
        res = ezkl.setup(compiled_model_path, vk_path, pk_path)
        assert res == True, "setup failed"
        assert os.path.isfile(vk_path), "vk file not created"
        assert os.path.isfile(pk_path), "pk file not created"
    except Exception as e:
        print(f"ERROR in setup: {e}")
        return None
    timings['setup'] = time.time() - start
    print(f"    Done in {timings['setup']:.2f}s")

    # Step 8: Generate proof
    print("\n[8/8] Generating proof...")
    start = time.time()
    try:
        res = ezkl.prove(witness_path, compiled_model_path, pk_path, proof_path)
        assert os.path.isfile(proof_path), "proof file not created"
    except Exception as e:
        print(f"ERROR in prove: {e}")
        return None
    timings['prove'] = time.time() - start
    print(f"    Done in {timings['prove']:.2f}s")

    # Get proof size
    proof_size = os.path.getsize(proof_path) / 1024  # KB
    timings['proof_size_kb'] = proof_size

    # Step 9: Verify proof
    print("\n[9/9] Verifying proof...")
    start = time.time()
    try:
        res = ezkl.verify(proof_path, settings_path, vk_path)
        assert res == True, "verify failed"
    except Exception as e:
        print(f"ERROR in verify: {e}")
        return None
    timings['verify'] = time.time() - start
    print(f"    Done in {timings['verify']:.2f}s")

    return timings


def main():
    parser = argparse.ArgumentParser(description='Prove GPT-2 inference with EZKL')
    parser.add_argument('--seq-len', type=int, default=1, help='Input sequence length')
    parser.add_argument('--mock-only', action='store_true', help='Only run mock proof (faster)')
    parser.add_argument('--output-dir', type=str, default='gpt2_ezkl_output', help='Output directory')
    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    model_path = os.path.join(args.output_dir, 'gpt2.onnx')
    data_path = os.path.join(args.output_dir, 'input.json')

    print("=" * 60)
    print("GPT-2 Small Proof Generation with EZKL")
    print("=" * 60)
    print(f"Sequence length: {args.seq_len}")
    print(f"Output directory: {args.output_dir}")
    print(f"Mock only: {args.mock_only}")
    print("=" * 60)

    # Step 1: Export GPT-2 to ONNX
    total_start = time.time()
    input_tensor = export_gpt2_to_onnx(model_path, args.seq_len)

    # Step 2: Create input JSON
    create_input_json(input_tensor, data_path)

    # Step 3: Run EZKL pipeline
    timings = run_ezkl_pipeline(model_path, data_path, args.output_dir, args.mock_only)

    total_time = time.time() - total_start

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    if timings:
        print(f"\nTimings:")
        for step, t in timings.items():
            if step == 'proof_size_kb':
                print(f"  Proof size: {t:.2f} KB")
            else:
                print(f"  {step}: {t:.2f}s")

        if 'prove' in timings:
            print(f"\n  PROVER TIME: {timings['prove']:.2f}s")
        if 'verify' in timings:
            print(f"  VERIFIER TIME: {timings['verify']:.2f}s")
        if 'proof_size_kb' in timings:
            print(f"  PROOF SIZE: {timings['proof_size_kb']:.2f} KB")

    print(f"\n  TOTAL TIME: {total_time:.2f}s ({total_time/60:.2f} min)")
    print("=" * 60)

    # Save timings to JSON
    timings_path = os.path.join(args.output_dir, 'timings.json')
    with open(timings_path, 'w') as f:
        json.dump({'timings': timings, 'total_time': total_time, 'seq_len': args.seq_len}, f, indent=2)
    print(f"\nTimings saved to {timings_path}")


if __name__ == '__main__':
    main()
