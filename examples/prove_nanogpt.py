#!/usr/bin/env python3
"""
Prove nanoGPT using EZKL's pre-existing ONNX model.

Uses the nanoGPT model from examples/onnx/nanoGPT/ which is known to work with EZKL.

Model specs (from gen.py):
- block_size: 64 (seq_len)
- vocab_size: 65
- n_layer: 4
- n_head: 4
- n_embd: 64
- Parameters: ~209K

Usage:
  python prove_nanogpt.py --mock-only    # Quick test
  python prove_nanogpt.py                # Full proof

Requirements:
  pip install ezkl
"""

import argparse
import json
import os
import time


def run_ezkl_pipeline(model_path, data_path, output_dir, mock_only=False):
    """Run the complete EZKL proving pipeline."""
    import ezkl

    timings = {}

    # File paths
    settings_path = os.path.join(output_dir, 'settings.json')
    compiled_path = os.path.join(output_dir, 'model.compiled')
    witness_path = os.path.join(output_dir, 'witness.json')
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
    parser = argparse.ArgumentParser(description='Prove nanoGPT with EZKL')
    parser.add_argument('--output-dir', default='nanogpt_proof_output',
                        help='Output directory (default: nanogpt_proof_output)')
    parser.add_argument('--mock-only', action='store_true',
                        help='Only run mock proof (skip full proving)')
    args = parser.parse_args()

    # Paths to EZKL's pre-existing nanoGPT model
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, 'onnx', 'nanoGPT', 'network.onnx')
    data_path = os.path.join(script_dir, 'onnx', 'nanoGPT', 'input.json')

    if not os.path.exists(model_path):
        print(f"ERROR: Model not found at {model_path}")
        print("Make sure you're running from the examples/ directory")
        return

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("nanoGPT Proof Generation with EZKL")
    print("=" * 60)
    print(f"Model: {model_path}")
    print(f"Input: {data_path}")
    print(f"Output directory: {output_dir}")
    print(f"Mock only: {args.mock_only}")
    print()
    print("Model specs (from EZKL examples):")
    print("  - Layers: 4")
    print("  - Embed dim: 64")
    print("  - Heads: 4")
    print("  - Seq len: 64")
    print("  - Vocab size: 65")
    print("  - Parameters: ~209K")
    print("=" * 60)

    # Run EZKL pipeline
    total_start = time.time()
    timings = run_ezkl_pipeline(model_path, data_path, output_dir, args.mock_only)
    total_time = time.time() - total_start

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("Model: nanoGPT (~209K parameters, 4 layers)")

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
    timings['model'] = 'nanoGPT'
    timings['num_params'] = 209000
    timings['num_layers'] = 4
    timings_path = os.path.join(output_dir, 'timings.json')
    with open(timings_path, 'w') as f:
        json.dump(timings, f, indent=2)
    print(f"\nTimings saved to {timings_path}")

    # Comparison note
    print("\n" + "=" * 60)
    print("COMPARISON WITH ZKTRANSFORMER")
    print("=" * 60)
    print("| System        | Model       | Params | Layers | Prove Time |")
    print("|---------------|-------------|--------|--------|------------|")
    print(f"| EZKL          | nanoGPT     | 209K   | 4      | {timings.get('prove', 'N/A')}s |")
    print("| zkTransformer | GPT-2       | 124M   | 12     | 19.4s      |")
    print()
    print("zkTransformer proves a 593x larger model.")
    print("=" * 60)


if __name__ == '__main__':
    main()
