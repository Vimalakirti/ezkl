#!/usr/bin/env python3
"""
Prove nanoGPT (seq_len=1) using EZKL.

Usage:
  python gen_nanogpt_seq1.py --size medium
  python prove_nanogpt_seq1.py --model-dir nanogpt_medium_seq1

Requirements:
  pip install ezkl
"""

import argparse
import json
import os
import time


def run_ezkl_pipeline(model_path, data_path, output_dir, mock_only=False, logrows=None,
                      skip_calibration=False, input_scale=0, param_scale=13):
    """Run the complete EZKL proving pipeline."""
    import ezkl

    timings = {}

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
        # Configure run args - set logrows BEFORE gen_settings for large models
        py_run_args = ezkl.PyRunArgs()
        py_run_args.input_visibility = "private"
        py_run_args.output_visibility = "public"
        py_run_args.param_visibility = "fixed"
        if logrows is not None:
            py_run_args.logrows = logrows
            print(f"  Using logrows={logrows}")
        res = ezkl.gen_settings(model_path, settings_path, py_run_args=py_run_args)
        if not res:
            raise Exception("gen_settings returned False")
        timings['gen_settings'] = time.time() - start
        print(f"  Done in {timings['gen_settings']:.2f}s")
    except Exception as e:
        print(f"ERROR in gen_settings: {e}")
        timings['gen_settings'] = None
        return timings

    # 2. Calibrate settings (or skip and set scales manually)
    if skip_calibration:
        print("\n[2/9] Skipping calibration, setting scales manually...")
        start = time.time()
        try:
            with open(settings_path, 'r') as f:
                settings = json.load(f)
            settings['run_args']['input_scale'] = input_scale
            settings['run_args']['param_scale'] = param_scale
            with open(settings_path, 'w') as f:
                json.dump(settings, f, indent=2)
            timings['calibrate'] = time.time() - start
            print(f"  Set input_scale={input_scale}, param_scale={param_scale}")
        except Exception as e:
            print(f"ERROR setting scales: {e}")
            timings['calibrate'] = None
            return timings
    else:
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
    print("\n[4/9] Getting SRS...")
    start = time.time()
    try:
        import asyncio
        # Use asyncio.run() for Python 3.10+ compatibility
        async def fetch_srs():
            return await ezkl.get_srs(settings_path)
        res = asyncio.run(fetch_srs())
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
        print("\n[MOCK ONLY] Skipping setup, prove, verify")
        return timings

    # 7. Setup
    print("\n[7/9] Setting up keys...")
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
        timings['proof_size_bytes'] = os.path.getsize(proof_path)
        print(f"  Proof size: {timings['proof_size_bytes'] / 1024:.2f} KB")
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
    parser.add_argument('--model-dir', required=True,
                        help='Directory containing network.onnx and input.json')
    parser.add_argument('--mock-only', action='store_true',
                        help='Only run mock proof')
    parser.add_argument('--logrows', type=int, default=None,
                        help='Override logrows in settings (e.g., 20, 22, 24 for larger models)')
    parser.add_argument('--skip-calibration', action='store_true',
                        help='Skip calibration and use provided scales')
    parser.add_argument('--input-scale', type=int, default=0,
                        help='Input scale (default: 0)')
    parser.add_argument('--param-scale', type=int, default=13,
                        help='Parameter scale (default: 13)')
    args = parser.parse_args()

    model_path = os.path.join(args.model_dir, 'network.onnx')
    data_path = os.path.join(args.model_dir, 'input.json')

    if not os.path.exists(model_path):
        print(f"ERROR: {model_path} not found")
        print(f"Run: python gen_nanogpt_seq1.py --size <size> --output-dir {args.model_dir}")
        return

    print("=" * 60)
    print("nanoGPT Proof with EZKL (seq_len=1)")
    print("=" * 60)
    print(f"Model: {model_path}")
    print(f"Mock only: {args.mock_only}")
    print(f"Logrows override: {args.logrows}")
    print(f"Skip calibration: {args.skip_calibration}")
    if args.skip_calibration:
        print(f"  input_scale={args.input_scale}, param_scale={args.param_scale}")
    print("=" * 60)

    total_start = time.time()
    timings = run_ezkl_pipeline(model_path, data_path, args.model_dir, args.mock_only, args.logrows,
                                args.skip_calibration, args.input_scale, args.param_scale)
    total_time = time.time() - total_start

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for step, t in timings.items():
        if t is not None:
            if step == 'proof_size_bytes':
                print(f"  {step}: {t / 1024:.2f} KB")
            else:
                print(f"  {step}: {t:.2f}s")
        else:
            print(f"  {step}: FAILED")
    print(f"\n  TOTAL: {total_time:.2f}s ({total_time/60:.2f} min)")
    print("=" * 60)

    timings['total'] = total_time
    with open(os.path.join(args.model_dir, 'timings.json'), 'w') as f:
        json.dump(timings, f, indent=2)


if __name__ == '__main__':
    main()
