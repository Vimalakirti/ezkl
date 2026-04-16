#!/usr/bin/env python3
"""
Test EZKL with different num_inner_cols settings for GPT-2.

EZKL strategy: increase columns rather than rows to fit large models.
This script tests whether increasing num_inner_cols allows GPT-2 to fit
within Halo2's field size constraints.

Usage:
  python test_gpt2_cols.py --model-dir nanogpt_gpt2_seq1
"""

import argparse
import ezkl
import os
import json


def test_gen_settings(model_path, settings_path, logrows, num_cols):
    """Try gen_settings with specific parameters."""
    print(f"\n=== Testing logrows={logrows}, num_inner_cols={num_cols} ===")
    try:
        py_run_args = ezkl.PyRunArgs()
        py_run_args.input_visibility = "private"
        py_run_args.output_visibility = "public"
        py_run_args.param_visibility = "fixed"
        py_run_args.logrows = logrows
        py_run_args.num_inner_cols = num_cols

        res = ezkl.gen_settings(model_path, settings_path, py_run_args=py_run_args)
        if res:
            print(f"  SUCCESS!")
            with open(settings_path) as f:
                settings = json.load(f)
            print(f"  run_args: {json.dumps(settings.get('run_args', {}), indent=4)}")
            return True
        else:
            print(f"  gen_settings returned False")
            return False
    except Exception as e:
        error_str = str(e)
        if "extended_k" in error_str:
            print(f"  Failed: extended_k constraint - {error_str[:300]}")
        elif "too many values" in error_str:
            print(f"  Failed: too many values to flush (need more rows or columns)")
        else:
            print(f"  Failed: {error_str[:300]}")
        return False


def main():
    parser = argparse.ArgumentParser(description='Test EZKL column settings for GPT-2')
    parser.add_argument('--model-dir', required=True,
                        help='Directory containing network.onnx')
    args = parser.parse_args()

    model_path = os.path.join(args.model_dir, 'network.onnx')
    settings_path = os.path.join(args.model_dir, 'settings_test.json')

    if not os.path.exists(model_path):
        print(f"ERROR: {model_path} not found")
        return

    print("=" * 60)
    print("Testing EZKL num_inner_cols for GPT-2")
    print("=" * 60)
    print(f"Model: {model_path}")
    print("Strategy: Increase columns to reduce row requirements")
    print("=" * 60)

    # Test configurations:
    # - Lower logrows with more columns might avoid extended_k overflow
    # - Default num_inner_cols=2, try higher values

    configs = [
        # (logrows, num_inner_cols)
        # Focus on k=25, try increasing num_inner_cols
        (25, 2),    # baseline (default)
        (25, 4),
        (25, 8),
        (25, 16),
        (25, 32),
        (25, 64),
        (25, 128),
        (25, 256),
    ]

    success = False
    for logrows, num_cols in configs:
        if test_gen_settings(model_path, settings_path, logrows, num_cols):
            success = True
            print(f"\n*** Found working config: logrows={logrows}, num_inner_cols={num_cols} ***")
            break

    if not success:
        print("\n" + "=" * 60)
        print("All configurations failed.")
        print("GPT-2 likely exceeds EZKL/Halo2's fundamental limits.")
        print("=" * 60)


if __name__ == '__main__':
    main()
