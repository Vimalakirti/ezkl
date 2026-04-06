# GPT-2 Proof Generation with EZKL

This guide explains how to generate a ZK proof for GPT-2 inference using EZKL, for comparison with zkTransformer.

## Prerequisites

- Python 3.8+
- ~32GB RAM recommended (GPT-2 is large)
- Several hours of compute time

## Installation

```bash
# Create virtual environment (recommended)
python3 -m venv ezkl_env
source ezkl_env/bin/activate

# Install dependencies
pip install ezkl transformers torch onnx

# Verify installation
python3 -c "import ezkl; print('EZKL installed successfully')"
```

## Running the Script

### Option 1: Quick Test (Mock Proof Only)

This runs the full pipeline except actual proof generation. Use this to verify everything works:

```bash
cd /path/to/zkTransformer/ref/ezkl/examples
python3 gpt2_prove.py --mock-only --seq-len 1
```

Expected output: Should complete in a few minutes.

### Option 2: Full Proof Generation

**Warning:** This will take a LONG time (potentially hours or may run out of memory).

```bash
cd /path/to/zkTransformer/ref/ezkl/examples
python3 gpt2_prove.py --seq-len 1
```

### Option 3: Run in Background (Recommended for Server)

```bash
cd /path/to/zkTransformer/ref/ezkl/examples
nohup python3 gpt2_prove.py --seq-len 1 > gpt2_ezkl.log 2>&1 &

# Monitor progress
tail -f gpt2_ezkl.log
```

## Script Options

| Option | Default | Description |
|--------|---------|-------------|
| `--seq-len` | 1 | Input sequence length |
| `--mock-only` | False | Only run mock proof (skip actual proving) |
| `--output-dir` | `gpt2_ezkl_output` | Directory for output files |

## Output Files

After running, check `gpt2_ezkl_output/`:

```
gpt2_ezkl_output/
├── gpt2.onnx           # Exported GPT-2 model
├── input.json          # Input data
├── settings.json       # EZKL circuit settings
├── network.compiled    # Compiled circuit
├── witness.json        # Witness data
├── proving.key         # Proving key
├── verifying.key       # Verifying key
├── proof.json          # Generated proof
└── timings.json        # Timing measurements (IMPORTANT)
```

## Expected Results

The `timings.json` file will contain:

```json
{
  "timings": {
    "gen_settings": <seconds>,
    "calibrate": <seconds>,
    "compile": <seconds>,
    "get_srs": <seconds>,
    "gen_witness": <seconds>,
    "mock": <seconds>,
    "setup": <seconds>,
    "prove": <seconds>,        // <-- PROVER TIME
    "verify": <seconds>,       // <-- VERIFIER TIME
    "proof_size_kb": <KB>      // <-- PROOF SIZE
  },
  "total_time": <seconds>,
  "seq_len": 1
}
```

## Comparison with zkTransformer

| Metric | zkTransformer | EZKL (fill in) |
|--------|---------------|----------------|
| Prover time | 19.4s | ___ |
| Verifier time | 0.21s | ___ |
| Proof size | 2.17 MB | ___ |

## Troubleshooting

### Out of Memory

If the script runs out of memory, try reducing the model size by editing `gpt2_prove.py`:

```python
# Change this line in export_gpt2_to_onnx():
config = GPT2Config(
    vocab_size=50257,
    n_positions=1024,
    n_embd=768,
    n_layer=2,      # <-- Reduce from 12 to 2
    n_head=12,
    use_cache=False,
)
```

### EZKL Errors

If EZKL fails during calibration or compilation:

1. Check EZKL version: `pip show ezkl`
2. Try updating: `pip install --upgrade ezkl`
3. Check the [EZKL GitHub issues](https://github.com/zkonduit/ezkl/issues)

### Timeout

If the script takes too long (>24 hours), document this as:
> "EZKL could not complete GPT-2 proof generation within 24 hours"

This is a valid result for the paper comparison.

## For the Paper

After running, update `tex/experiment.tex` with the results:

```latex
% Replace XX with actual value or failure message
EZKL required \textbf{XX} hours to prove GPT-2 inference
```

If EZKL fails or times out, change to:
```latex
EZKL could not complete GPT-2 proof generation within 24 hours due to memory/time constraints,
highlighting the scalability challenges of PLONK-based systems for large transformer models.
```
