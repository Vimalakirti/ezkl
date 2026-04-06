# nanoGPT Proving with EZKL

This directory contains scripts to prove nanoGPT inference with EZKL for comparison with zkTransformer.

## Why nanoGPT instead of GPT-2?

EZKL cannot handle HuggingFace GPT-2 (124M params) due to tract backend limitations with dynamic shapes in transformer architectures. Instead, we use nanoGPT - a minimal GPT implementation that EZKL can process.

## Model Configurations

| Size   | Parameters | Layers | Embed Dim | Heads | FF Dim |
|--------|------------|--------|-----------|-------|--------|
| tiny   | ~250K      | 4      | 128       | 4     | 512    |
| small  | ~1M        | 8      | 256       | 8     | 1024   |
| medium | ~10M       | 12     | 512       | 8     | 2048   |

Note: GPT-2 small has 124M parameters - approximately 124x larger than nanoGPT-small.

## Installation

```bash
pip install ezkl torch numpy onnx
```

## Usage

### Quick Test (Mock Proof Only)

Test the pipeline without generating a real proof:

```bash
python nanogpt_prove.py --size tiny --mock-only
```

### Full Proof Generation

Generate a real proof (takes longer):

```bash
# Tiny model (~250K params) - fastest
python nanogpt_prove.py --size tiny

# Small model (~1M params) - matches EZKL blog experiments
python nanogpt_prove.py --size small

# Medium model (~10M params) - may require significant resources
python nanogpt_prove.py --size medium
```

### Custom Output Directory

```bash
python nanogpt_prove.py --size small --output-dir my_experiment
```

## Expected Results

Based on EZKL's blog post benchmarks (AMD EPYC, 256GB RAM):

| Model | Parameters | Prove Time | Verify Time |
|-------|------------|------------|-------------|
| tiny  | ~250K      | ~4 min     | ~0.3s       |
| small | ~1M        | ~16 min    | ~0.4s       |

Note: Times vary significantly based on hardware.

## Comparison with zkTransformer

| System        | Model          | Parameters | Prove Time |
|---------------|----------------|------------|------------|
| EZKL          | nanoGPT-small  | ~1M        | ~16 min    |
| zkTransformer | GPT-2          | 124M       | 19.4s      |

zkTransformer proves a 124x larger model in 1/50th the time.

## Output Files

After running, the output directory contains:

```
nanogpt_ezkl_output/
├── nanogpt.onnx      # Exported model
├── input.json        # Sample input
├── settings.json     # EZKL settings
├── model.compiled    # Compiled circuit
├── witness.json      # Witness data
├── proving.key       # Proving key (if full proof)
├── verifying.key     # Verifying key (if full proof)
├── proof.json        # Proof (if full proof)
└── timings.json      # Timing breakdown
```

## Troubleshooting

### Out of Memory

Try a smaller model size or use `--mock-only` to test the pipeline.

### EZKL Not Found

Make sure EZKL is installed: `pip install ezkl`

### Slow Performance

EZKL proof generation is CPU-intensive. Expected times on consumer hardware:
- tiny model: 5-15 minutes
- small model: 15-45 minutes
- medium model: 1-3 hours

## References

- [EZKL nanoGPT Blog Post](https://blog.ezkl.xyz/post/nanogpt/)
- [EZKL Documentation](https://docs.ezkl.xyz/)
- [nanoGPT Original Repository](https://github.com/karpathy/nanoGPT)
