# TurboPairFormer Quickstart

A single executable notebook showing how to use
[**turbopairformer**](https://pypi.org/project/turbopairformer/), developed by
[Lambda](https://lambda.ai), [Stevens Institute of Technology](https://www.stevens.edu), and
[OpenFold](https://openfold.io) — standalone CUDA kernels for the two most expensive
operations in an [AlphaFold3](https://www.nature.com/articles/s41586-024-07487-w)-style pair stack.

| Operation | Function | Replaces |
|---|---|---|
| Triangle Attention | `turbo_attention` | Row/column-wise attention over an `N x N` pair representation |
| Triangle Multiplicative Update | `turbo_trimul` | The outgoing / incoming "TriMul" pair update |

Both are ordinary autograd-aware functions. Neither requires OpenFold3.

**[→ Open `quickstart.ipynb`](quickstart.ipynb)** — it ships with outputs, so you can read
every result without a GPU.

## Background

AlphaFold3 predicts the structure of biomolecular complexes by refining a **pair
representation**: a table holding one feature vector for every pair of residues in the input.
The stack that refines it is the **Pairformer**, and each of its blocks applies the two
operations above. Both enforce the same idea — what the model believes about two residues
should stay consistent with what it believes about each of them and any third residue.
Hence "triangle".

Both scale with the cube of the sequence length, and together they dominate the trunk's time
and memory. TurboPairFormer replaces them with fused CUDA kernels for Hopper GPUs, built for
**training** rather than inference alone: the backward pass is fused, gradients reach every
input including the pair bias, and a DDP consistency check ships with it.

## What the notebook covers

1. Environment check and the pinned install
2. A minimal triangle attention call, with the exact tensor layout contract
3. TriMul against your own module's weights, plus the fail-closed support check
4. Numerical correctness against an FP32 reference — kernel vs. stock BF16 PyTorch
5. Speed and peak-memory benchmarks from `N = 128` to `N = 1024`
6. Masking for padded inputs, checked against FP32 for both operators

## Measured results

Executed on a single [NVIDIA H100 80GB HBM3](https://www.nvidia.com/en-us/data-center/h100/), batch 1, 4 heads, head dimension 32,
BF16 — full tables and methodology are in the notebook.

Triangle attention vs. stock PyTorch:

| N | forward speedup | fwd+bwd speedup | fwd+bwd peak memory |
|---:|---:|---:|---:|
| 256 | 5.0x | 2.8x | 1.18 GB → 0.36 GB |
| 512 | 7.1x | 3.8x | 8.51 GB → 1.59 GB |
| 768 | 7.8x | 4.1x | 28.06 GB → 3.73 GB |
| 1024 | 9.5x | 5.1x | 65.84 GB → 7.39 GB |

TriMul (outgoing, `C_z = C_hidden = 128`) reaches 6.6x forward and 7.0x forward+backward at
`N = 1024`. Accuracy sits at or below the stock BF16 error against an FP32 reference in
every case, masked and unmasked — the kernel is not trading precision for speed.

## Requirements

TurboPairFormer 0.1.0 ships one ahead-of-time compiled wheel, and the loader verifies the
ABI and binary hashes before loading an extension rather than silently recompiling. The
environment has to match:

- Linux x86-64, glibc ≥ 2.28
- CPython 3.14
- PyTorch 2.10.0 with the CUDA 12.8 runtime
- NVIDIA H100 or H200 (Hopper, compute capability 9.0)

Only the first two are yours to arrange. `turbopairformer` pins `torch==2.10.0`, and the
PyPI wheel for that version is already the CUDA 12.8 build, so pip resolves the rest:

```bash
conda create -n turbopairformer python=3.14 pip -y
conda activate turbopairformer

# Put the environment's own pip first, so the install lands in this env.
export PATH="$CONDA_PREFIX/bin:$PATH"
hash -r

pip install turbopairformer

# To re-run the notebook, from the same environment:
pip install jupyterlab
jupyter lab quickstart.ipynb
```

No CUDA toolkit is needed to run the wheel — only to build one; an NVIDIA driver is enough.

## Supported range

The operators are narrower than the environment. Anything outside this range raises before
allocation with a reason string, so a rejected call site is easy to route to a stock
PyTorch fallback:

| | |
|---|---|
| GPU | H100 / H200 (Hopper, compute capability 9.0), one `sm_90a` wheel |
| dtype | Q/K/V BF16; pair bias BF16 or FP32; TriMul weights FP32, `z` BF16 |
| head dimension | `1 <= D <= 64`; multiples of 8 run natively, others zero-pad |
| TriMul channels | `c_z == c_hidden` in `{32, 64, 96, 128}`, `N > 100` |
| batch prefix | any non-empty broadcastable prefix, flattened into one CUDA batch |

## Links

- PyPI: <https://pypi.org/project/turbopairformer/>
- Reference architecture: [OpenFold3 Pairformer](https://github.com/aqlaboratory/openfold-3/blob/main/openfold3/core/model/latent/pairformer.py)

## License

Apache-2.0, matching the package. See [LICENSE](LICENSE).
