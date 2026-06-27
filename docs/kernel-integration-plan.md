# Kernel Integration Plan

## Current Kernel Assets

The project already contains two groups of kernel experiments.

### CUDA Basics

Location: `cuda基础/`

- `Vector Add/add.cu`
- `Reductioon/reduction.cu`
- `Reductioon/reduction_v2.cu`
- `Softmax/softmax.cu`
- `GEMM/gemm.cu`
- `FlashAttention/flashattn.cu`

These files are useful as low-level CUDA learning evidence. They should be kept as a separate kernel-learning track unless a specific kernel is integrated into the inference engine.

### Triton Kernel Zoo

Location: `triton-kernel-zoo/kernels/`

- `01_vector_add/vector_add.py`
- `02_softmax/softmax.py`
- `03_layernorm/layernorm.py`
- `03_layernorm/RMSNorm.py`
- `04_gemm/GEMM.py`
- `05_falsh_prefill/FlashAttention Prefill.py`
- `06_Paged Decode Attention/Paged Attention.py`
- `07_Fused RoPE/RoPE.py`
- `08_Fused SiLU-Mul/silu.py`
- `09_W4A量化GEMM/w4a16.py`

These are closer to TurboInfer's final direction because they map naturally to LLM inference operations.

## Integration Principle

Do not integrate kernels just because they exist. Each kernel should enter TurboInfer only when it answers a specific performance question.

Every integrated kernel should have:

- A PyTorch reference implementation
- Correctness tests
- A benchmark script
- A short report explaining when it helps and when it does not
- A clear connection to an inference bottleneck

## Recommended Order

### Stage 1: KV Cache

Use PyTorch/Hugging Face model execution first. Do not introduce custom kernels yet.

Goal:

- Prove that caching key/value states reduces decode cost.
- Compare `naive_no_kv_cache` with `hf_kv_cache`.

### Stage 2: RMSNorm or RoPE Triton Kernel

Use one small, explainable Triton kernel first.

Recommended candidates:

- `triton-kernel-zoo/kernels/03_layernorm/RMSNorm.py`
- `triton-kernel-zoo/kernels/07_Fused RoPE/RoPE.py`

Goal:

- Show kernel-level optimization workflow.
- Validate correctness against PyTorch.
- Measure isolated kernel latency before trying full-engine integration.

### Stage 3: Flash Attention Prefill

Candidate:

- `triton-kernel-zoo/kernels/05_falsh_prefill/FlashAttention Prefill.py`

Goal:

- Explain why prefill is attention-heavy.
- Compare standard attention with a tiled/fused attention implementation on controlled tensor shapes.

### Stage 4: Paged Decode Attention

Candidate:

- `triton-kernel-zoo/kernels/06_Paged Decode Attention/Paged Attention.py`

Goal:

- Connect paged KV cache layout with decode attention.
- This should come after the project already has KV cache and block-based cache management concepts.

Integration should advance in layers:

1. isolated paged decode attention kernel benchmark;
2. `PagedKVBuffer` benchmark with real physical K/V tensor storage;
3. controlled single-layer attention benchmark that projects hidden states into Q/K/V before using paged decode attention;
4. model-specific attention integration with RoPE and GQA details.

### Stage 5: W4A16 Quantized GEMM

Candidate:

- `triton-kernel-zoo/kernels/09_W4A量化GEMM/w4a16.py`

Goal:

- Discuss memory bandwidth, weight-only quantization, and the tradeoff between speed, memory, and accuracy.

## GitHub Note

Before pushing to GitHub, decide how to handle `triton-kernel-zoo/`:

1. Vendor it into TurboInfer by removing its nested `.git` metadata.
2. Keep it as a separate repository and add it as a submodule.
3. Keep only selected kernels copied into a future `kernels/` or `experiments/kernels/` directory.

For this project, option 3 is the cleanest: copy only the kernels that are actually used by TurboInfer, and keep the full zoo as personal reference material.
