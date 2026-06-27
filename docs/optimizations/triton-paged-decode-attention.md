# Triton Paged Decode Attention

## Goal

This benchmark validates the first real attention-path kernel in TurboInfer.

It compares:

- `pytorch_paged_decode_attention`: a readable reference implementation;
- `triton_paged_decode_attention`: a Triton kernel that consumes paged KV metadata.

Unlike earlier paged KV simulations, this benchmark runs an actual decode
attention kernel over:

- `q`: `[batch, num_heads, head_dim]`;
- `k_cache` / `v_cache`: `[num_blocks, num_heads, block_size, head_dim]`;
- `block_table`: `[batch, max_blocks_per_request]`;
- `context_lens`: `[batch]`.

This is the same metadata shape exported by `PagedKVAllocator.decode_metadata()`.

## Run

```bash
python benchmarks/bench_paged_decode_attention.py \
  --batch-sizes 1 4 8 \
  --context-lens 128 512 2048 \
  --num-heads 14 \
  --head-dim 64 \
  --block-size 16 \
  --dtype float16 \
  --warmup 25 \
  --iters 100
```

## Expected Interpretation

The PyTorch reference is not an optimized baseline. It is the correctness oracle.
The key checks are:

- `max_abs_diff` stays within a reasonable fp16 decode-attention tolerance;
- Triton compiles on the target GPU;
- Triton latency improves over the simple PyTorch reference as context and batch grow.

This still does not mean TurboInfer's full model serving path is accelerated.
The next step is to connect the kernel call site to continuous batching's active
decode batch and real KV cache layout.

## AutoDL RTX 3090 Result

The first AutoDL RTX 3090 run completed successfully with:

- GPU: NVIDIA GeForce RTX 3090;
- PyTorch: 2.1.2+cu121;
- CUDA runtime used by PyTorch: 12.1;
- dtype: float16;
- batch sizes: 1, 4, 8;
- context lengths: 128, 512, 2048.

Summary:

- max absolute difference range: `4.77e-7` to `1.22e-4`;
- Triton latency range: `0.0520 ms` to `0.5152 ms`;
- speedup range versus the simple PyTorch reference: `6.23x` to `55.27x`;
- best measured Triton bandwidth estimate: `114.01 GB/s`.

Full report: [../../reports/triton-paged-decode-attention-autodl-3090.md](../../reports/triton-paged-decode-attention-autodl-3090.md).

## Current Boundary

The kernel now validates the paged attention metadata contract, but it is still
an isolated benchmark. The continuous batching engine records paged metadata,
while the real model decode path still uses Hugging Face `past_key_values`.

The next engineering step is to introduce a real paged K/V tensor buffer:

- allocate K/V storage by physical block;
- write prefill and decode token K/V into the correct block slots;
- export the tensors and metadata consumed by `triton_paged_decode_attention`;
- validate the path with a focused integration benchmark before attempting to
  modify a full Hugging Face model attention module.
