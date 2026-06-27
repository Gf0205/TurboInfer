# Single-Layer Paged Attention Integration

## Goal

This step connects TurboInfer's paged K/V cache path to a controlled transformer
attention data flow.

It does not patch a full Hugging Face model yet. Instead, it validates the
middle layer between raw hidden states and the paged decode attention kernel:

1. project prompt hidden states into K/V;
2. project the current decode hidden state into Q/K/V;
3. write projected K/V into `PagedKVBuffer`;
4. export allocator metadata as `block_table` and `context_lens`;
5. run paged decode attention over the physical K/V blocks.

## Why This Step Exists

Previous steps proved separate pieces:

- `PagedKVAllocator` can manage request block tables;
- `PagedKVBuffer` can store real K/V tensors by physical block;
- `triton_paged_decode_attention` can consume paged metadata and K/V tensors.

This integration step proves that Q/K/V projection output can flow into the
paged K/V path and produce the same decode attention result as a contiguous
reference implementation.

## Run Tests

```bash
python -m pytest tests/test_single_layer_attention.py
```

On AutoDL, run the broader paged path tests:

```bash
python -m pytest \
  tests/test_paged_allocator.py \
  tests/test_paged_decode_attention.py \
  tests/test_paged_kv_buffer.py \
  tests/test_single_layer_attention.py
```

## Benchmark

```bash
python benchmarks/bench_single_layer_paged_attention.py \
  --batch-sizes 1 4 8 \
  --context-lens 128 512 2048 \
  --hidden-size 896 \
  --num-heads 14 \
  --head-dim 64 \
  --block-size 16 \
  --dtype float16 \
  --warmup 10 \
  --iters 50
```

## AutoDL RTX 3090 Result

The first AutoDL RTX 3090 run completed successfully:

- test suite: `19 passed in 3.25s`;
- `paged_triton_attention_ms`: about `0.052 ms` to `0.452 ms`;
- attention-only speedup versus paged PyTorch reference: about `5.96x` to
  `53.47x`;
- best measured paged Triton attention bandwidth estimate: `129.95 GB/s`.

Full report:
[../../reports/single-layer-paged-attention-autodl-3090.md](../../reports/single-layer-paged-attention-autodl-3090.md).

## Expected Interpretation

The benchmark intentionally separates:

- `setup_ms`: Q/K/V projection plus writing K/V into paged storage;
- `contiguous_full_reference_ms`: full contiguous projection plus attention
  reference;
- `paged_pytorch_attention_ms`: paged attention reference only;
- `paged_triton_attention_ms`: Triton paged attention only.

This separation is important because a production attention module would not
rebuild all paged state every decode step. The attention kernel latency should
be interpreted separately from the controlled setup cost.

The correctness checks are:

- `max_abs_diff_paged_ref`: contiguous reference versus paged PyTorch attention;
- `max_abs_diff_triton`: contiguous reference versus paged Triton attention.

## Current Boundary

This still is not an end-to-end model serving acceleration result. It does not
modify a full Hugging Face attention module, does not handle GQA/MQA separately,
and does not implement all model-specific details such as RoPE placement.

The next step after this benchmark is to use the same controlled integration
shape with model-specific Qwen2.5 attention details, especially grouped-query
attention and RoPE, before attempting a full model patch.

The first AutoDL run also exposed a setup bottleneck: prompt K/V was written to
`PagedKVBuffer` through a Python token loop. `PagedKVBuffer.write_tokens` now
writes contiguous spans by physical block. A second AutoDL run confirmed the
effect: the largest benchmark case improved from `667.47 ms` setup time to
`42.22 ms`, while attention kernel latency stayed in the same range.
