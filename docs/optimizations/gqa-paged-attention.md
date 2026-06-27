# GQA-Aware Paged Attention

## Goal

This step moves TurboInfer's controlled paged attention path from equal Q/K/V
heads to grouped-query attention.

This matters because Qwen-family small models use GQA:

- Qwen2.5-0.5B profile: `q_heads=14`, `kv_heads=2`, `head_dim=64`;
- Qwen3-0.6B profile: `q_heads=16`, `kv_heads=8`, `head_dim=128`.

The key mapping is:

```text
kv_head = q_head // (num_q_heads / num_kv_heads)
```

## What Changed

TurboInfer now has:

- model shape profiles in `src/turboinfer/model_profiles.py`;
- `pytorch_paged_decode_attention_gqa` as the correctness reference;
- `triton_paged_decode_attention_gqa` as the Triton kernel path;
- single-layer attention support for separate Q heads and KV heads;
- benchmark support for `--profile qwen2.5-0.5b` and `--profile qwen3-0.6b`.

## Run

Qwen2.5-0.5B-shaped benchmark:

```bash
python benchmarks/bench_single_layer_paged_attention.py \
  --profile qwen2.5-0.5b \
  --batch-sizes 1 4 8 \
  --context-lens 128 512 2048 \
  --dtype float16 \
  --warmup 10 \
  --iters 50
```

Qwen3-0.6B-shaped benchmark:

```bash
python benchmarks/bench_single_layer_paged_attention.py \
  --profile qwen3-0.6b \
  --batch-sizes 1 4 8 \
  --context-lens 128 512 2048 \
  --dtype float16 \
  --warmup 10 \
  --iters 50
```

## Interpretation

This is a stronger infra signal than another standalone microkernel because it
matches real model attention shapes. Qwen2.5 and Qwen3 use different head
layouts, so passing both profiles demonstrates that TurboInfer's paged attention
path is parameterized rather than hard-coded to one model.

## AutoDL RTX 3090 Result

Both profiles completed successfully on AutoDL RTX 3090.

Summary:

- Qwen2.5-0.5B shape: Triton GQA attention latency ranged from about `0.055 ms`
  to `0.497 ms`; attention-only speedup versus the PyTorch GQA reference ranged
  from `11.11x` to `313.31x`.
- Qwen3-0.6B shape: Triton GQA attention latency ranged from about `0.053 ms`
  to `0.462 ms`; attention-only speedup versus the PyTorch GQA reference ranged
  from `11.83x` to `362.04x`.
- Qwen3 reached the highest simple bandwidth estimate: `145.28 GB/s` at
  `batch=8, context_len=2048`.

Full report:
[../../reports/gqa-paged-attention-profiles-autodl-3090.md](../../reports/gqa-paged-attention-profiles-autodl-3090.md).

The next step is to add RoPE placement to the same controlled path before
attempting a full Hugging Face model attention patch.
