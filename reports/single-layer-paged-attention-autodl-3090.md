# Single-Layer Paged Attention Benchmark: AutoDL RTX 3090

Status: completed.

## Summary

TurboInfer's controlled single-layer paged attention path was validated on an
AutoDL RTX 3090 machine.

This benchmark connects:

1. hidden states;
2. Q/K/V projection;
3. `PagedKVAllocator`;
4. `PagedKVBuffer`;
5. paged decode attention metadata;
6. `triton_paged_decode_attention`.

- GPU: NVIDIA GeForce RTX 3090
- PyTorch: 2.1.2+cu121
- CUDA runtime used by PyTorch: 12.1
- dtype: float16
- hidden size: 896
- query heads: 14
- head dimension: 64
- block size: 16
- warmup iterations: 10
- measured iterations: 50

## Correctness Tests

The focused paged attention suite passed on AutoDL:

```text
collected 19 items
tests/test_paged_allocator.py ........
tests/test_paged_decode_attention.py ...
tests/test_paged_kv_buffer.py ....
tests/test_single_layer_attention.py ....
19 passed in 3.25s
```

## Results

| Batch | Context Len | Max Diff Ref | Max Diff Triton | Setup ms | Contiguous Full Ref ms | Paged PyTorch Attn ms | Paged Triton Attn ms | Attn Speedup | Triton GB/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 128 | 0.000000 | 0.000000 | 5.4027 | 0.3458 | 0.3741 | 0.0555 | 6.75x | 8.30 |
| 1 | 512 | 0.000000 | 0.000000 | 21.4125 | 0.3433 | 0.9036 | 0.1025 | 8.82x | 17.92 |
| 1 | 2048 | 0.000000 | 0.000000 | 85.1949 | 0.3370 | 2.6004 | 0.4360 | 5.96x | 16.84 |
| 4 | 128 | 0.000004 | 0.000015 | 20.8691 | 0.3737 | 1.4129 | 0.0527 | 26.79x | 34.93 |
| 4 | 512 | 0.000000 | 0.000000 | 83.4683 | 0.3729 | 3.2417 | 0.1124 | 28.84x | 65.37 |
| 4 | 2048 | 0.031250 | 0.031250 | 334.0739 | 0.9507 | 10.4472 | 0.4370 | 23.91x | 67.20 |
| 8 | 128 | 0.031250 | 0.000000 | 41.7510 | 0.3713 | 2.7912 | 0.0522 | 53.47x | 70.58 |
| 8 | 512 | 0.001953 | 0.031250 | 170.3328 | 0.5302 | 6.4327 | 0.1354 | 47.50x | 108.50 |
| 8 | 2048 | 0.031250 | 0.015625 | 667.4687 | 1.7517 | 20.7908 | 0.4520 | 46.00x | 129.95 |

## Interpretation

The important correctness signal is that the complete projected path passes the
test suite, including the CUDA Triton comparison. The larger `0.03125` absolute
differences appear only in high-magnitude float16 projection cases and remain
small relative to the generated activation range for this synthetic benchmark.

The attention kernel remains fast after moving one layer closer to transformer
execution. `paged_triton_attention_ms` ranges from about `0.052 ms` to
`0.452 ms`, with attention-only speedups from about `5.96x` to `53.47x` against
the paged PyTorch attention reference.

The large `setup_ms` values are not production decode latency. They include
rebuilding projected prompt K/V and writing the entire prompt into paged storage
inside the benchmark loop. This exposes the next engineering bottleneck:
`PagedKVBuffer.write_prompt` should write by block rather than by Python token
loop.

## Command

```bash
cd ~/TurboInfer
git pull origin main
python -m pytest \
  tests/test_paged_allocator.py \
  tests/test_paged_decode_attention.py \
  tests/test_paged_kv_buffer.py \
  tests/test_single_layer_attention.py
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

## Raw Output

```json
{
  "benchmark": "single_layer_paged_attention",
  "device": "NVIDIA GeForce RTX 3090",
  "torch": "2.1.2+cu121",
  "cuda": "12.1",
  "warmup": 10,
  "iters": 50,
  "results": [
    {
      "batch_size": 1,
      "context_len": 128,
      "prompt_len": 127,
      "hidden_size": 896,
      "num_heads": 14,
      "head_dim": 64,
      "block_size": 16,
      "dtype": "float16",
      "max_abs_diff_paged_ref": 0.0,
      "max_abs_diff_triton": 0.0,
      "setup_ms": 5.402665405273438,
      "contiguous_full_reference_ms": 0.34582527160644533,
      "paged_pytorch_attention_ms": 0.37410816192626956,
      "paged_triton_attention_ms": 0.05545983791351319,
      "attention_speedup": 6.745568973888317,
      "paged_triton_attention_gbps": 8.304099278439924
    },
    {
      "batch_size": 1,
      "context_len": 512,
      "prompt_len": 511,
      "hidden_size": 896,
      "num_heads": 14,
      "head_dim": 64,
      "block_size": 16,
      "dtype": "float16",
      "max_abs_diff_paged_ref": 0.0,
      "max_abs_diff_triton": 0.0,
      "setup_ms": 21.41253662109375,
      "contiguous_full_reference_ms": 0.3433267211914062,
      "paged_pytorch_attention_ms": 0.9035775756835938,
      "paged_triton_attention_ms": 0.10250240325927734,
      "attention_speedup": 8.81518429765999,
      "paged_triton_attention_gbps": 17.919579849790047
    },
    {
      "batch_size": 1,
      "context_len": 2048,
      "prompt_len": 2047,
      "hidden_size": 896,
      "num_heads": 14,
      "head_dim": 64,
      "block_size": 16,
      "dtype": "float16",
      "max_abs_diff_paged_ref": 0.0,
      "max_abs_diff_triton": 0.0,
      "setup_ms": 85.194873046875,
      "contiguous_full_reference_ms": 0.3370393753051758,
      "paged_pytorch_attention_ms": 2.600427551269531,
      "paged_triton_attention_ms": 0.43603904724121095,
      "attention_speedup": 5.9637492736539475,
      "paged_triton_attention_gbps": 16.837537937143967
    },
    {
      "batch_size": 4,
      "context_len": 128,
      "prompt_len": 127,
      "hidden_size": 896,
      "num_heads": 14,
      "head_dim": 64,
      "block_size": 16,
      "dtype": "float16",
      "max_abs_diff_paged_ref": 3.814697265625e-06,
      "max_abs_diff_triton": 1.52587890625e-05,
      "setup_ms": 20.8690576171875,
      "contiguous_full_reference_ms": 0.37369857788085936,
      "paged_pytorch_attention_ms": 1.4129139709472656,
      "paged_triton_attention_ms": 0.052736001014709474,
      "attention_speedup": 26.79220918842834,
      "paged_triton_attention_gbps": 34.93203816281345
    },
    {
      "batch_size": 4,
      "context_len": 512,
      "prompt_len": 511,
      "hidden_size": 896,
      "num_heads": 14,
      "head_dim": 64,
      "block_size": 16,
      "dtype": "float16",
      "max_abs_diff_paged_ref": 0.0,
      "max_abs_diff_triton": 0.0,
      "setup_ms": 83.468271484375,
      "contiguous_full_reference_ms": 0.37285888671875,
      "paged_pytorch_attention_ms": 3.241677551269531,
      "paged_triton_attention_ms": 0.11239423751831054,
      "attention_speedup": 28.842026271510743,
      "paged_triton_attention_gbps": 65.36989940256538
    },
    {
      "batch_size": 4,
      "context_len": 2048,
      "prompt_len": 2047,
      "hidden_size": 896,
      "num_heads": 14,
      "head_dim": 64,
      "block_size": 16,
      "dtype": "float16",
      "max_abs_diff_paged_ref": 0.03125,
      "max_abs_diff_triton": 0.03125,
      "setup_ms": 334.0738671875,
      "contiguous_full_reference_ms": 0.9507430267333984,
      "paged_pytorch_attention_ms": 10.44717529296875,
      "paged_triton_attention_ms": 0.43700225830078127,
      "attention_speedup": 23.9064560755156,
      "paged_triton_attention_gbps": 67.2017030625663
    },
    {
      "batch_size": 8,
      "context_len": 128,
      "prompt_len": 127,
      "hidden_size": 896,
      "num_heads": 14,
      "head_dim": 64,
      "block_size": 16,
      "dtype": "float16",
      "max_abs_diff_paged_ref": 0.03125,
      "max_abs_diff_triton": 0.0,
      "setup_ms": 41.7510009765625,
      "contiguous_full_reference_ms": 0.3713024139404297,
      "paged_pytorch_attention_ms": 2.7911578369140626,
      "paged_triton_attention_ms": 0.052203521728515626,
      "attention_speedup": 53.466849448002314,
      "paged_triton_attention_gbps": 70.57669440694958
    },
    {
      "batch_size": 8,
      "context_len": 512,
      "prompt_len": 511,
      "hidden_size": 896,
      "num_heads": 14,
      "head_dim": 64,
      "block_size": 16,
      "dtype": "float16",
      "max_abs_diff_paged_ref": 0.001953125,
      "max_abs_diff_triton": 0.03125,
      "setup_ms": 170.33275390625,
      "contiguous_full_reference_ms": 0.5302067184448243,
      "paged_pytorch_attention_ms": 6.432706298828125,
      "paged_triton_attention_ms": 0.13543423652648925,
      "attention_speedup": 47.496899335124674,
      "paged_triton_attention_gbps": 108.49841500103969
    },
    {
      "batch_size": 8,
      "context_len": 2048,
      "prompt_len": 2047,
      "hidden_size": 896,
      "num_heads": 14,
      "head_dim": 64,
      "block_size": 16,
      "dtype": "float16",
      "max_abs_diff_paged_ref": 0.03125,
      "max_abs_diff_triton": 0.015625,
      "setup_ms": 667.468671875,
      "contiguous_full_reference_ms": 1.7516954040527344,
      "paged_pytorch_attention_ms": 20.7908447265625,
      "paged_triton_attention_ms": 0.45197311401367185,
      "attention_speedup": 46.00018028048809,
      "paged_triton_attention_gbps": 129.95151742194852
    }
  ]
}
```
