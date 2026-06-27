# Triton Paged Decode Attention Benchmark: AutoDL RTX 3090

Status: completed.

## Summary

TurboInfer's Triton paged decode attention kernel was compared against a readable PyTorch reference implementation on an AutoDL RTX 3090 machine.

- GPU: NVIDIA GeForce RTX 3090
- PyTorch: 2.1.2+cu121
- CUDA runtime used by PyTorch: 12.1
- dtype: float16
- query heads: 14
- head dimension: 64
- block size: 16
- warmup iterations: 25
- measured iterations: 100

| Batch | Context Len | Max Abs Diff | PyTorch ms | Triton ms | Speedup | PyTorch GB/s | Triton GB/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 128 | 0.000000 | 0.3773 | 0.0530 | 7.12x | 1.22 | 8.70 |
| 1 | 512 | 0.000061 | 0.8394 | 0.1018 | 8.24x | 2.19 | 18.04 |
| 1 | 2048 | 0.000008 | 2.7142 | 0.4354 | 6.23x | 2.71 | 16.86 |
| 4 | 128 | 0.000031 | 1.4458 | 0.0514 | 28.13x | 1.27 | 35.84 |
| 4 | 512 | 0.000122 | 3.3312 | 0.1117 | 29.82x | 2.21 | 65.77 |
| 4 | 2048 | 0.000008 | 10.5509 | 0.4359 | 24.21x | 2.78 | 67.37 |
| 8 | 128 | 0.000031 | 2.8751 | 0.0520 | 55.27x | 1.28 | 70.83 |
| 8 | 512 | 0.000061 | 6.5666 | 0.1345 | 48.81x | 2.24 | 109.23 |
| 8 | 2048 | 0.000015 | 21.3507 | 0.5152 | 41.44x | 2.75 | 114.01 |

The Triton kernel is substantially faster than the simple PyTorch reference across all tested decode shapes. The maximum absolute difference stays within a practical float16 tolerance for this controlled attention benchmark.

## Command

```bash
cd ~/TurboInfer
git pull
pip install -e .
pip install triton
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

## Interpretation

This benchmark validates the first real attention-path Triton kernel in TurboInfer. It consumes the same paged metadata shape exported by `PagedKVAllocator.decode_metadata()`:

- `block_table`: physical KV block ids for each active request;
- `context_lens`: valid KV tokens per request;
- `k_cache` and `v_cache`: paged K/V tensors laid out by physical block.

The PyTorch implementation is intentionally not an optimized baseline. It is the correctness oracle. The important project signal is that the paged metadata contract works and the Triton kernel compiles and runs correctly on the target AutoDL RTX 3090 environment.

This still does not mean the full TurboInfer serving path is accelerated. The current continuous batching engine still uses Hugging Face `past_key_values`; the next step is to add a real paged K/V tensor buffer and then connect that buffer to an integration benchmark.

## Raw Output

```json
{
  "benchmark": "paged_decode_attention",
  "device": "NVIDIA GeForce RTX 3090",
  "torch": "2.1.2+cu121",
  "cuda": "12.1",
  "warmup": 25,
  "iters": 100,
  "results": [
    {
      "batch_size": 1,
      "context_len": 128,
      "num_heads": 14,
      "head_dim": 64,
      "block_size": 16,
      "dtype": "float16",
      "max_abs_diff": 4.76837158203125e-07,
      "pytorch_ms": 0.3772518539428711,
      "triton_ms": 0.05295104026794434,
      "speedup": 7.124540935057945,
      "pytorch_gbps": 1.2207865784795908,
      "triton_gbps": 8.697543951347175
    },
    {
      "batch_size": 1,
      "context_len": 512,
      "num_heads": 14,
      "head_dim": 64,
      "block_size": 16,
      "dtype": "float16",
      "max_abs_diff": 6.103515625e-05,
      "pytorch_ms": 0.839372787475586,
      "triton_ms": 0.10182656288146973,
      "speedup": 8.243161349290068,
      "pytorch_gbps": 2.1883006304316543,
      "triton_gbps": 18.0385151774013
    },
    {
      "batch_size": 1,
      "context_len": 2048,
      "num_heads": 14,
      "head_dim": 64,
      "block_size": 16,
      "dtype": "float16",
      "max_abs_diff": 7.62939453125e-06,
      "pytorch_ms": 2.71415283203125,
      "triton_ms": 0.43535327911376953,
      "speedup": 6.234368643223125,
      "pytorch_gbps": 2.705014954705199,
      "triton_gbps": 16.86406041306372
    },
    {
      "batch_size": 4,
      "context_len": 128,
      "num_heads": 14,
      "head_dim": 64,
      "block_size": 16,
      "dtype": "float16",
      "max_abs_diff": 3.0517578125e-05,
      "pytorch_ms": 1.4458367919921875,
      "triton_ms": 0.05140480041503906,
      "speedup": 28.126493641033406,
      "pytorch_gbps": 1.274124444890979,
      "triton_gbps": 35.836653097111345
    },
    {
      "batch_size": 4,
      "context_len": 512,
      "num_heads": 14,
      "head_dim": 64,
      "block_size": 16,
      "dtype": "float16",
      "max_abs_diff": 0.0001220703125,
      "pytorch_ms": 3.3312255859375,
      "triton_ms": 0.11171839714050293,
      "speedup": 29.818057465933528,
      "pytorch_gbps": 2.2055546256055463,
      "triton_gbps": 65.7653545705617
    },
    {
      "batch_size": 4,
      "context_len": 2048,
      "num_heads": 14,
      "head_dim": 64,
      "block_size": 16,
      "dtype": "float16",
      "max_abs_diff": 7.62939453125e-06,
      "pytorch_ms": 10.550897216796875,
      "triton_ms": 0.43588607788085937,
      "speedup": 24.205630214417514,
      "pytorch_gbps": 2.7833932410267153,
      "triton_gbps": 67.37378753360174
    },
    {
      "batch_size": 8,
      "context_len": 128,
      "num_heads": 14,
      "head_dim": 64,
      "block_size": 16,
      "dtype": "float16",
      "max_abs_diff": 3.0517578125e-05,
      "pytorch_ms": 2.8750949096679688,
      "triton_ms": 0.052019200325012206,
      "speedup": 55.26987903898144,
      "pytorch_gbps": 1.2814714351205498,
      "triton_gbps": 70.82677121102274
    },
    {
      "batch_size": 8,
      "context_len": 512,
      "num_heads": 14,
      "head_dim": 64,
      "block_size": 16,
      "dtype": "float16",
      "max_abs_diff": 6.103515625e-05,
      "pytorch_ms": 6.566563720703125,
      "triton_ms": 0.13453311920166017,
      "speedup": 48.81001614821767,
      "pytorch_gbps": 2.237760969816124,
      "triton_gbps": 109.22514907257623
    },
    {
      "batch_size": 8,
      "context_len": 2048,
      "num_heads": 14,
      "head_dim": 64,
      "block_size": 16,
      "dtype": "float16",
      "max_abs_diff": 1.52587890625e-05,
      "pytorch_ms": 21.35067626953125,
      "triton_ms": 0.5151846313476562,
      "speedup": 41.44276628299382,
      "pytorch_gbps": 2.7509476167655604,
      "triton_gbps": 114.00687913837398
    }
  ]
}
```
