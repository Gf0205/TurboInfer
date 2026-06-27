# Triton RoPE Benchmark: AutoDL RTX 3090

Status: completed.

## Summary

TurboInfer's Triton RoPE kernel was compared against a PyTorch reference implementation on an AutoDL RTX 3090 machine.

- GPU: NVIDIA GeForce RTX 3090
- PyTorch: 2.1.2+cu121
- CUDA runtime used by PyTorch: 12.1
- dtype: float16
- Q heads: 14
- KV heads: 2
- head dimension: 64
- block heads: 4
- warmup iterations: 25
- measured iterations: 100

| Seq Len | Max Diff Q | Max Diff K | PyTorch ms | Triton ms | Speedup | PyTorch GB/s | Triton GB/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.000000 | 0.000000 | 0.2619 | 0.0779 | 3.36x | 0.02 | 0.05 |
| 8 | 0.000000 | 0.000000 | 0.2531 | 0.0723 | 3.50x | 0.13 | 0.45 |
| 32 | 0.000008 | 0.000000 | 0.2539 | 0.0845 | 3.00x | 0.52 | 1.55 |
| 128 | 0.000977 | 0.000000 | 0.2886 | 0.0794 | 3.64x | 1.82 | 6.61 |
| 512 | 0.000977 | 0.000977 | 0.2710 | 0.0727 | 3.73x | 7.74 | 28.85 |

The Triton RoPE kernel is consistently around 3.0x to 3.73x faster than the PyTorch reference. The maximum absolute error is within normal float16 tolerance.

## Command

```bash
cd ~/TurboInfer
git pull
pip install -e .
pip install triton
python benchmarks/bench_rope.py \
  --seq-lens 1 8 32 128 512 \
  --q-heads 14 \
  --kv-heads 2 \
  --head-dim 64 \
  --dtype float16 \
  --warmup 25 \
  --iters 100
```

## Interpretation

RoPE is the position-encoding step applied to Q and K before attention. This benchmark is attention-adjacent but is not a full attention kernel.

The small-sequence cases are launch-overhead-sensitive. Larger sequence lengths show better effective bandwidth because the same kernel launch covers more token/head vectors.

Together with the RMSNorm and SiLU-Mul benchmarks, this result gives TurboInfer real custom-kernel coverage across:

- normalization path: RMSNorm;
- attention preprocessing path: RoPE;
- FFN/MLP path: SiLU-Mul.

## Raw Output

```json
{
  "benchmark": "rope",
  "device": "NVIDIA GeForce RTX 3090",
  "torch": "2.1.2+cu121",
  "cuda": "12.1",
  "warmup": 25,
  "iters": 100,
  "block_heads": 4,
  "results": [
    {
      "seq_len": 1,
      "q_heads": 14,
      "kv_heads": 2,
      "head_dim": 64,
      "dtype": "float16",
      "max_abs_diff_q": 0.0,
      "max_abs_diff_k": 0.0,
      "pytorch_ms": 0.26191871643066406,
      "triton_ms": 0.07787519931793213,
      "speedup": 3.3633135930908966,
      "pytorch_gbps": 0.01563843949687462,
      "triton_gbps": 0.052596976134567965
    },
    {
      "seq_len": 8,
      "q_heads": 14,
      "kv_heads": 2,
      "head_dim": 64,
      "dtype": "float16",
      "max_abs_diff_q": 0.0,
      "max_abs_diff_k": 0.0,
      "pytorch_ms": 0.2530508804321289,
      "triton_ms": 0.07226367950439454,
      "speedup": 3.5017713209128836,
      "pytorch_gbps": 0.12949174468013258,
      "triton_gbps": 0.45345047781586184
    },
    {
      "seq_len": 32,
      "q_heads": 14,
      "kv_heads": 2,
      "head_dim": 64,
      "dtype": "float16",
      "max_abs_diff_q": 7.62939453125e-06,
      "max_abs_diff_k": 0.0,
      "pytorch_ms": 0.2538598442077637,
      "triton_ms": 0.08450048446655273,
      "speedup": 3.00424128702182,
      "pytorch_gbps": 0.51631639658901,
      "triton_gbps": 1.5511390357990358
    },
    {
      "seq_len": 128,
      "q_heads": 14,
      "kv_heads": 2,
      "head_dim": 64,
      "dtype": "float16",
      "max_abs_diff_q": 0.0009765625,
      "max_abs_diff_k": 0.0,
      "pytorch_ms": 0.28858367919921873,
      "triton_ms": 0.07937024116516113,
      "speedup": 3.635917882606737,
      "pytorch_gbps": 1.8167624775414513,
      "triton_gbps": 6.605599180541883
    },
    {
      "seq_len": 512,
      "q_heads": 14,
      "kv_heads": 2,
      "head_dim": 64,
      "dtype": "float16",
      "max_abs_diff_q": 0.0009765625,
      "max_abs_diff_k": 0.0009765625,
      "pytorch_ms": 0.270960636138916,
      "triton_ms": 0.07268352031707764,
      "speedup": 3.7279514662590083,
      "pytorch_gbps": 7.739692487748784,
      "triton_gbps": 28.853197958096914
    }
  ]
}
```
