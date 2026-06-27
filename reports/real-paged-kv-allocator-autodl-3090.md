# Real Paged KV Allocator: AutoDL RTX 3090

Status: completed.

## Summary

TurboInfer's real paged KV metadata allocator was benchmarked with a mixed-length request workload.

- Requests: 32
- Arrival interval: 4 steps
- Short prompt/output: 128 + 64 tokens
- Long prompt/output: 2048 + 256 tokens
- Block size: 16 tokens
- Total blocks: 2048
- Total token slot budget: 32768
- Contiguous max sequence reservation: 2304 tokens/request

| Metric | Paged Allocator |
| --- | ---: |
| Completed requests | 30 |
| Rejected requests | 2 |
| Peak live requests | 22 |
| Peak used blocks | 1974 |
| Peak allocated token slots | 31584 |
| Peak used token slots | 31430 |
| Peak wasted token slots | 154 |
| Peak utilization | 99.51% |
| Allocation failures | 2 |

## Comparison With Full Contiguous Reservation

| Metric | Contiguous Full Reservation | Paged Allocator Peak |
| --- | ---: | ---: |
| Allocated token slots | 73728 | 31584 |
| Used token slots | 39936 | 31430 |
| Wasted token slots | 33792 | 154 |
| Utilization | 54.17% | 99.51% |

The paged allocator reduces peak allocated token slots by about 2.33x and saves 33638 wasted token slots in this workload.

Important note: the contiguous baseline shown here is a full-reservation baseline over the configured workload, while paged allocation is measured at runtime peak. This is useful for showing memory waste, but a later benchmark should add a dynamic contiguous baseline under the same admission policy.

## Allocator Benchmark Command

```bash
python benchmarks/bench_paged_allocator.py \
  --num-requests 32 \
  --arrival-interval-steps 4 \
  --short-prompt-tokens 128 \
  --long-prompt-tokens 2048 \
  --short-output-tokens 64 \
  --long-output-tokens 256 \
  --block-size 16 \
  --total-blocks 2048 \
  --max-sequence-tokens 2304
```

## Continuous Server Command

```bash
python scripts/start_server_background.py \
  --model /root/autodl-tmp/models/Qwen2.5-0.5B \
  --device cuda \
  --host 127.0.0.1 \
  --port 8000 \
  --max-batch-size 8 \
  --batch-wait-seconds 0.002 \
  --kv-block-size 16 \
  --kv-total-blocks 4096 \
  --preload
```

## Continuous HTTP Benchmark Command

```bash
python benchmarks/bench_http_completions.py \
  --url http://127.0.0.1:8000/v1/completions \
  --engine continuous \
  --num-requests 8 \
  --concurrency 8 \
  --max-tokens 64 \
  --timeout-seconds 300
```

## Result

## Raw Allocator Benchmark Output

```json
{
  "workload": {
    "num_requests": 32,
    "arrival_interval_steps": 4,
    "short_prompt_tokens": 128,
    "long_prompt_tokens": 2048,
    "short_output_tokens": 64,
    "long_output_tokens": 256,
    "block_size": 16,
    "total_blocks": 2048,
    "max_sequence_tokens": 2304
  },
  "paged_allocator": {
    "completed_requests": 30,
    "rejected_requests": 2,
    "final_stats": {
      "block_size": 16,
      "total_blocks": 2048,
      "used_blocks": 0,
      "free_blocks": 2048,
      "live_requests": 0,
      "peak_used_blocks": 1974,
      "peak_live_requests": 22,
      "allocated_token_slots": 0,
      "used_token_slots": 0,
      "wasted_token_slots": 0,
      "utilization": 1.0,
      "total_allocated_requests": 30,
      "total_freed_requests": 30,
      "allocation_failures": 2
    },
    "peak_stats": {
      "block_size": 16,
      "total_blocks": 2048,
      "used_blocks": 1974,
      "free_blocks": 74,
      "live_requests": 14,
      "peak_used_blocks": 1974,
      "peak_live_requests": 22,
      "allocated_token_slots": 31584,
      "used_token_slots": 31430,
      "wasted_token_slots": 154,
      "utilization": 0.9951241134751773,
      "total_allocated_requests": 30,
      "total_freed_requests": 16,
      "allocation_failures": 2
    }
  },
  "contiguous_full_reservation": {
    "allocated_token_slots": 73728,
    "used_token_slots": 39936,
    "wasted_token_slots": 33792,
    "utilization": 0.5416666666666666
  },
  "comparison": {
    "peak_allocated_token_slots_reduction_ratio": 2.3343465045592704,
    "peak_wasted_token_slots_saved": 33638
  }
}
```

## Next Verification

Run the continuous HTTP benchmark again and record the `paged_kv_allocator` object returned in each response's metrics. That verifies the allocator metadata is also integrated into the real serving path.
