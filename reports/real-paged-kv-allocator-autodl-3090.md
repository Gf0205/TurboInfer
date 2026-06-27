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

## Comparison With Contiguous Reservation

| Metric | Contiguous Full Reservation | Paged Allocator Peak |
| --- | ---: | ---: |
| Allocated token slots | 73728 | 31584 |
| Used token slots | 39936 | 31430 |
| Wasted token slots | 33792 | 154 |
| Utilization | 54.17% | 99.51% |

The paged allocator reduces peak allocated token slots by about 2.33x and saves 33638 wasted token slots in this workload.

The stricter comparison is the dynamic contiguous baseline below. It uses the same arrival/decode/free policy as paged allocation, but every live request reserves `max_sequence_tokens` slots.

| Metric | Dynamic Contiguous | Paged Allocator |
| --- | ---: | ---: |
| Completed requests | 21 | 30 |
| Rejected requests | 11 | 2 |
| Peak live requests | 14 | 22 |
| Peak allocated token slots | 32256 | 31584 |
| Peak used token slots | 15610 | 31430 |
| Peak wasted token slots | 16646 | 154 |
| Peak utilization | 48.39% | 99.51% |

Under the same token-slot budget, paged allocation completes 9 more requests and rejects 9 fewer requests than dynamic contiguous reservation. It also saves 16492 wasted token slots at peak.

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
  "dynamic_contiguous_reservation": {
    "completed_requests": 21,
    "rejected_requests": 11,
    "peak_live_requests": 14,
    "peak_used_token_slots": 16212,
    "peak_stats": {
      "live_requests": 14,
      "allocated_token_slots": 32256,
      "used_token_slots": 15610,
      "wasted_token_slots": 16646,
      "utilization": 0.4839409722222222
    }
  },
  "comparison": {
    "peak_allocated_token_slots_reduction_ratio": 2.3343465045592704,
    "peak_wasted_token_slots_saved": 33638,
    "dynamic_completed_request_delta": 9,
    "dynamic_rejected_request_delta": -9,
    "dynamic_peak_allocated_token_slots_reduction_ratio": 1.0212765957446808,
    "dynamic_peak_wasted_token_slots_saved": 16492
  }
}
```

## Next Verification

Completed. The continuous HTTP benchmark now returns `paged_kv_allocator` metrics inside each response.

## Continuous Server Integration Result

Workload:

- Engine: `continuous`
- Requests: 8
- HTTP concurrency: 8
- Output tokens per request: 64
- KV block size: 16
- KV total blocks: 4096

| Metric | Value |
| --- | ---: |
| Total seconds | 2.1830 |
| Request throughput | 3.6647 req/s |
| Completion token throughput | 234.5400 tokens/s |
| Mean latency | 2.1259 s |
| P50 latency | 2.1325 s |
| P95 latency | 2.1366 s |
| Total completion tokens | 512 |
| Peak memory | 1908.48 MB |

Allocator integration metrics observed in response metadata:

| Allocator Metric | Value |
| --- | ---: |
| Block size | 16 |
| Total blocks | 4096 |
| Peak used blocks | 40 |
| Peak live requests | 8 |
| Peak allocated token slots | 640 |
| Final used blocks | 0 |
| Final free blocks | 4096 |
| Total allocated requests | 8 |
| Total freed requests | 8 |
| Allocation failures | 0 |
| Per-request utilization near completion | 97.5% |

This verifies that the paged KV metadata allocator is integrated into the real continuous batching serving path. Blocks are allocated during prefill, appended during decode, and released when each request finishes.

Important limitation remains: the allocator currently tracks real metadata, while Hugging Face `past_key_values` still store the actual K/V tensors. The next system step is to connect these block tables to a paged decode attention kernel.

## Raw Continuous HTTP Summary

```json
{
  "engine": "continuous",
  "num_requests": 8,
  "concurrency": 8,
  "max_tokens": 64,
  "total_seconds": 2.1829962208867073,
  "request_throughput_per_second": 3.664687974929473,
  "completion_token_throughput_per_second": 234.54003039548627,
  "mean_latency_seconds": 2.1258502453565598,
  "p50_latency_seconds": 2.132468707859516,
  "p95_latency_seconds": 2.1365623772144318,
  "total_completion_tokens": 512,
  "final_paged_kv_allocator": {
    "block_size": 16,
    "total_blocks": 4096,
    "used_blocks": 0,
    "free_blocks": 4096,
    "live_requests": 0,
    "peak_used_blocks": 40,
    "peak_live_requests": 8,
    "allocated_token_slots": 0,
    "used_token_slots": 0,
    "wasted_token_slots": 0,
    "utilization": 1.0,
    "total_allocated_requests": 8,
    "total_freed_requests": 8,
    "allocation_failures": 0
  }
}
```
