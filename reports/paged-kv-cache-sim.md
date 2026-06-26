# Paged KV Cache Simulation

## Run Context

- Date: 2026-06-26
- Number of requests: 16
- Short prompt tokens: 128
- Long prompt tokens: 2048
- Output tokens per request: 128
- Block size: 16 tokens
- Max sequence tokens for contiguous allocation: 2176

The workload alternates short and long prompts. This represents a common serving situation where active requests have different context lengths.

## Results

| Allocation | Allocated Token Slots | Used Token Slots | Wasted Token Slots | Utilization |
| --- | ---: | ---: | ---: | ---: |
| Contiguous KV | 34,816 | 19,456 | 15,360 | 55.88% |
| Paged KV | 19,456 | 19,456 | 0 | 100.00% |

## Savings

- Allocated token slots saved: `15,360`
- Wasted token slots saved: `15,360`
- Allocation reduction ratio: `1.79x`

## Interpretation

Contiguous allocation reserves `max_sequence_tokens` for every request. Short prompts therefore waste most of their reserved KV cache slots.

Paged KV Cache allocates fixed-size blocks on demand. With a 16-token block size and token counts aligned to that block size, this run eliminates internal waste.

In a real engine, utilization will not always be exactly 100% because request lengths are not perfectly aligned with block boundaries. The important result is the direction: block-based allocation reduces waste and makes dynamic batching more memory efficient.

This simulation explains why PagedAttention-style systems need a block table. The attention kernel must read keys/values through logical-to-physical block mappings rather than assuming every request has one contiguous KV tensor.

