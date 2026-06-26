# Dynamic Paged KV Cache Simulation

## Run Context

- Date: 2026-06-26
- Mode: dynamic
- Number of requests: 64
- Arrival interval: 4 decode steps
- Short prompt tokens: 128
- Long prompt tokens: 2048
- Short output tokens: 64
- Long output tokens: 256
- Block size: 16 tokens
- Total KV budget: 2048 blocks = 32,768 token slots
- Contiguous max sequence tokens: 2304

## Results

| Allocation | Completed | Rejected | Peak Live Requests | Peak Allocated Slots | Peak Used Slots | Peak Wasted Slots | Peak Utilization |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Contiguous KV | 35 | 29 | 14 | 32,256 | 17,108 | 15,148 | 53.04% |
| Paged KV | 44 | 20 | 22 | 32,768 | 32,603 | 165 | 99.50% |

## Comparison

- Paged KV completed `9` more requests under the same token-slot budget.
- Paged KV rejected `9` fewer requests.
- Peak live requests increased from `14` to `22`.
- Peak utilization improved from `53.04%` to `99.50%`.
- Peak wasted token slots dropped from `15,148` to `165`.

## Interpretation

This dynamic simulation connects Paged KV Cache to continuous batching.

When requests arrive and finish over time, contiguous allocation must reserve `max_sequence_tokens` for every live request. Short requests waste most of that reservation. Under a fixed memory budget, this lowers the number of requests the system can keep active.

Paged KV Cache allocates blocks on demand and releases them when requests finish. Under the same 32,768-token-slot budget, the paged manager supported a higher live request count and completed more requests.

This is the serving-capacity argument behind PagedAttention: the attention kernel is important, but the system-level win starts from block-based KV memory management.

