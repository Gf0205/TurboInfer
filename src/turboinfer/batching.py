from __future__ import annotations

import time
from dataclasses import asdict, dataclass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from turboinfer.engine import cuda_sync, peak_memory_mb, pick_device, pick_dtype


@dataclass(frozen=True)
class BatchMetrics:
    model: str
    device: str
    num_requests: int
    prompt_tokens_per_request: int
    output_tokens_per_request: int
    total_output_tokens: int
    total_seconds: float
    request_throughput_per_second: float
    token_throughput_per_second: float
    mean_tpot_seconds: float
    peak_memory_mb: float | None
    optimization: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class BatchGenerationResult:
    texts: list[str]
    metrics: BatchMetrics


class StaticBatchKVCacheEngine:
    """Fixed-arrival batched decode with KV cache.

    This is the first stepping stone toward continuous batching. All requests
    arrive together, prefill is batched once, and every decode step is executed
    as one batch while all requests remain active.
    """

    def __init__(
        self,
        model_name: str,
        device: str = "auto",
        trust_remote_code: bool = False,
    ) -> None:
        self.model_name = model_name
        self.device = pick_device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=trust_remote_code,
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=pick_dtype(self.device),
            trust_remote_code=trust_remote_code,
        )
        self.model.to(self.device)
        self.model.eval()

    @torch.inference_mode()
    def generate_batch(
        self,
        prompts: list[str],
        max_new_tokens: int,
    ) -> BatchGenerationResult:
        if not prompts:
            raise ValueError("prompts must not be empty")

        encoded = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
        )
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)
        prompt_tokens_per_request = int(attention_mask[0].sum().item())
        batch_size = int(input_ids.shape[0])
        generated_tokens: list[torch.Tensor] = []

        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)

        cuda_sync(self.device)
        start_time = time.perf_counter()

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=True,
        )
        past_key_values = outputs.past_key_values
        next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
        generated_tokens.append(next_token)

        for _ in range(max_new_tokens - 1):
            attention_mask = torch.cat(
                [
                    attention_mask,
                    torch.ones((batch_size, 1), dtype=attention_mask.dtype, device=self.device),
                ],
                dim=-1,
            )
            outputs = self.model(
                input_ids=next_token,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
            )
            past_key_values = outputs.past_key_values
            next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
            generated_tokens.append(next_token)

        cuda_sync(self.device)
        end_time = time.perf_counter()

        output_ids = torch.cat(generated_tokens, dim=-1)
        texts = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        total_output_tokens = int(output_ids.numel())
        total_seconds = end_time - start_time
        token_throughput = total_output_tokens / total_seconds if total_seconds > 0 else 0.0
        request_throughput = batch_size / total_seconds if total_seconds > 0 else 0.0
        mean_tpot = total_seconds / max_new_tokens if max_new_tokens > 0 else 0.0

        return BatchGenerationResult(
            texts=texts,
            metrics=BatchMetrics(
                model=self.model_name,
                device=str(self.device),
                num_requests=batch_size,
                prompt_tokens_per_request=prompt_tokens_per_request,
                output_tokens_per_request=max_new_tokens,
                total_output_tokens=total_output_tokens,
                total_seconds=total_seconds,
                request_throughput_per_second=request_throughput,
                token_throughput_per_second=token_throughput,
                mean_tpot_seconds=mean_tpot,
                peak_memory_mb=peak_memory_mb(self.device),
                optimization="static_batch_kv_cache",
            ),
        )

