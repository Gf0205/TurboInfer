from __future__ import annotations

import time
from dataclasses import dataclass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from turboinfer.metrics import GenerationMetrics, summarize_token_timings


@dataclass(frozen=True)
class GenerationResult:
    text: str
    metrics: GenerationMetrics


def pick_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def pick_dtype(device: torch.device) -> torch.dtype:
    if device.type == "cuda":
        return torch.float16
    return torch.float32


def cuda_sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def peak_memory_mb(device: torch.device) -> float | None:
    if device.type != "cuda":
        return None
    return torch.cuda.max_memory_allocated(device) / 1024**2


class NaiveEngine:
    """Single-request greedy decoding without KV cache.

    This intentionally recomputes the full context on every decode step. It is
    the baseline that later KV-cache and batching optimizations should beat.
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
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=pick_dtype(self.device),
            trust_remote_code=trust_remote_code,
        )
        self.model.to(self.device)
        self.model.eval()

    @torch.inference_mode()
    def generate(self, prompt: str, max_new_tokens: int) -> GenerationResult:
        encoded = self.tokenizer(prompt, return_tensors="pt")
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)

        prompt_tokens = int(input_ids.shape[-1])
        generated = input_ids
        token_timestamps: list[float] = []

        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)

        cuda_sync(self.device)
        start_time = time.perf_counter()

        for _ in range(max_new_tokens):
            current_attention_mask = torch.ones_like(generated, device=self.device)
            outputs = self.model(
                input_ids=generated,
                attention_mask=current_attention_mask,
                use_cache=False,
            )
            next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=-1)

            cuda_sync(self.device)
            token_timestamps.append(time.perf_counter())

            eos_token_id = self.tokenizer.eos_token_id
            if eos_token_id is not None and int(next_token.item()) == eos_token_id:
                break

        cuda_sync(self.device)
        end_time = time.perf_counter()

        output_ids = generated[:, prompt_tokens:]
        output_tokens = int(output_ids.shape[-1])
        text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        ttft_seconds, tpot_seconds, tokens_per_second = summarize_token_timings(
            token_timestamps,
            start_time,
            end_time,
        )

        return GenerationResult(
            text=text,
            metrics=GenerationMetrics(
                model=self.model_name,
                device=str(self.device),
                prompt_tokens=prompt_tokens,
                output_tokens=output_tokens,
                total_seconds=end_time - start_time,
                ttft_seconds=ttft_seconds,
                tpot_seconds=tpot_seconds,
                tokens_per_second=tokens_per_second,
                peak_memory_mb=peak_memory_mb(self.device),
                optimization="naive_no_kv_cache",
            ),
        )


class KVCacheEngine:
    """Single-request greedy decoding with Hugging Face KV cache.

    The prompt is processed once during prefill. Decode then feeds only the
    newest token while reusing cached key/value states from previous tokens.
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
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=pick_dtype(self.device),
            trust_remote_code=trust_remote_code,
        )
        self.model.to(self.device)
        self.model.eval()

    @torch.inference_mode()
    def generate(self, prompt: str, max_new_tokens: int) -> GenerationResult:
        encoded = self.tokenizer(prompt, return_tensors="pt")
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)
        else:
            attention_mask = torch.ones_like(input_ids, device=self.device)

        prompt_tokens = int(input_ids.shape[-1])
        generated_tokens: list[torch.Tensor] = []
        token_timestamps: list[float] = []

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

        cuda_sync(self.device)
        token_timestamps.append(time.perf_counter())

        eos_token_id = self.tokenizer.eos_token_id
        if eos_token_id is not None and int(next_token.item()) == eos_token_id:
            max_decode_steps = 0
        else:
            max_decode_steps = max_new_tokens - 1

        for _ in range(max_decode_steps):
            attention_mask = torch.cat(
                [attention_mask, torch.ones((1, 1), dtype=attention_mask.dtype, device=self.device)],
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
            token_timestamps.append(time.perf_counter())

            if eos_token_id is not None and int(next_token.item()) == eos_token_id:
                break

        cuda_sync(self.device)
        end_time = time.perf_counter()

        if generated_tokens:
            output_ids = torch.cat(generated_tokens, dim=-1)
        else:
            output_ids = torch.empty((1, 0), dtype=input_ids.dtype, device=self.device)
        output_tokens = int(output_ids.shape[-1])
        text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        ttft_seconds, tpot_seconds, tokens_per_second = summarize_token_timings(
            token_timestamps,
            start_time,
            end_time,
        )

        return GenerationResult(
            text=text,
            metrics=GenerationMetrics(
                model=self.model_name,
                device=str(self.device),
                prompt_tokens=prompt_tokens,
                output_tokens=output_tokens,
                total_seconds=end_time - start_time,
                ttft_seconds=ttft_seconds,
                tpot_seconds=tpot_seconds,
                tokens_per_second=tokens_per_second,
                peak_memory_mb=peak_memory_mb(self.device),
                optimization="hf_kv_cache",
            ),
        )
