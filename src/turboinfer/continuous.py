from __future__ import annotations

import queue
import threading
import time
from concurrent.futures import Future
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from turboinfer.engine import GenerationResult, cuda_sync, peak_memory_mb, pick_device, pick_dtype
from turboinfer.metrics import GenerationMetrics, summarize_token_timings


PastKeyValues = tuple[tuple[torch.Tensor, torch.Tensor], ...]


@dataclass
class ContinuousRequest:
    request_id: int
    prompt: str
    max_new_tokens: int
    future: Future[GenerationResult]
    arrival_time: float
    prompt_tokens: int = 0
    generated_tokens: list[torch.Tensor] = field(default_factory=list)
    token_timestamps: list[float] = field(default_factory=list)
    attention_mask: torch.Tensor | None = None
    past_key_values: PastKeyValues | None = None
    next_token: torch.Tensor | None = None


class ContinuousBatchingEngine:
    """A small real continuous batching scheduler on top of HF KV cache.

    This engine is intentionally simple. It demonstrates request queueing,
    dynamic admission, active-set decode steps, and per-request latency metrics.
    It pads legacy Hugging Face KV caches when active requests have different
    context lengths, which is useful for learning but not a replacement for
    vLLM's paged KV cache implementation.
    """

    def __init__(
        self,
        model_name: str,
        device: str = "auto",
        trust_remote_code: bool = False,
        max_batch_size: int = 8,
        batch_wait_seconds: float = 0.002,
    ) -> None:
        self.model_name = model_name
        self.device = pick_device(device)
        self.max_batch_size = max_batch_size
        self.batch_wait_seconds = batch_wait_seconds
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

        self._queue: queue.Queue[ContinuousRequest | None] = queue.Queue()
        self._next_request_id = 0
        self._id_lock = threading.Lock()
        self._worker = threading.Thread(target=self._worker_loop, name="turboinfer-continuous", daemon=True)
        self._worker.start()

    def generate(self, prompt: str, max_new_tokens: int) -> GenerationResult:
        with self._id_lock:
            request_id = self._next_request_id
            self._next_request_id += 1
        future: Future[GenerationResult] = Future()
        self._queue.put(
            ContinuousRequest(
                request_id=request_id,
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                future=future,
                arrival_time=time.perf_counter(),
            )
        )
        return future.result()

    def close(self) -> None:
        self._queue.put(None)
        self._worker.join(timeout=5.0)

    def _drain_new_requests(self, limit: int, wait: bool) -> list[ContinuousRequest]:
        requests: list[ContinuousRequest] = []
        if limit <= 0:
            return requests
        timeout = self.batch_wait_seconds if wait else 0.0
        try:
            item = self._queue.get(timeout=timeout)
        except queue.Empty:
            return requests
        if item is None:
            self._queue.put(None)
            return requests
        requests.append(item)

        while len(requests) < limit:
            try:
                item = self._queue.get_nowait()
            except queue.Empty:
                break
            if item is None:
                self._queue.put(None)
                break
            requests.append(item)
        return requests

    @torch.inference_mode()
    def _worker_loop(self) -> None:
        active: list[ContinuousRequest] = []
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)

        while True:
            new_requests = self._drain_new_requests(
                limit=max(0, self.max_batch_size - len(active)),
                wait=not active,
            )
            if not active and not new_requests and self._should_stop():
                break

            if new_requests:
                self._prefill(new_requests)
                for request in new_requests:
                    if self._is_finished(request):
                        self._finish(request)
                    else:
                        active.append(request)

            if active:
                batch = active[: self.max_batch_size]
                self._decode_step(batch)
                still_active: list[ContinuousRequest] = []
                batch_ids = {request.request_id for request in batch}
                for request in active:
                    if request.request_id in batch_ids and self._is_finished(request):
                        self._finish(request)
                    else:
                        still_active.append(request)
                active = still_active

    def _should_stop(self) -> bool:
        try:
            item = self._queue.get_nowait()
        except queue.Empty:
            return False
        if item is None:
            return True
        self._queue.put(item)
        return False

    def _prefill(self, requests: list[ContinuousRequest]) -> None:
        encoded = self.tokenizer(
            [request.prompt for request in requests],
            return_tensors="pt",
            padding=True,
        )
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)
        prompt_lengths = attention_mask.sum(dim=-1).tolist()

        cuda_sync(self.device)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=True,
        )
        next_tokens = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
        cuda_sync(self.device)
        now = time.perf_counter()

        for idx, request in enumerate(requests):
            request.prompt_tokens = int(prompt_lengths[idx])
            request.attention_mask = attention_mask[idx : idx + 1]
            request.past_key_values = self._slice_past(outputs.past_key_values, idx)
            request.next_token = next_tokens[idx : idx + 1]
            request.generated_tokens.append(request.next_token)
            request.token_timestamps.append(now)

    def _decode_step(self, requests: list[ContinuousRequest]) -> None:
        past = self._batch_past([self._require_past(request) for request in requests])
        masks = [self._require_attention_mask(request) for request in requests]
        max_len = max(int(mask.shape[-1]) for mask in masks)
        padded_masks = [
            F.pad(mask, (max_len - int(mask.shape[-1]), 0), value=0) if int(mask.shape[-1]) < max_len else mask
            for mask in masks
        ]
        attention_mask = torch.cat(padded_masks, dim=0)
        input_ids = torch.cat([self._require_next_token(request) for request in requests], dim=0)
        attention_mask = torch.cat(
            [
                attention_mask,
                torch.ones((len(requests), 1), dtype=attention_mask.dtype, device=self.device),
            ],
            dim=-1,
        )

        cuda_sync(self.device)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past,
            use_cache=True,
        )
        next_tokens = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
        cuda_sync(self.device)
        now = time.perf_counter()

        for idx, request in enumerate(requests):
            request.attention_mask = attention_mask[idx : idx + 1]
            request.past_key_values = self._slice_past(outputs.past_key_values, idx)
            request.next_token = next_tokens[idx : idx + 1]
            request.generated_tokens.append(request.next_token)
            request.token_timestamps.append(now)

    def _finish(self, request: ContinuousRequest) -> None:
        end_time = time.perf_counter()
        output_ids = torch.cat(request.generated_tokens, dim=-1)
        text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        output_tokens = int(output_ids.shape[-1])
        ttft_seconds, tpot_seconds, tokens_per_second = summarize_token_timings(
            request.token_timestamps,
            request.arrival_time,
            end_time,
        )
        result = GenerationResult(
            text=text,
            metrics=GenerationMetrics(
                model=self.model_name,
                device=str(self.device),
                prompt_tokens=request.prompt_tokens,
                output_tokens=output_tokens,
                total_seconds=end_time - request.arrival_time,
                ttft_seconds=ttft_seconds,
                tpot_seconds=tpot_seconds,
                tokens_per_second=tokens_per_second,
                peak_memory_mb=peak_memory_mb(self.device),
                optimization="continuous_batching_hf_kv_cache",
            ),
        )
        request.future.set_result(result)

    def _is_finished(self, request: ContinuousRequest) -> bool:
        if len(request.generated_tokens) >= request.max_new_tokens:
            return True
        eos_token_id = self.tokenizer.eos_token_id
        if eos_token_id is None or request.next_token is None:
            return False
        return int(request.next_token.item()) == eos_token_id

    def _slice_past(self, past: PastKeyValues, batch_idx: int) -> PastKeyValues:
        return tuple((key[batch_idx : batch_idx + 1].contiguous(), value[batch_idx : batch_idx + 1].contiguous()) for key, value in past)

    def _batch_past(self, past_values: list[PastKeyValues]) -> PastKeyValues:
        num_layers = len(past_values[0])
        batched_layers = []
        for layer_idx in range(num_layers):
            keys = [past[layer_idx][0] for past in past_values]
            values = [past[layer_idx][1] for past in past_values]
            max_len = max(int(key.shape[2]) for key in keys)
            padded_keys = [self._left_pad_cache(key, max_len) for key in keys]
            padded_values = [self._left_pad_cache(value, max_len) for value in values]
            batched_layers.append((torch.cat(padded_keys, dim=0), torch.cat(padded_values, dim=0)))
        return tuple(batched_layers)

    def _left_pad_cache(self, tensor: torch.Tensor, target_len: int) -> torch.Tensor:
        pad_len = target_len - int(tensor.shape[2])
        if pad_len <= 0:
            return tensor
        return F.pad(tensor, (0, 0, pad_len, 0), value=0)

    def _require_attention_mask(self, request: ContinuousRequest) -> torch.Tensor:
        if request.attention_mask is None:
            raise RuntimeError("request has not been prefilled")
        return request.attention_mask

    def _require_past(self, request: ContinuousRequest) -> PastKeyValues:
        if request.past_key_values is None:
            raise RuntimeError("request has not been prefilled")
        return request.past_key_values

    def _require_next_token(self, request: ContinuousRequest) -> torch.Tensor:
        if request.next_token is None:
            raise RuntimeError("request has not been prefilled")
        return request.next_token
