from __future__ import annotations

from dataclasses import dataclass

import torch

from turboinfer.paged_allocator import PagedKVAllocator


@dataclass(frozen=True)
class PagedKVBufferShape:
    total_blocks: int
    num_heads: int
    block_size: int
    head_dim: int

    def to_dict(self) -> dict[str, int]:
        return {
            "total_blocks": self.total_blocks,
            "num_heads": self.num_heads,
            "block_size": self.block_size,
            "head_dim": self.head_dim,
        }


class PagedKVBuffer:
    """Own the physical K/V tensors used by paged decode attention.

    `PagedKVAllocator` tracks request ids, context lengths, and physical block
    ids. This buffer stores the actual key/value vectors in those physical
    blocks with layout `[num_blocks, num_heads, block_size, head_dim]`.
    """

    def __init__(
        self,
        allocator: PagedKVAllocator,
        num_heads: int,
        head_dim: int,
        dtype: torch.dtype = torch.float16,
        device: torch.device | str = "cpu",
    ) -> None:
        if num_heads <= 0:
            raise ValueError("num_heads must be positive")
        if head_dim <= 0:
            raise ValueError("head_dim must be positive")
        self.allocator = allocator
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.dtype = dtype
        self.device = torch.device(device)
        shape = (
            allocator.total_blocks,
            num_heads,
            allocator.block_size,
            head_dim,
        )
        self.k_cache = torch.empty(shape, dtype=dtype, device=self.device)
        self.v_cache = torch.empty_like(self.k_cache)

    @property
    def block_size(self) -> int:
        return self.allocator.block_size

    @property
    def total_blocks(self) -> int:
        return self.allocator.total_blocks

    @property
    def shape(self) -> PagedKVBufferShape:
        return PagedKVBufferShape(
            total_blocks=self.total_blocks,
            num_heads=self.num_heads,
            block_size=self.block_size,
            head_dim=self.head_dim,
        )

    def zero_(self) -> None:
        self.k_cache.zero_()
        self.v_cache.zero_()

    def write_prompt(
        self,
        request_id: int,
        keys: torch.Tensor,
        values: torch.Tensor,
    ) -> None:
        """Write the already-allocated prompt K/V vectors for a request.

        `keys` and `values` must have shape `[tokens, num_heads, head_dim]`.
        The request should already be allocated with the same prompt length.
        """

        self._validate_token_batch(keys, values)
        expected_tokens = self.allocator.context_length(request_id)
        if int(keys.shape[0]) != expected_tokens:
            raise ValueError(
                f"prompt K/V has {keys.shape[0]} tokens, but request {request_id} "
                f"has context length {expected_tokens}"
            )
        self.write_tokens(request_id, start_token_index=0, keys=keys, values=values)

    def append_decode_token(
        self,
        request_id: int,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> int:
        """Append one decode token to the allocator and write its K/V vector."""

        self._validate_single_token(key, value)
        token_index = self.allocator.context_length(request_id)
        self.allocator.append_token(request_id)
        self.write_token(request_id, token_index, key, value)
        return token_index

    def write_tokens(
        self,
        request_id: int,
        start_token_index: int,
        keys: torch.Tensor,
        values: torch.Tensor,
    ) -> None:
        self._validate_token_batch(keys, values)
        if start_token_index < 0:
            raise ValueError("start_token_index must be non-negative")
        token_count = int(keys.shape[0])
        if token_count == 0:
            return
        end_token_index = start_token_index + token_count
        context_len = self.allocator.context_length(request_id)
        if end_token_index > context_len:
            raise IndexError(
                f"write range [{start_token_index}, {end_token_index}) is outside "
                f"context length {context_len}"
            )

        source_offset = 0
        token_index = start_token_index
        keys = keys.to(device=self.device, dtype=self.dtype)
        values = values.to(device=self.device, dtype=self.dtype)
        while token_index < end_token_index:
            physical_block, block_offset = self.allocator.token_slot(request_id, token_index)
            tokens_this_block = min(
                self.block_size - block_offset,
                end_token_index - token_index,
            )
            source_end = source_offset + tokens_this_block
            block_end = block_offset + tokens_this_block
            self.k_cache[physical_block, :, block_offset:block_end, :] = keys[
                source_offset:source_end
            ].transpose(0, 1)
            self.v_cache[physical_block, :, block_offset:block_end, :] = values[
                source_offset:source_end
            ].transpose(0, 1)
            source_offset = source_end
            token_index += tokens_this_block

    def write_token(
        self,
        request_id: int,
        token_index: int,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> None:
        self._validate_single_token(key, value)
        physical_block, offset = self.allocator.token_slot(request_id, token_index)
        self.k_cache[physical_block, :, offset, :] = key.to(
            device=self.device,
            dtype=self.dtype,
        )
        self.v_cache[physical_block, :, offset, :] = value.to(
            device=self.device,
            dtype=self.dtype,
        )

    def write_token_batch(
        self,
        request_ids: list[int],
        token_indices: list[int],
        keys: torch.Tensor,
        values: torch.Tensor,
    ) -> None:
        """Write one token for each request with a single tensor assignment."""

        if len(request_ids) != len(token_indices):
            raise ValueError("request_ids and token_indices must have the same length")
        if keys.shape != values.shape:
            raise ValueError(f"keys and values shapes must match, got {keys.shape} and {values.shape}")
        expected_shape = (len(request_ids), self.num_heads, self.head_dim)
        if keys.ndim != 3 or tuple(keys.shape) != expected_shape:
            raise ValueError(
                "keys and values must have shape "
                f"[{len(request_ids)}, {self.num_heads}, {self.head_dim}], got {tuple(keys.shape)}"
            )
        if not request_ids:
            return

        slots = [self.allocator.token_slot(request_id, token_index) for request_id, token_index in zip(request_ids, token_indices)]
        physical_blocks = torch.tensor([slot[0] for slot in slots], device=self.device, dtype=torch.long)
        offsets = torch.tensor([slot[1] for slot in slots], device=self.device, dtype=torch.long)
        self.write_token_batch_at_slots(physical_blocks, offsets, keys, values)

    def write_token_batch_at_slots(
        self,
        physical_blocks: torch.Tensor,
        offsets: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
    ) -> None:
        """Write batched K/V using precomputed physical block slots."""

        if physical_blocks.ndim != 1 or offsets.ndim != 1:
            raise ValueError("physical_blocks and offsets must be 1D tensors")
        if int(physical_blocks.shape[0]) != int(offsets.shape[0]):
            raise ValueError("physical_blocks and offsets must have the same length")
        if keys.shape != values.shape:
            raise ValueError(f"keys and values shapes must match, got {keys.shape} and {values.shape}")
        expected_shape = (int(physical_blocks.shape[0]), self.num_heads, self.head_dim)
        if keys.ndim != 3 or tuple(keys.shape) != expected_shape:
            raise ValueError(
                "keys and values must have shape "
                f"[{expected_shape[0]}, {self.num_heads}, {self.head_dim}], got {tuple(keys.shape)}"
            )
        if expected_shape[0] == 0:
            return

        physical_blocks = physical_blocks.to(device=self.device, dtype=torch.long)
        offsets = offsets.to(device=self.device, dtype=torch.long)
        self.k_cache[physical_blocks, :, offsets, :] = keys.to(device=self.device, dtype=self.dtype)
        self.v_cache[physical_blocks, :, offsets, :] = values.to(device=self.device, dtype=self.dtype)

    def read_token(self, request_id: int, token_index: int) -> tuple[torch.Tensor, torch.Tensor]:
        physical_block, offset = self.allocator.token_slot(request_id, token_index)
        return (
            self.k_cache[physical_block, :, offset, :],
            self.v_cache[physical_block, :, offset, :],
        )

    def gather_request(self, request_id: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return contiguous K/V tensors with shape `[tokens, heads, dim]`."""

        context_len = self.allocator.context_length(request_id)
        keys = []
        values = []
        for token_index in range(context_len):
            key, value = self.read_token(request_id, token_index)
            keys.append(key)
            values.append(value)
        if not keys:
            empty = torch.empty(
                (0, self.num_heads, self.head_dim),
                dtype=self.dtype,
                device=self.device,
            )
            return empty, empty.clone()
        return torch.stack(keys, dim=0), torch.stack(values, dim=0)

    def _validate_token_batch(self, keys: torch.Tensor, values: torch.Tensor) -> None:
        expected_shape = (self.num_heads, self.head_dim)
        if keys.shape != values.shape:
            raise ValueError(f"keys and values shapes must match, got {keys.shape} and {values.shape}")
        if keys.ndim != 3 or tuple(keys.shape[1:]) != expected_shape:
            raise ValueError(
                "keys and values must have shape "
                f"[tokens, {self.num_heads}, {self.head_dim}], got {tuple(keys.shape)}"
            )

    def _validate_single_token(self, key: torch.Tensor, value: torch.Tensor) -> None:
        expected_shape = (self.num_heads, self.head_dim)
        if key.shape != value.shape:
            raise ValueError(f"key and value shapes must match, got {key.shape} and {value.shape}")
        if key.ndim != 2 or tuple(key.shape) != expected_shape:
            raise ValueError(
                f"key and value must have shape [{self.num_heads}, {self.head_dim}], "
                f"got {tuple(key.shape)}"
            )
