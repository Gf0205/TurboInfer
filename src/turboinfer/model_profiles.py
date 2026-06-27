from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class ModelProfile:
    name: str
    hidden_size: int
    num_q_heads: int
    num_kv_heads: int
    head_dim: int
    block_size: int = 16

    @property
    def q_out_features(self) -> int:
        return self.num_q_heads * self.head_dim

    @property
    def kv_out_features(self) -> int:
        return self.num_kv_heads * self.head_dim

    @property
    def gqa_group_size(self) -> int:
        if self.num_q_heads % self.num_kv_heads != 0:
            raise ValueError(
                f"profile {self.name} has q_heads={self.num_q_heads} not divisible "
                f"by kv_heads={self.num_kv_heads}"
            )
        return self.num_q_heads // self.num_kv_heads

    @property
    def uses_gqa(self) -> bool:
        return self.num_q_heads != self.num_kv_heads

    def to_dict(self) -> dict[str, object]:
        return asdict(self) | {
            "q_out_features": self.q_out_features,
            "kv_out_features": self.kv_out_features,
            "gqa_group_size": self.gqa_group_size,
            "uses_gqa": self.uses_gqa,
        }


MODEL_PROFILES: dict[str, ModelProfile] = {
    "qwen2.5-0.5b": ModelProfile(
        name="qwen2.5-0.5b",
        hidden_size=896,
        num_q_heads=14,
        num_kv_heads=2,
        head_dim=64,
        block_size=16,
    ),
    "qwen3-0.6b": ModelProfile(
        name="qwen3-0.6b",
        hidden_size=1024,
        num_q_heads=16,
        num_kv_heads=8,
        head_dim=128,
        block_size=16,
    ),
}


def get_model_profile(name: str) -> ModelProfile:
    normalized = name.lower()
    try:
        return MODEL_PROFILES[normalized]
    except KeyError as exc:
        available = ", ".join(sorted(MODEL_PROFILES))
        raise KeyError(f"unknown model profile {name!r}; available profiles: {available}") from exc
