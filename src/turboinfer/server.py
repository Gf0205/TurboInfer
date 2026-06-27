from __future__ import annotations

import argparse
import time
from dataclasses import asdict, dataclass
from threading import Lock
from typing import Literal

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from turboinfer.engine import KVCacheEngine, NaiveEngine


class CompletionRequest(BaseModel):
    model: str | None = None
    prompt: str
    max_tokens: int = Field(default=32, ge=1)
    temperature: float = Field(default=0.0)
    engine: Literal["naive", "kv-cache"] = "kv-cache"


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class CompletionChoice(BaseModel):
    index: int
    text: str
    finish_reason: str


class CompletionResponse(BaseModel):
    id: str
    object: str
    model: str
    choices: list[CompletionChoice]
    usage: Usage
    metrics: dict[str, object]


class HealthResponse(BaseModel):
    status: str
    loaded_model: str | None
    loaded_engine: str | None


@dataclass
class LoadedEngine:
    model_name: str
    engine_name: str
    engine: NaiveEngine | KVCacheEngine


class EngineRegistry:
    def __init__(self, default_model: str | None, device: str, trust_remote_code: bool) -> None:
        self.default_model = default_model
        self.device = device
        self.trust_remote_code = trust_remote_code
        self._loaded: LoadedEngine | None = None
        self._lock = Lock()

    def get(self, requested_model: str | None, engine_name: str) -> LoadedEngine:
        model_name = requested_model or self.default_model
        if model_name is None:
            raise HTTPException(
                status_code=400,
                detail="model must be provided in the request or via --model at startup",
            )

        with self._lock:
            if (
                self._loaded is not None
                and self._loaded.model_name == model_name
                and self._loaded.engine_name == engine_name
            ):
                return self._loaded

            engine_cls = KVCacheEngine if engine_name == "kv-cache" else NaiveEngine
            engine = engine_cls(
                model_name=model_name,
                device=self.device,
                trust_remote_code=self.trust_remote_code,
            )
            self._loaded = LoadedEngine(
                model_name=model_name,
                engine_name=engine_name,
                engine=engine,
            )
            return self._loaded

    def loaded_model(self) -> str | None:
        return self._loaded.model_name if self._loaded else None

    def loaded_engine(self) -> str | None:
        return self._loaded.engine_name if self._loaded else None


def create_app(
    default_model: str | None = None,
    device: str = "auto",
    trust_remote_code: bool = False,
) -> FastAPI:
    app = FastAPI(title="TurboInfer", version="0.1.0")
    registry = EngineRegistry(
        default_model=default_model,
        device=device,
        trust_remote_code=trust_remote_code,
    )

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse(
            status="ok",
            loaded_model=registry.loaded_model(),
            loaded_engine=registry.loaded_engine(),
        )

    @app.post("/v1/completions", response_model=CompletionResponse)
    def completions(request: CompletionRequest) -> CompletionResponse:
        if request.temperature != 0.0:
            raise HTTPException(
                status_code=400,
                detail="TurboInfer v0 only supports greedy decoding with temperature=0",
            )

        loaded = registry.get(request.model, request.engine)
        started = time.perf_counter()
        result = loaded.engine.generate(
            prompt=request.prompt,
            max_new_tokens=request.max_tokens,
        )
        served_seconds = time.perf_counter() - started
        metrics = result.metrics.to_dict()
        metrics["served_seconds"] = served_seconds

        return CompletionResponse(
            id=f"cmpl-{time.time_ns()}",
            object="text_completion",
            model=loaded.model_name,
            choices=[
                CompletionChoice(
                    index=0,
                    text=result.text,
                    finish_reason="length",
                )
            ],
            usage=Usage(
                prompt_tokens=result.metrics.prompt_tokens,
                completion_tokens=result.metrics.output_tokens,
                total_tokens=result.metrics.prompt_tokens + result.metrics.output_tokens,
            ),
            metrics=metrics,
        )

    return app


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the TurboInfer HTTP server.")
    parser.add_argument("--model", default=None, help="Default Hugging Face model name or path.")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, etc.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    app = create_app(
        default_model=args.model,
        device=args.device,
        trust_remote_code=args.trust_remote_code,
    )
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()

