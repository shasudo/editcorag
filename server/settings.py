"""Central configuration.

Every tunable lives here so that behaviour is reproducible from a single .env
file. Nothing else in the codebase should read os.environ directly.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import AliasChoices, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DeviceName = Literal["auto", "cuda", "mps", "cpu"]
AnswerMode = Literal["fast", "careful"]


def _default_workers() -> int:
    """Leave headroom so a fanless laptop does not sit on the thermal ceiling."""
    return max(1, min(4, (os.cpu_count() or 4) - 2))


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="LOCALRAG_",
        env_file=(PROJECT_ROOT / ".env", ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        # Without this a field carrying a validation_alias can only be set via
        # that alias, so Settings(llm_api_key=...) was silently dropped by
        # extra="ignore" - tests and callers got the .env value regardless.
        populate_by_name=True,
    )

    data_dir: Path = PROJECT_ROOT / "data"
    # Where the built SPA lives. A field rather than a path derived from this
    # file's location, because in the packaged app `server` is installed into
    # site-packages and the UI sits somewhere else entirely.
    web_dist: Path = PROJECT_ROOT / "web" / "dist"

    # Answering runs on Groq. Its OpenAI-compatible endpoint is the default, so
    # llm_base_url only needs setting to point at a proxy.
    llm_model: str = "openai/gpt-oss-120b"
    llm_base_url: str | None = None
    # GROQ_API_KEY is accepted too: it is the name the key is issued under, and
    # what anyone following Groq's own docs will already have set.
    llm_api_key: str = Field(
        "", validation_alias=AliasChoices("LOCALRAG_LLM_API_KEY", "GROQ_API_KEY")
    )
    llm_temperature: float = 0.0
    llm_timeout_s: float = 180.0
    llm_max_retries: int = 2

    embed_model: str = "BAAI/bge-m3"
    embed_batch_size: int = 8
    embed_device: DeviceName = "auto"
    embed_normalize: bool = True
    embed_max_seq_length: int = 1024

    rerank_model: str = "BAAI/bge-reranker-v2-m3"
    rerank_device: DeviceName = "auto"
    rerank_top_n: int = 8
    rerank_batch_size: int = 16

    dense_k: int = 50
    fts_k: int = 50
    rrf_k: int = 60
    fts_tokenizer: str = "icu"
    fts_stem: bool = False
    fts_remove_stopwords: bool = False
    fts_language: str = "English"
    ann_index_min_rows: int = 50_000
    context_token_budget: int = 6000
    abstain_score_min: float = Field(0.30, ge=0.0, le=1.0)

    answer_mode: AnswerMode = "careful"
    enable_graders: bool = True
    max_query_rewrites: int = 2
    max_regenerations: int = 1

    # RapidOCR reads english, latin and chinese, and loads one set per pass.
    ocr_languages: list[str] = Field(default_factory=lambda: ["en"])

    route_min_chars: int = 50
    route_quality_min: float = 0.60
    route_scanned_image_cover: float = 0.50
    route_hybrid_image_cover: float = 0.15
    route_hybrid_text_cover: float = 0.02

    pipeline_version: str = "1"
    batch_pages: int = Field(6, ge=1)
    max_workers: int = Field(default_factory=_default_workers)
    page_timeout_s: int = 180
    max_page_attempts: int = 2
    figure_min_area: float = 0.01
    dedup_similarity: float = Field(0.90, gt=0.0, le=1.0)
    max_image_pixels: int = 40_000_000
    table_mode: Literal["fast", "accurate"] = "accurate"
    figure_images_scale: float = 2.5
    ingest_in_process: bool = False
    worker_max_batches: int = 25

    chunk_max_tokens: int = 600
    chunk_merge_peers: bool = True

    host: str = "127.0.0.1"
    port: int = 8000
    cors_origins: list[str] = Field(default_factory=lambda: ["http://localhost:5173"])

    log_level: str = "INFO"
    log_json: bool = False

    @field_validator("ocr_languages", "cors_origins", mode="before")
    @classmethod
    def _split_csv(cls, v: object) -> object:
        """Accept both JSON lists and plain comma-separated env values."""
        if isinstance(v, str):
            s = v.strip()
            if not s:
                return []
            if s.startswith("["):
                return v
            return [part.strip() for part in s.split(",") if part.strip()]
        return v

    @field_validator("llm_model", mode="before")
    @classmethod
    def _strip_provider_prefix(cls, v: object) -> object:
        """Accept the older 'provider:model' spelling.

        Model ids on Groq contain slashes but never colons, so anything before a
        colon is a leftover provider prefix from when several backends were
        supported. Dropping it keeps existing .env files working instead of
        failing with a confusing 'model not served'.
        """
        if isinstance(v, str) and ":" in v:
            return v.split(":", 1)[1]
        return v

    @field_validator("data_dir", mode="before")
    @classmethod
    def _expand(cls, v: object) -> object:
        """Expand '~': Path('~/data') would otherwise be a literal directory named '~'."""
        if isinstance(v, str):
            return Path(v).expanduser()
        if isinstance(v, Path):
            return v.expanduser()
        return v

    @property
    def worker_threads(self) -> int:
        """Native threads each extraction worker may use.

        Single source of truth: several worker processes each spawning
        cpu_count threads is how a laptop ends up thrashing, so the budget is
        split across them. Both the OMP/MKL env guard and docling's own
        accelerator options read this - setting them from different
        expressions let the explicit one silently override the guard.
        """
        return max(1, self.max_workers // 2)

    @property
    def pdf_dir(self) -> Path:
        return self.data_dir / "pdfs"

    @property
    def lance_dir(self) -> Path:
        return self.data_dir / "lance"

    @property
    def assets_dir(self) -> Path:
        return self.data_dir / "assets"

    @property
    def sqlite_path(self) -> Path:
        return self.data_dir / "localrag.db"

    @property
    def checkpoint_path(self) -> Path:
        """Chat thread memory, kept separate from the app's own state.

        Only the query graph checkpoints; ingest resumes from the pages table.
        """
        return self.data_dir / "checkpoints.db"

    def ensure_dirs(self) -> None:
        for p in (self.data_dir, self.pdf_dir, self.lance_dir, self.assets_dir):
            p.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Process-wide singleton; .env edits need a restart to take effect."""
    return Settings()
