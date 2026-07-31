"""EditcoRAG command line.

`editcorag doctor` is the first thing to run on a new machine: it reports what
this platform can actually do (device, OCR backends and their language
coverage, model cache state, LLM reachability) instead of failing halfway
through an 800-page ingest.
"""

from __future__ import annotations

import json
import os
import socket
import sys
import threading
import time
from collections import Counter
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from . import __version__
from .settings import AnswerMode, get_settings
from .util.logging import configure_logging

app = typer.Typer(
    no_args_is_help=True,
    add_completion=False,
    help="Fully-local RAG over large, mixed text/scanned PDFs.",
)
console = Console()

def _kv(title: str, *rows: tuple[str, str]) -> Table:
    """Two-column key/value table, the shape most of these commands print."""
    table = Table(title=title, show_header=False, box=None)
    table.add_column(style="bold")
    table.add_column()
    for key, value in rows:
        table.add_row(key, value)
    return table


def _table(title: str, *cols: str) -> Table:
    table = Table(title=title)
    for col in cols:
        table.add_column(col)
    return table


_OK = "[green]OK[/green]"
_WARN = "[yellow]WARN[/yellow]"
_BAD = "[red]FAIL[/red]"


@app.command()
def version() -> None:
    """Print the EditcoRAG version."""
    console.print(f"editcorag {__version__}")


@app.command()
def paths() -> None:
    """Show where EditcoRAG keeps its data."""
    cfg = get_settings()
    table = _table("Paths", "what", "path", "exists")
    for label, path in [
        ("data dir", cfg.data_dir),
        ("pdfs", cfg.pdf_dir),
        ("lancedb", cfg.lance_dir),
        ("assets", cfg.assets_dir),
        ("sqlite", cfg.sqlite_path),
        ("checkpoints", cfg.checkpoint_path),
        ("web dist", cfg.web_dist),
    ]:
        table.add_row(label, str(path), "yes" if path.exists() else "no")
    console.print(table)


@app.command()
def doctor(
    check_llm: bool = typer.Option(True, help="Probe the configured LLM endpoint."),
    check_structured: bool = typer.Option(
        False,
        help="Also test whether the LLM can hold a JSON schema (needed by the grader nodes).",
    ),
    strict: bool = typer.Option(False, help="Exit non-zero if anything is not OK."),
) -> None:
    """Report what this machine can do, and what is missing."""
    cfg = get_settings()
    configure_logging(cfg.log_level, cfg.log_json)
    problems: list[str] = []

    from .util.device import device_report

    dev = device_report()
    env = _kv(
        "Environment",
        ("platform", str(dev["platform"])),
        ("python", str(dev["python"])),
        ("torch", str(dev.get("torch"))),
        ("device (auto)", f"[cyan]{dev['resolved']}[/cyan]"),
    )
    if dev.get("cuda_devices"):
        env.add_row("cuda devices", ", ".join(dev["cuda_devices"]))
    env.add_row("workers", str(cfg.max_workers))
    console.print(env)
    console.print()

    from .ingest import ocr

    unreadable = ocr.unsupported(cfg.ocr_languages)
    table = _kv("OCR (RapidOCR)")
    table.add_row("languages", ", ".join(cfg.ocr_languages))
    table.add_row("model set", ocr.model_set_for(cfg.ocr_languages))
    if unreadable:
        table.add_row("coverage", f"{_WARN} cannot read: {', '.join(unreadable)}")
        problems.append(f"OCR cannot read: {', '.join(unreadable)}")
    else:
        table.add_row("coverage", _OK)
    console.print(table)
    console.print()

    from . import setup as provisioning

    models = _table("Models", "role", "id", "cached")
    for entry in provisioning.status(cfg)["models"]:
        models.add_row(
            entry["role"],
            entry["repo"],
            _OK if entry["cached"] else "[yellow]downloads on first use[/yellow]",
        )
    console.print(models)
    console.print()

    stores = _kv("Stores")
    try:
        import lancedb

        lancedb.connect(str(cfg.lance_dir))
        stores.add_row("lancedb", f"{_OK} {cfg.lance_dir}")
    except Exception as exc:  # noqa: BLE001
        stores.add_row("lancedb", f"{_BAD} {exc}")
        problems.append(f"lancedb: {exc}")
    try:
        from .db.store import get_conn

        conn = get_conn(cfg.sqlite_path)
        n = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
        stores.add_row("sqlite", f"{_OK} {cfg.sqlite_path} ({n} documents)")
    except Exception as exc:  # noqa: BLE001
        stores.add_row("sqlite", f"{_BAD} {exc}")
        problems.append(f"sqlite: {exc}")
    console.print(stores)
    console.print()

    if check_llm:
        from .llm.models import probe_llm

        info = probe_llm(cfg)
        llm = _kv(
            "LLM (Groq)",
            ("model", info["model"]),
            ("base_url", info["base_url"]),
        )
        if info["reachable"]:
            llm.add_row("status", f"{_OK} replied: {info['detail']!r}")
        elif info["constructed"]:
            llm.add_row("status", f"{_WARN} endpoint unreachable - {info['detail']}")
            problems.append("LLM endpoint unreachable")
        else:
            llm.add_row("status", f"{_BAD} could not construct client - {info['detail']}")
            problems.append("LLM client construction failed")
        console.print(llm)

        if check_structured and info["reachable"]:
            from .llm.models import supports_structured_output

            ok, detail = supports_structured_output(cfg)
            console.print(
                f"structured output: {_OK if ok else _WARN} {detail}"
                + ("" if ok else "  -> set LOCALRAG_ENABLE_GRADERS=false")
            )
            if not ok:
                problems.append("structured output unsupported")
        console.print()

    if problems:
        console.print(f"[yellow]{len(problems)} item(s) need attention.[/yellow]")
        if strict:
            raise typer.Exit(code=1)
    else:
        console.print("[green]All checks passed.[/green]")


def _exit_when_stdin_closes() -> None:
    """Quit as soon as the launching process goes away.

    The desktop shell spawns this with a stdin pipe it never writes to. If the
    shell dies - clean quit, crash, force-quit, kill -9 - the pipe closes and
    the read returns, which is the one signal that survives every way a parent
    can disappear. Without it an orphaned worker keeps several GB of models
    resident with no window left to close it.

    The exit is deliberately abrupt: nothing is left to serve, and both stores
    are crash-safe (SQLite runs in WAL), so there is nothing to unwind.
    """

    def wait() -> None:
        try:
            sys.stdin.buffer.read()
        except Exception:  # noqa: BLE001 - any read failure means the pipe is gone
            pass
        os._exit(0)

    threading.Thread(target=wait, daemon=True, name="parent-watchdog").start()


def _free_port(host: str) -> int:
    """Ask the OS for an unused port.

    Bound and released rather than handed straight to uvicorn: uvicorn accepts
    port 0 but never reports which port it landed on, and the shell has to know
    where to point the window.
    """
    with socket.socket() as probe:
        probe.bind((host, 0))
        return int(probe.getsockname()[1])


def _parse_pages(spec: str | None, page_count: int) -> range | None:
    """'1-50' / '7' / None -> a zero-based range over pages."""
    if not spec:
        return None
    spec = spec.strip()
    try:
        if "-" in spec:
            lo_s, hi_s = spec.split("-", 1)
            lo, hi = int(lo_s), int(hi_s)
        else:
            lo = hi = int(spec)
    except ValueError:
        raise typer.BadParameter(f"cannot parse page range {spec!r}; use '1-50' or '7'") from None
    lo = max(1, lo)
    hi = min(page_count, hi)
    if lo > hi:
        raise typer.BadParameter(f"empty page range {spec!r}")
    return range(lo - 1, hi)


@app.command()
def audit(
    pdf: Path = typer.Argument(..., exists=True, dir_okay=False, help="PDF to inspect."),
    samples: int = typer.Option(2, help="Sample pages to show per route."),
    pages: str | None = typer.Option(None, "--pages", help="Limit to a range, e.g. 1-50."),
    sample_chars: int = typer.Option(220, help="Characters of page text to show."),
    verbose: bool = typer.Option(False, "--verbose", help="Per-page table."),
    as_json: bool = typer.Option(False, "--json", help="Machine-readable output."),
) -> None:
    """Explain how every page would be read, without extracting anything.

    Run this before a long ingest: it turns 800 opaque routing decisions into a
    histogram plus evidence, so a misrouted document is caught in seconds
    rather than after an hour of OCR.
    """
    from .ingest.inventory import PdfOpenError, inventory
    from .ingest.route import Route, histogram, needs_ocr, route_pages

    cfg = get_settings()
    configure_logging(cfg.log_level, cfg.log_json)

    started = time.perf_counter()
    try:
        import pymupdf

        with pymupdf.open(pdf) as probe:
            page_range = _parse_pages(pages, probe.page_count)
        inv = inventory(pdf, min_chars=cfg.route_min_chars, pages=page_range)
    except PdfOpenError as exc:
        console.print(f"{_BAD} {exc}")
        raise typer.Exit(code=1) from None
    decisions = route_pages(inv.pages, cfg)
    elapsed = time.perf_counter() - started

    hist = histogram(decisions)
    ocr_pages = sum(n for route, n in hist.items() if needs_ocr(Route(route)))

    if as_json:
        print(
            json.dumps(
                {
                    "file": inv.filename,
                    "sha256": inv.sha256,
                    "page_count": inv.page_count,
                    "audited": len(decisions),
                    "histogram": hist,
                    "ocr_pages": ocr_pages,
                    "pages": [
                        {
                            "page_no": d.page_no,
                            "route": d.route.value,
                            "reason": d.reason,
                            "quality": round(d.metrics.quality.score, 3),
                            "n_chars": d.metrics.n_chars,
                            "text_cover": round(d.metrics.text_cover, 3),
                            "image_cover": round(d.metrics.image_cover, 3),
                        }
                        for d in decisions
                    ],
                }
            )
        )
        return

    head = _kv(f"{inv.filename}")
    head.add_row("sha256", inv.sha256[:16] + "...")
    scope = f" (audited {len(decisions)})" if page_range else ""
    head.add_row("pages", f"{inv.page_count}{scope}")
    head.add_row("size", f"{inv.byte_size / 1_048_576:.1f} MiB")
    if inv.title:
        head.add_row("title", inv.title)
    if inv.encrypted:
        head.add_row("encrypted", "yes (opened with empty password)")
    if inv.repaired:
        head.add_row("repaired", "[yellow]yes - the xref was damaged[/yellow]")
    head.add_row("audit time", f"{elapsed:.2f}s")
    console.print(head)
    console.print()

    meaning = {
        "digital": "trust text layer; layout + tables, no OCR",
        "scanned": "render + full-page OCR",
        "hybrid": "text layer + OCR of picture regions only",
        "empty": "nothing renderable; skipped",
    }
    total = max(1, len(decisions))
    table = Table(title="Routing")
    table.add_column("route")
    table.add_column("pages", justify="right")
    table.add_column("share", justify="right")
    table.add_column("extraction strategy")
    for route, count in hist.items():
        style = "cyan" if count else "dim"
        table.add_row(
            f"[{style}]{route}[/{style}]",
            str(count),
            f"{count / total:.0%}",
            meaning[route],
        )
    console.print(table)
    console.print(
        f"pages needing OCR: [cyan]{ocr_pages}[/cyan] of {len(decisions)}"
        f"  ({ocr_pages / total:.0%})"
    )

    from .ingest import ocr

    console.print(
        f"OCR: [cyan]RapidOCR/{ocr.model_set_for(cfg.ocr_languages)}[/cyan]"
        f"  languages: {', '.join(cfg.ocr_languages)}"
    )
    for code in ocr.unsupported(cfg.ocr_languages):
        console.print(f"  {_WARN} RapidOCR cannot read {code!r}")
    console.print()

    for route in Route:
        picked = [d for d in decisions if d.route is route][:samples]
        if not picked:
            continue
        console.print(f"[bold]{route.value}[/bold] samples")
        for d in picked:
            m = d.metrics
            console.print(f"  page {d.page_no}: [dim]{d.reason}[/dim]")
            console.print(
                f"    chars={m.n_chars} text={m.text_cover:.0%} img={m.image_cover:.0%} "
                f"quality={m.quality.score:.2f} script={m.quality.script}"
            )
            if m.sample:
                console.print(f"    [italic]{m.sample[:sample_chars]}[/italic]")
        console.print()

    defects: dict[str, list[int]] = {}
    for d in decisions:
        for reason in d.metrics.quality.reasons:
            defects.setdefault(reason.split(" (")[0], []).append(d.page_no)
    if defects:
        table = Table(title="Text-layer defects")
        table.add_column("defect")
        table.add_column("pages", justify="right")
        table.add_column("share", justify="right")
        table.add_column("still indexed as-is", justify="right")
        digital_pages = {d.page_no for d in decisions if d.route is Route.DIGITAL}
        for reason, pages in sorted(defects.items(), key=lambda kv: -len(kv[1])):
            silent = len(set(pages) & digital_pages)
            table.add_row(
                reason,
                str(len(pages)),
                f"{len(pages) / total:.0%}",
                f"[yellow]{silent}[/yellow]" if silent else "0",
            )
        console.print(table)
        silent_total = len(
            {p for pages in defects.values() for p in pages} & digital_pages
        )
        if silent_total:
            console.print(
                f"{_WARN} {silent_total} page(s) have a detected text-layer defect but "
                f"still scored above LOCALRAG_ROUTE_QUALITY_MIN={cfg.route_quality_min}, "
                f"so their text is indexed as-is. Raise the threshold to force OCR "
                f"on them (slower, but recovers the corrupted characters)."
            )
        console.print()

    borderline = sorted(
        (
            d
            for d in decisions
            if d.route is Route.DIGITAL
            and d.metrics.quality.score < cfg.route_quality_min + 0.2
        ),
        key=lambda d: d.metrics.quality.score,
    )[:5]
    if borderline:
        console.print(
            f"[yellow]borderline text quality[/yellow] "
            f"(threshold {cfg.route_quality_min:.2f}) - inspect to tune "
            f"LOCALRAG_ROUTE_QUALITY_MIN:"
        )
        for d in borderline:
            reasons = "; ".join(d.metrics.quality.reasons) or "no specific defect"
            console.print(
                f"  page {d.page_no} score={d.metrics.quality.score:.2f} [dim]{reasons}[/dim]"
            )
            if d.metrics.sample:
                console.print(f"    [italic]{d.metrics.sample[:sample_chars]}[/italic]")
        console.print()

    failed = [d for d in decisions if d.metrics.error]
    if failed:
        console.print(f"[red]{len(failed)} page(s) failed inventory:[/red]")
        for d in failed[:10]:
            console.print(f"  page {d.page_no}: {d.metrics.error}")
        console.print()

    if verbose:
        per_page = Table(title="Per-page detail")
        for col, just in (
            ("page", "right"),
            ("route", "left"),
            ("chars", "right"),
            ("text%", "right"),
            ("img%", "right"),
            ("qual", "right"),
            ("script", "left"),
            ("reason", "left"),
        ):
            per_page.add_column(col, justify=just)  # type: ignore[arg-type]
        for d in decisions:
            m = d.metrics
            per_page.add_row(
                str(d.page_no),
                d.route.value,
                str(m.n_chars),
                f"{m.text_cover:.0%}",
                f"{m.image_cover:.0%}",
                f"{m.quality.score:.2f}",
                m.quality.script,
                d.reason,
            )
        console.print(per_page)


@app.command()
def extract(
    pdf: Path = typer.Argument(..., exists=True, dir_okay=False),
    pages: str = typer.Option("1-6", "--pages", help="Page range to extract, e.g. 1-6."),
    out: Path | None = typer.Option(None, "--out", help="Write per-batch Markdown here."),
    show_text: bool = typer.Option(True, help="Print chunk text."),
    chars: int = typer.Option(160, help="Characters of each chunk to print."),
) -> None:
    """Extract a page range and show exactly what would be indexed.

    This is the honest check on extraction quality: read the chunks and the
    Markdown for a handful of pages spanning every route before committing to a
    full run. Nothing is written to the database.
    """
    import pymupdf

    from .ingest import ocr
    from .ingest.inventory import PdfOpenError, file_sha256, inventory
    from .ingest.route import route_pages
    from .ingest.runner import BatchRunner, jobs_for

    cfg = get_settings()
    configure_logging(cfg.log_level, cfg.log_json)
    cfg.ensure_dirs()

    try:
        with pymupdf.open(pdf) as probe:
            page_range = _parse_pages(pages, probe.page_count)
        inv = inventory(pdf, min_chars=cfg.route_min_chars, pages=page_range)
    except PdfOpenError as exc:
        console.print(f"{_BAD} {exc}")
        raise typer.Exit(code=1) from None

    decisions = route_pages(inv.pages, cfg)
    for code in ocr.unsupported(cfg.ocr_languages):
        console.print(f"{_WARN} RapidOCR cannot read {code!r}")

    routed = [(d.page_no, d.route.value) for d in decisions]
    jobs = jobs_for(pdf, file_sha256(pdf), routed, cfg.batch_pages)
    console.print(
        f"extracting {len(routed)} page(s) in {len(jobs)} batch(es) "
        f"via [cyan]RapidOCR[/cyan] on [cyan]{cfg.embed_model}[/cyan] tokenizer"
    )

    results = []
    with console.status("converting..."):
        with BatchRunner(cfg, want_markdown=True) as runner:
            for result in runner.run(jobs):
                results.append(result)
    results.sort(key=lambda r: r.page_from)

    total_chunks = sum(len(r.chunks) for r in results)
    total_assets = sum(len(r.assets) for r in results)
    duration = sum(r.duration_s for r in results)
    per_page = duration / max(1, len(routed))

    summary = _table("Extraction", "batch", "route", "chunks", "assets", "seconds", "error")
    for r in results:
        summary.add_row(
            f"{r.page_from}-{r.page_to}",
            r.route,
            str(len(r.chunks)),
            str(len(r.assets)),
            f"{r.duration_s:.1f}",
            (r.error or "")[:60],
        )
    console.print(summary)
    line = f"{total_chunks} chunks, {total_assets} figure asset(s), {per_page:.2f}s/page"
    if len(routed) >= 12:
        line += f"  -> ~{per_page * 800 / 60:.0f} min projected for 800 pages"
    else:
        line += "  (too few pages to project; try --pages 1-24)"
    console.print(line)
    console.print()

    kinds = Counter(c.kind for r in results for c in r.chunks)
    if kinds:
        console.print("chunk kinds: " + ", ".join(f"{k}={v}" for k, v in sorted(kinds.items())))
        console.print()

    if show_text:
        for r in results:
            for c in r.chunks:
                span = (
                    f"p{c.page_no}"
                    if len(c.page_span) <= 1
                    else f"p{c.page_span[0]}-{c.page_span[-1]}"
                )
                bbox = (
                    "[" + ", ".join(f"{v:.3f}" for v in c.bbox) + "]" if c.bbox else "none"
                )
                console.print(f"[bold]{c.kind}[/bold] {span}  bbox={bbox}")
                if c.section:
                    console.print(f"  section: [dim]{c.section}[/dim]")
                if c.asset_path:
                    console.print(f"  asset: [dim]{c.asset_path}[/dim]")
                console.print(f"  {c.text[:chars]!r}")
            console.print()

    if out:
        out.mkdir(parents=True, exist_ok=True)
        for r in results:
            target = out / f"pages-{r.page_from:04d}-{r.page_to:04d}-{r.route}.md"
            target.write_text(r.markdown or "", encoding="utf-8")
        console.print(f"wrote Markdown for {len(results)} batch(es) to {out}")


@app.command()
def ingest(
    pdf: Path = typer.Argument(..., exists=True, dir_okay=False),
    force: bool = typer.Option(
        False, "--force", help="Discard existing chunks for this document and start over."
    ),
) -> None:
    """Ingest a PDF: route, extract, embed and index it.

    Safe to re-run and safe to interrupt. Finished pages are recorded in SQLite,
    so a killed run picks up where it stopped rather than starting again.
    """
    from rich.progress import (
        BarColumn,
        MofNCompleteColumn,
        Progress,
        SpinnerColumn,
        TextColumn,
        TimeElapsedColumn,
    )

    from .graphs.ingest_graph import build_ingest_graph
    from .ingest.inventory import file_sha256

    cfg = get_settings()
    configure_logging(cfg.log_level, cfg.log_json)
    cfg.ensure_dirs()

    sha = file_sha256(pdf)
    job_id = f"ingest-{sha[:12]}"

    console.print(f"ingesting [cyan]{pdf.name}[/cyan]  job={job_id}")
    summary: dict[str, object] = {}

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        graph = build_ingest_graph()
        task = progress.add_task("starting", total=None)
        for event in graph.stream(
            {"pdf_path": str(pdf), "job_id": job_id, "force": force},
            stream_mode="custom",
        ):
            stage = event.get("stage")
            if stage == "inventory":
                progress.update(
                    task, description=f"routing {event['pages']} pages", total=event["pages"]
                )
            elif stage == "route":
                hist = event.get("histogram", {})
                progress.console.print(
                    "routes: "
                    + ", ".join(f"{k}={v}" for k, v in hist.items() if v)
                    + f"  |  pending={event['pending']} skipped={event['skipped']}"
                    + (
                        f" resumed={event['already_done']}"
                        if event.get("already_done")
                        else ""
                    )
                )
                progress.update(task, description="extracting", completed=event["skipped"])
            elif stage == "extract_start":
                progress.console.print(
                    f"{event['batches']} batch(es), {event['pages']} page(s)"
                )
            elif stage == "batch":
                if event.get("ok"):
                    progress.update(
                        task,
                        completed=event.get("pages_done", 0),
                        total=event.get("pages_total"),
                        description=f"{event['route']} p{event['page_from']}-{event['page_to']}",
                    )
                else:
                    progress.console.print(
                        f"{_BAD} pages {event['page_from']}-{event['page_to']}: {event['error']}"
                    )
            elif stage == "warning":
                progress.console.print(f"{_WARN} {event['message']}")
            elif stage == "finalize":
                summary = event
                progress.update(task, description="done")

    if summary:
        table = _kv(
            "Ingest result",
            ("status", str(summary.get("status"))),
            ("chunks indexed", str(summary.get("chunks"))),
            ("failed pages", str(summary.get("failed_pages"))),
        )
        page_status = summary.get("page_status") or {}
        if isinstance(page_status, dict):
            table.add_row(
                "page status", ", ".join(f"{k}={v}" for k, v in sorted(page_status.items()))
            )
        console.print(table)
        if summary.get("failed_pages"):
            console.print(
                f"{_WARN} retry failed pages with: editcorag ingest {pdf} "
                f"(finished pages are skipped)"
            )


@app.command()
def query(
    question: str = typer.Argument(..., help="Question to ask the indexed documents."),
    mode: AnswerMode | None = typer.Option(None, help="Override settings.answer_mode."),
    show_sources: bool = typer.Option(True, help="List the cited sources."),
    show_trace: bool = typer.Option(False, "--trace", help="Show the graph's decisions."),
    chars: int = typer.Option(220, help="Characters of each source to show."),
) -> None:
    """Ask a question and get a cited answer."""
    from .graphs.query_graph import answer_question
    from .index.store import ChunkStore

    cfg = get_settings()
    configure_logging(cfg.log_level, cfg.log_json)

    try:
        store_count = ChunkStore(cfg).count()
    except RuntimeError as exc:
        console.print(f"{_BAD} {exc}")
        raise typer.Exit(code=1) from None
    if store_count == 0:
        console.print(f"{_WARN} the index is empty; run `editcorag ingest <pdf>` first")
        raise typer.Exit(code=1)

    with console.status("thinking..."):
        state = answer_question(question, cfg=cfg, mode=mode)

    console.print()
    if state.get("abstained"):
        console.print(f"[yellow]{state.get('answer')}[/yellow]")
    else:
        console.print(state.get("answer") or "(no answer)")
    console.print()

    citations = state.get("citations") or []
    if show_sources and citations:
        table = _table("Sources", "ref", "page", "kind", "score", "text")
        for c in citations:
            table.add_row(
                c["ref"],
                c["page_label"],
                c["kind"],
                f"{c['score']:.3f}",
                " ".join((c["text"] or "").split())[:chars],
            )
        console.print(table)
    elif show_sources:
        console.print("[dim]no sources cited[/dim]")

    if show_trace:
        console.print()
        console.print(
            f"[dim]top rerank score: {float(state.get('top_score', 0.0)):.4f} "
            f"(abstain below {cfg.abstain_score_min})  "
            f"rewrites: {state.get('rewrites', 0)}  "
            f"regenerations: {state.get('regenerations', 0)}  "
            f"grounded: {state.get('grounded')}[/dim]"
        )
        for note in state.get("notes") or []:
            console.print(f"  [dim]- {note}[/dim]")


@app.command()
def init() -> None:
    """Create the data directories and initialise the database."""
    cfg = get_settings()
    cfg.ensure_dirs()
    from .db.store import get_conn

    get_conn(cfg.sqlite_path)
    console.print(f"[green]initialised[/green] {cfg.data_dir}")


@app.command()
def serve(
    host: str = typer.Option("", help="Bind address (default: settings)."),
    port: int | None = typer.Option(None, help="Port; 0 picks any free one."),
    reload: bool = typer.Option(False, help="Auto-reload on code changes (development)."),
    exit_with_parent: bool = typer.Option(
        False, "--exit-with-parent", help="Quit when the launching process does (desktop shell)."
    ),
) -> None:
    """Run the API and the web UI from one process.

    Pass --port 0 to bind any free port. The chosen port is printed to stdout as
    a single JSON line prefixed LOCALRAG_READY, which is how the desktop shell
    learns where to connect - a fixed port would collide with whatever else the
    user happens to be running.
    """
    import uvicorn

    cfg = get_settings()
    cfg.ensure_dirs()
    if exit_with_parent:
        _exit_when_stdin_closes()
    bind_host = host or cfg.host
    bind_port = cfg.port if port is None else port
    if bind_port == 0:
        bind_port = _free_port(bind_host)
    if not cfg.web_dist.is_dir():
        console.print(
            f"{_WARN} frontend not built - API only. Build it with: "
            f"cd web && pnpm install && pnpm build"
        )
    # Plain stdout, never rich: this line is parsed, and rich would wrap ANSI
    # colour codes around it whenever the terminal supports them.
    print(
        f"LOCALRAG_READY {json.dumps({'host': bind_host, 'port': bind_port})}",
        flush=True,
    )
    console.print(f"serving on [cyan]http://{bind_host}:{bind_port}[/cyan]  (API docs at /docs)")
    uvicorn.run(
        "server.api:app",
        host=bind_host,
        port=bind_port,
        reload=reload,
        log_level=cfg.log_level.lower(),
    )


@app.command("eval")
def evaluate(
    questions: Path = typer.Option(
        Path("eval/questions.yaml"), "--questions", exists=True, dir_okay=False
    ),
    k: int = typer.Option(8, help="Cut-off for Recall@k / nDCG."),
    ablate_flag: bool = typer.Option(
        False, "--ablate", help="Compare dense / bm25 / hybrid / hybrid+rerank."
    ),
    answers: bool = typer.Option(
        False, "--answers", help="Also run end-to-end answer checks (needs the LLM, slower)."
    ),
    export: Path | None = typer.Option(
        None, "--export", help="Write JSONL for an external scorer such as RAGAS."
    ),
    detail: bool = typer.Option(False, "--detail", help="Show per-question results."),
) -> None:
    """Measure retrieval and answer quality against a gold set.

    Retrieval metrics are deterministic and need no LLM, so they are the ones to
    tune chunking, embeddings and reranking against.
    """
    from .rag.evaluate import ablate, export_jsonl, load_cases, score_answers, score_retrieval

    cfg = get_settings()
    configure_logging(cfg.log_level, cfg.log_json)
    cases = load_cases(questions)
    graded = [c for c in cases if c.pages]
    console.print(
        f"{len(cases)} case(s): {len(graded)} with gold pages, "
        f"{sum(1 for c in cases if c.should_abstain)} expecting abstention"
    )

    if ablate_flag:
        table = _table(
            f"Retrieval ablation (k={k})", "strategy", "recall", "mrr", "ndcg", "found"
        )
        results = ablate(cases, cfg, k=k)
        best = max(results.items(), key=lambda kv: kv[1].recall_at_k)[0] if results else ""
        for name, score in results.items():
            row = score.as_row()
            label = f"[bold cyan]{name}[/bold cyan]" if name == best else name
            table.add_row(
                label, str(row["recall"]), str(row["mrr"]), str(row["ndcg"]), row["found"]
            )
        console.print(table)
        console.print(f"best by recall: [cyan]{best}[/cyan]")
    else:
        score, rows = score_retrieval(cases, cfg, k=k)
        table = _kv(
            f"Retrieval (k={k})", *((key, str(value)) for key, value in score.as_row().items())
        )
        console.print(table)
        if detail:
            per = _table("Per question", "question", "gold", "retrieved", "rank", "recall")
            for r in rows:
                per.add_row(
                    r["question"][:48],
                    ",".join(map(str, r["gold_pages"])),
                    ",".join(map(str, r["retrieved_pages"][:6])),
                    str(r["first_hit_rank"] or "-"),
                    str(r["recall"]),
                )
            console.print(per)

    if answers:
        summary, rows = score_answers(cases, cfg)
        table = _kv("Answers", *((key, str(value)) for key, value in summary.items()))
        console.print(table)
        if detail:
            for r in rows:
                mark = "ok" if r["expected_abstain"] == r["abstained"] else "MISMATCH"
                console.print(
                    f"[dim]{mark}[/dim] {r['question'][:60]} -> "
                    f"cites {r['cited_pages']} | {r['answer'][:90]!r}"
                )

    if export:
        n = export_jsonl(cases, export, cfg)
        console.print(f"wrote {n} record(s) to {export}")
        console.print(
            "[dim]score with RAGAS out-of-band:\n"
            "  uvx --with ragas --with datasets --with langchain-openai "
            "python eval/ragas_suite.py "
            f"{export}[/dim]"
        )


if __name__ == "__main__":  # pragma: no cover
    app()
