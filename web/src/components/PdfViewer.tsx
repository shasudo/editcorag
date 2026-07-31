import { useEffect, useRef, useState } from "react";
import * as pdfjs from "pdfjs-dist";
import workerUrl from "pdfjs-dist/build/pdf.worker.min.mjs?url";
import type { Citation } from "../api";
import { api } from "../api";
import { DocIcon, ZoomIn, ZoomOut } from "./icons";

pdfjs.GlobalWorkerOptions.workerSrc = workerUrl;

// Bounding the cache is what actually reclaims pdf.js memory: this pane is
// never unmounted in practice, so an unmount-only cleanup would never run.
const MAX_CACHED_DOCS = 3;

/**
 * Render the cited page from the original PDF and draw the source region on it.
 *
 * The server never rasterises pages: it stores a normalised bbox and pdf.js
 * renders here, which keeps ~1 GB of page images off disk for an 800-page
 * document and makes the highlight resolution-independent. Because the bbox is
 * 0..1 with a top-left origin, scaling to the rendered canvas is a plain
 * multiply.
 */
export function PdfViewer({ citation }: { citation: Citation | null }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const paneRef = useRef<HTMLDivElement>(null);
  const [zoom, setZoom] = useState(1);
  const [size, setSize] = useState({ width: 0, height: 0 });
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  // Cache the *promise*, not the resolved document: two rapid citation clicks
  // would otherwise both miss and start duplicate downloads of the same PDF.
  const docCache = useRef<Map<number, Promise<pdfjs.PDFDocumentProxy>>>(new Map());
  const renderTask = useRef<pdfjs.RenderTask | null>(null);

  useEffect(() => {
    if (!citation) return;
    let cancelled = false;

    (async () => {
      setLoading(true);
      setError(null);
      try {
        const docId = citation.doc_id;
        let pending = docCache.current.get(docId);
        if (!pending) {
          pending = pdfjs.getDocument(api.pdfUrl(docId)).promise;
          pending.catch(() => {
            if (docCache.current.get(docId) === pending) docCache.current.delete(docId);
          });
          docCache.current.set(docId, pending);
          for (const [id, old] of docCache.current) {
            if (docCache.current.size <= MAX_CACHED_DOCS) break;
            if (id === docId) continue;
            docCache.current.delete(id);
            void old.then((d) => d.destroy()).catch(() => undefined);
          }
        }
        const doc = await pending;
        if (cancelled) return;
        const page = await doc.getPage(citation.page_no);
        if (cancelled) return;

        const avail = (paneRef.current?.clientWidth ?? 720) - 32;
        const natural = page.getViewport({ scale: 1 });
        const viewport = page.getViewport({
          scale: Math.max(0.2, (avail / natural.width) * zoom),
        });
        const canvas = canvasRef.current;
        if (!canvas) return;
        const context = canvas.getContext("2d");
        if (!context) return;

        const dpr = window.devicePixelRatio || 1;
        canvas.width = Math.floor(viewport.width * dpr);
        canvas.height = Math.floor(viewport.height * dpr);
        canvas.style.width = `${viewport.width}px`;
        canvas.style.height = `${viewport.height}px`;
        context.setTransform(dpr, 0, 0, dpr, 0, 0);
        setSize({ width: viewport.width, height: viewport.height });

        const task = page.render({ canvasContext: context, viewport, canvas });
        renderTask.current = task;
        await task.promise;
        if (cancelled) return;
        if (renderTask.current === task) renderTask.current = null;
      } catch (err) {
        if (!cancelled && (err as Error)?.name !== "RenderingCancelledException") {
          setError((err as Error).message);
        }
      } finally {
        if (!cancelled) setLoading(false);
      }
    })();

    return () => {
      cancelled = true;
      renderTask.current?.cancel();
      renderTask.current = null;
    };
  }, [citation, zoom]);

  if (!citation) {
    return (
      <div className="flex h-full flex-col items-center justify-center gap-2 p-8 text-center">
        <div className="grid h-10 w-10 place-items-center rounded-full bg-[var(--surface-2)] text-[var(--text-3)]">
          <DocIcon className="h-4 w-4" />
        </div>
        <p className="max-w-[220px] text-sm text-[var(--text-3)]">
          Click a source to see it highlighted on the original page.
        </p>
      </div>
    );
  }

  const box = citation.bbox;

  return (
    <div className="flex h-full flex-col">
      <div className="flex h-12 shrink-0 items-center gap-2.5 border-b border-[var(--line)] px-4">
        <span className="chip !border-[var(--accent)]/30 !bg-[var(--accent-soft)] !text-[var(--accent)]">
          {citation.page_label}
        </span>
        {citation.section && (
          <span className="min-w-0 truncate text-xs text-[var(--text-2)]" title={citation.section}>
            {citation.section}
          </span>
        )}
        <span className="ml-auto flex shrink-0 items-center gap-0.5">
          <button
            type="button"
            aria-label="Zoom out"
            className="rounded-md p-1.5 text-[var(--text-2)] transition hover:bg-[var(--surface-2)] hover:text-[var(--text)]"
            onClick={() => setZoom((z) => Math.max(0.5, +(z - 0.2).toFixed(1)))}
          >
            <ZoomOut className="h-3.5 w-3.5" />
          </button>
          <span aria-live="polite" className="tabnum w-11 text-center text-[11px] text-[var(--text-3)]">
            {Math.round(zoom * 100)}%
          </span>
          <button
            type="button"
            aria-label="Zoom in"
            className="rounded-md p-1.5 text-[var(--text-2)] transition hover:bg-[var(--surface-2)] hover:text-[var(--text)]"
            onClick={() => setZoom((z) => Math.min(3, +(z + 0.2).toFixed(1)))}
          >
            <ZoomIn className="h-3.5 w-3.5" />
          </button>
        </span>
      </div>

      <div ref={paneRef} className="min-h-0 flex-1 overflow-auto bg-[var(--bg)] p-4">
        {error && (
          <div className="rounded-lg border border-[var(--danger)]/30 bg-[var(--danger)]/10 p-3 text-sm text-[var(--danger)]">
            {error}
          </div>
        )}
        <div className="relative mx-auto w-fit overflow-hidden rounded-lg shadow-[var(--shadow-lg)] ring-1 ring-[var(--line)]">
          <canvas ref={canvasRef} className="block bg-white" />
          {box && size.width > 0 && (
            <div
              className="pointer-events-none absolute border-2 border-amber-500 bg-amber-400/25"
              style={{
                left: box[0] * size.width,
                top: box[1] * size.height,
                width: (box[2] - box[0]) * size.width,
                height: (box[3] - box[1]) * size.height,
              }}
            />
          )}
          {loading && (
            <div className="absolute inset-0 flex items-center justify-center bg-[var(--surface)]/70 text-xs text-[var(--text-2)] backdrop-blur-[1px]">
              rendering…
            </div>
          )}
        </div>

        {citation.asset_path && (
          <div className="mx-auto mt-4 w-fit">
            <div className="mb-1.5 text-[10px] font-medium uppercase tracking-wider text-[var(--text-3)]">
              Extracted figure
            </div>
            <img
              src={api.assetUrl(citation.asset_path)}
              alt="extracted figure"
              className="max-w-md rounded-lg border border-[var(--line)] bg-white shadow-sm"
            />
          </div>
        )}
      </div>
    </div>
  );
}
