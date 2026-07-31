import { useCallback, useEffect, useRef, useState } from "react";
import type { DocumentInfo, JobProgress } from "../api";
import { api, watchJob } from "../api";
import { CloseIcon, DocIcon, PlusIcon } from "./icons";

const ROUTE_HELP: Record<string, string> = {
  digital: "read straight from the text layer",
  scanned: "no text layer — read by OCR",
  hybrid: "text layer plus OCR of picture regions",
  empty: "nothing renderable — skipped",
};

const ROUTE_TONE: Record<string, string> = {
  digital: "text-[var(--ok)] border-[var(--ok)]/25 bg-[var(--ok)]/10",
  scanned: "text-[var(--accent)] border-[var(--accent)]/25 bg-[var(--accent)]/10",
  hybrid: "text-[var(--warn)] border-[var(--warn)]/25 bg-[var(--warn)]/10",
};

export function DocumentList({
  selected,
  onToggle,
  onDocumentsChanged,
}: {
  selected: number[];
  onToggle: (id: number) => void;
  onDocumentsChanged: () => void;
}) {
  const [docs, setDocs] = useState<DocumentInfo[]>([]);
  const [job, setJob] = useState<JobProgress | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [dragging, setDragging] = useState(false);
  const fileRef = useRef<HTMLInputElement>(null);
  const stopWatching = useRef<(() => void) | null>(null);

  const refresh = useCallback(async () => {
    try {
      setDocs(await api.documents());
      onDocumentsChanged();
    } catch (e) {
      setError((e as Error).message);
    }
  }, [onDocumentsChanged]);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  useEffect(() => () => stopWatching.current?.(), []);

  const upload = async (file: File) => {
    setError(null);
    try {
      const { job_id } = await api.upload(file);
      setJob({
        id: job_id,
        status: "queued",
        pages_total: 0,
        pages_done: 0,
        message: null,
        error: null,
      });
      stopWatching.current?.();
      stopWatching.current = watchJob(
        job_id,
        setJob,
        (err) => {
          stopWatching.current = null;
          setJob(null);
          if (err) setError(err);
          void refresh();
        },
      );
    } catch (e) {
      setError((e as Error).message);
    }
  };

  const pct =
    job && job.pages_total > 0 ? Math.round((job.pages_done / job.pages_total) * 100) : 0;

  return (
    <div
      className="flex h-full min-h-0 flex-col"
      onDragOver={(e) => {
        e.preventDefault();
        setDragging(true);
      }}
      onDragLeave={() => setDragging(false)}
      onDrop={(e) => {
        e.preventDefault();
        setDragging(false);
        const f = Array.from(e.dataTransfer.files).find((x) =>
          x.name.toLowerCase().endsWith(".pdf"),
        );
        if (f) void upload(f);
      }}
    >
      <div className="p-3">
        <input
          ref={fileRef}
          type="file"
          accept="application/pdf"
          className="hidden"
          onChange={(e) => {
            const file = e.target.files?.[0];
            if (file) void upload(file);
            e.target.value = "";
          }}
        />
        <button
          type="button"
          onClick={() => fileRef.current?.click()}
          className={`btn w-full border-dashed transition ${
            dragging
              ? "border-[var(--accent)] bg-[var(--accent-soft)] text-[var(--accent)]"
              : "btn-ghost"
          }`}
        >
          <PlusIcon className="h-3.5 w-3.5" />
          {dragging ? "Drop to add" : "Add PDF"}
        </button>

        {error && (
          <div className="mt-2 rounded-lg border border-[var(--danger)]/30 bg-[var(--danger)]/10 px-2.5 py-2 text-xs text-[var(--danger)]">
            {error}
          </div>
        )}
      </div>

      {job && (
        <div className="animate-rise mx-3 mb-3 rounded-lg border border-[var(--line)] bg-[var(--surface-2)] p-2.5">
          <div className="mb-1.5 flex items-baseline justify-between text-xs">
            <span className="font-medium">Indexing</span>
            <span className="tabnum text-[var(--text-3)]">
              {job.pages_done}/{job.pages_total || "?"}
            </span>
          </div>
          <div className="h-1 overflow-hidden rounded-full bg-[var(--line)]">
            <div
              className="h-full rounded-full bg-[var(--accent)] transition-all duration-500"
              style={{ width: `${pct}%` }}
            />
          </div>
          <div className="mt-1.5 truncate text-[11px] text-[var(--text-3)]">
            {job.error ?? job.message ?? job.status}
            {job.page_status?.failed ? ` · ${job.page_status.failed} failed` : ""}
          </div>
        </div>
      )}

      <div className="min-h-0 flex-1 overflow-y-auto px-3 pb-3">
        {docs.length === 0 && !job && (
          <p className="px-2 py-8 text-center text-xs leading-relaxed text-[var(--text-3)]">
            No documents yet.
            <br />
            Drop a PDF anywhere here.
          </p>
        )}

        <div className="flex flex-col gap-2">
          {docs.map((doc) => {
            const active = selected.includes(doc.id);
            return (
              <div
                key={doc.id}
                className={`card group relative cursor-pointer p-2.5 transition ${
                  active
                    ? "border-[var(--accent)] bg-[var(--accent-soft)]"
                    : "hover:border-[var(--line)] hover:bg-[var(--surface-2)]"
                }`}
                onClick={() => onToggle(doc.id)}
                role="button"
                tabIndex={0}
                aria-pressed={active}
                onKeyDown={(e) => {
                  if (e.key === "Enter" || e.key === " ") {
                    e.preventDefault();
                    onToggle(doc.id);
                  }
                }}
              >
                <div className="flex items-start gap-2">
                  <DocIcon
                    className={`mt-0.5 h-3.5 w-3.5 shrink-0 ${
                      active ? "text-[var(--accent)]" : "text-[var(--text-3)]"
                    }`}
                  />
                  <span
                    className="min-w-0 flex-1 truncate text-[13px] font-medium"
                    title={doc.filename}
                  >
                    {doc.filename}
                  </span>
                  <button
                    type="button"
                    aria-label={`Remove ${doc.filename}`}
                    className="shrink-0 rounded p-0.5 text-[var(--text-3)] opacity-0 transition hover:bg-[var(--surface-3)] hover:text-[var(--danger)] focus-visible:opacity-100 group-hover:opacity-100"
                    onClick={(e) => {
                      e.stopPropagation();
                      void api
                        .remove(doc.id)
                        .then(refresh)
                        .catch((err) => setError((err as Error).message));
                    }}
                  >
                    <CloseIcon className="h-3 w-3" />
                  </button>
                </div>

                <div className="mt-1 pl-[22px] text-[11px] text-[var(--text-3)]">
                  <span className="tabnum">{doc.pages}</span> pages ·{" "}
                  <span className="tabnum">{doc.chunks}</span> chunks
                  {doc.status !== "done" && (
                    <span
                      className={
                        doc.status === "partial"
                          ? " text-[var(--warn)]"
                          : " text-[var(--accent)]"
                      }
                    >
                      {" "}
                      · {doc.status}
                    </span>
                  )}
                </div>

                <div className="mt-1.5 flex flex-wrap gap-1 pl-[22px]">
                  {Object.entries(doc.routes || {})
                    .filter(([, n]) => n > 0)
                    .sort((a, b) => b[1] - a[1])
                    .map(([route, n]) => (
                      <span
                        key={route}
                        title={ROUTE_HELP[route] ?? route}
                        className={`chip !py-0 !text-[10px] ${ROUTE_TONE[route] ?? ""}`}
                      >
                        {route} <span className="tabnum opacity-75">{n}</span>
                      </span>
                    ))}
                </div>

                {doc.page_status?.failed ? (
                  <div className="mt-1 pl-[22px] text-[10px] text-[var(--warn)]">
                    {doc.page_status.failed} page(s) failed — re-run ingest to retry
                  </div>
                ) : null}
                {doc.error && (
                  <div className="mt-1 pl-[22px] text-[10px] text-[var(--danger)]">
                    {doc.error}
                  </div>
                )}
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}
