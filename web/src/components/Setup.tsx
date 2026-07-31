import { useEffect, useRef, useState } from "react";
import type { Health } from "../api";
import { api, streamSSE } from "../api";
import { AlertIcon, CheckIcon, CloseIcon } from "./icons";

/**
 * First-run and settings, both in the main window.
 *
 * Deliberately not a separate setup screen: the app opens straight onto the
 * home screen every launch, and whatever is still missing is asked for in
 * place. A key entered once is stored server-side, so it survives a restart
 * and works the same whether the app was opened from the desktop shell or
 * with `editcorag serve`.
 */

function Modal({
  title,
  children,
  onClose,
}: {
  title: string;
  children: React.ReactNode;
  onClose?: () => void;
}) {
  useEffect(() => {
    if (!onClose) return;
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  return (
    <div className="fixed inset-0 z-50 grid place-items-center bg-black/55 p-6 backdrop-blur-sm">
      <div
        role="dialog"
        aria-modal="true"
        aria-label={title}
        className="card w-full max-w-[420px] p-5 shadow-[var(--shadow-lg)]"
      >
        <div className="mb-3 flex items-start justify-between gap-3">
          <h2 className="text-[15px] font-semibold tracking-tight">{title}</h2>
          {onClose && (
            <button
              type="button"
              onClick={onClose}
              aria-label="Close"
              className="rounded p-1 text-[var(--text-3)] transition hover:bg-[var(--surface-2)] hover:text-[var(--text)]"
            >
              <CloseIcon className="h-3.5 w-3.5" />
            </button>
          )}
        </div>
        {children}
      </div>
    </div>
  );
}

/** Ask for a Groq key, verify it server-side, and report the outcome. */
export function ApiKeyForm({
  status,
  onSaved,
}: {
  status?: string;
  onSaved: () => void;
}) {
  const [key, setKey] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    inputRef.current?.focus();
  }, []);

  const submit = async () => {
    const value = key.trim();
    if (!value || busy) return;
    setBusy(true);
    setError(null);
    try {
      const result = await api.saveApiKey(value);
      if (result.status === "ready") {
        setKey("");
        onSaved();
      } else {
        setError(
          result.status === "bad_key"
            ? "Groq rejected that key. Check it was copied in full."
            : result.detail || "That key could not be verified.",
        );
      }
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  };

  return (
    <>
      <p className="text-[13px] leading-relaxed text-[var(--text-2)]">
        {status === "bad_key"
          ? "The saved key was rejected. Paste a new one to continue."
          : "Answers are written by Groq. Paste a Groq API key to get started — it is free to create and is stored only on this computer."}
      </p>
      <input
        ref={inputRef}
        type="password"
        value={key}
        spellCheck={false}
        autoComplete="off"
        placeholder="gsk_…"
        aria-label="Groq API key"
        onChange={(e) => setKey(e.target.value)}
        onKeyDown={(e) => e.key === "Enter" && void submit()}
        className="mt-3 w-full rounded-lg border border-[var(--line)] bg-[var(--surface-2)] px-3 py-2 font-mono text-[13px] outline-none focus:border-[var(--accent)]"
      />
      {error && (
        <div className="mt-2 flex items-start gap-1.5 text-xs text-[var(--danger)]">
          <AlertIcon className="mt-0.5 h-3.5 w-3.5 shrink-0" />
          {error}
        </div>
      )}
      <div className="mt-4 flex items-center justify-between gap-3">
        <a
          href="https://console.groq.com/keys"
          target="_blank"
          rel="noreferrer"
          className="text-xs text-[var(--accent)] hover:underline"
        >
          Get a free key
        </a>
        <button
          type="button"
          onClick={() => void submit()}
          disabled={!key.trim() || busy}
          className="btn btn-primary"
        >
          {busy ? "Checking…" : "Save"}
        </button>
      </div>
    </>
  );
}

/** Settings, reachable any time — currently the API key and what is loaded. */
export function SettingsPanel({
  health,
  onClose,
  onChanged,
}: {
  health: Health | null;
  onClose: () => void;
  onChanged: () => void;
}) {
  const [editing, setEditing] = useState(false);
  const llm = health?.llm;

  return (
    <Modal title="Settings" onClose={onClose}>
      <div className="flex flex-col gap-3 text-[13px]">
        <div className="rounded-lg border border-[var(--line)] p-3">
          <div className="mb-1.5 flex items-center justify-between">
            <span className="font-medium">Groq API key</span>
            {llm?.status === "ready" ? (
              <span className="flex items-center gap-1 text-xs text-[var(--ok)]">
                <CheckIcon className="h-3.5 w-3.5" /> working
              </span>
            ) : (
              <span className="flex items-center gap-1 text-xs text-[var(--warn)]">
                <AlertIcon className="h-3.5 w-3.5" />
                {llm?.status === "no_key" ? "not set" : (llm?.status ?? "unknown")}
              </span>
            )}
          </div>
          {editing ? (
            <ApiKeyForm
              status={llm?.status}
              onSaved={() => {
                setEditing(false);
                onChanged();
              }}
            />
          ) : (
            <div className="flex items-center justify-between gap-3">
              <span className="text-xs text-[var(--text-3)]">
                Stored on this computer. Replacing it takes effect immediately.
              </span>
              <button type="button" className="btn btn-ghost" onClick={() => setEditing(true)}>
                Change
              </button>
            </div>
          )}
        </div>

        <dl className="rounded-lg border border-[var(--line)] p-3 text-xs text-[var(--text-2)]">
          {[
            ["Answering model", health?.llm_model],
            ["Embedding model", health?.embed_model],
            ["Mode", health?.answer_mode],
            ["Indexed chunks", health ? String(health.chunks) : undefined],
          ].map(([label, value]) => (
            <div key={label} className="flex justify-between gap-3 py-0.5">
              <dt>{label}</dt>
              <dd className="truncate font-mono text-[11px] text-[var(--text-3)]">
                {value ?? "—"}
              </dd>
            </div>
          ))}
        </dl>
      </div>
    </Modal>
  );
}

/**
 * First run downloads ~7 GB of models. Shown as a banner rather than a blocking
 * screen so the rest of the app stays visible and the wait is explained.
 */
export function ModelDownload({ onDone }: { onDone: () => void }) {
  const [percent, setPercent] = useState(0);
  const [detail, setDetail] = useState("Preparing…");
  const [error, setError] = useState<string | null>(null);
  const started = useRef(false);

  useEffect(() => {
    if (started.current) return;
    started.current = true;
    void streamSSE("/api/setup/stream", undefined, {
      method: "GET",
      onEvent: (event, data) => {
        const e = data as {
          percent?: number;
          done_bytes?: number;
          total_bytes?: number;
          detail?: string;
        };
        if (event === "error") setError(e.detail ?? "Download failed.");
        else if (event === "download") {
          setPercent(e.percent ?? 0);
          if (e.total_bytes) {
            setDetail(
              `${((e.done_bytes ?? 0) / 1e9).toFixed(2)} of ${(e.total_bytes / 1e9).toFixed(2)} GB`,
            );
          }
        } else if (event === "complete") onDone();
      },
    }).catch((e) => setError((e as Error).message));
  }, [onDone]);

  return (
    <div className="border-b border-[var(--line)] bg-[var(--surface-2)] px-4 py-2.5">
      <div className="mx-auto flex w-full max-w-3xl items-center gap-3">
        <div className="min-w-0 flex-1">
          <div className="flex items-baseline justify-between gap-3 text-xs">
            <span className="font-medium">
              {error ? "Setup could not finish" : "Setting up — downloading the search models"}
            </span>
            <span className="tabnum text-[var(--text-3)]">
              {error ? "" : `${Math.round(percent)}%`}
            </span>
          </div>
          <div className="mt-1.5 h-1 overflow-hidden rounded-full bg-[var(--line)]">
            <div
              className="h-full rounded-full bg-[var(--accent)] transition-all duration-500"
              style={{ width: `${percent}%` }}
            />
          </div>
          <div className="mt-1 text-[11px] text-[var(--text-3)]">
            {error ?? `${detail} · this happens once, you can keep using the app`}
          </div>
        </div>
      </div>
    </div>
  );
}

export { Modal };
