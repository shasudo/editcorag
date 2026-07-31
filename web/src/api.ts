export type Citation = {
  ref: string;
  chunk_id: string;
  doc_id: number;
  page_no: number;
  page_span: number[];
  page_label: string;
  kind: string;
  section: string | null;
  text: string;
  bbox: number[] | null;
  asset_path: string | null;
  score: number;
};

export type DocumentInfo = {
  id: number;
  filename: string;
  pages: number;
  status: string;
  chunks: number;
  title: string | null;
  routes: Record<string, number>;
  page_status: Record<string, number>;
  error: string | null;
};

export type JobProgress = {
  id: string;
  status: string;
  pages_total: number;
  pages_done: number;
  message: string | null;
  error: string | null;
  page_status?: Record<string, number>;
};

export type FinalAnswer = {
  answer: string;
  citations: Citation[];
  abstained: boolean;
  grounded: boolean | null;
  top_score: number;
  rewrites: number;
  notes: string[];
  thread_id: string;
};

export type LlmStatus = {
  status: "ready" | "no_key" | "bad_key" | "offline" | "model_missing";
  detail?: string;
  model?: string;
};

export type Health = {
  ok: boolean;
  ready: boolean;
  documents: number;
  chunks: number;
  index_error: string | null;
  llm: LlmStatus;
  models: { ready: boolean; models: { role: string; repo: string; cached: boolean }[] };
  llm_model: string;
  embed_model: string;
  answer_mode: string;
  graders_enabled: boolean;
};

const json = async <T,>(res: Response): Promise<T> => {
  if (!res.ok) throw new Error(`${res.status} ${await res.text()}`);
  return res.json() as Promise<T>;
};

export const api = {
  health: () => fetch("/api/health").then(json<Health>),
  saveApiKey: (key: string) =>
    fetch("/api/setup/llm-key", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ key }),
    }).then(json<LlmStatus>),
  documents: () => fetch("/api/documents").then(json<DocumentInfo[]>),
  remove: (id: number) =>
    fetch(`/api/documents/${id}`, { method: "DELETE" }).then(json<unknown>),
  upload: (file: File) => {
    const body = new FormData();
    body.append("file", file);
    return fetch("/api/documents", { method: "POST", body }).then(
      json<{ job_id: string; filename: string }>,
    );
  },
  pdfUrl: (id: number) => `/api/documents/${id}/pdf`,
  assetUrl: (path: string) => `/api/assets/${path}`,
};

/**
 * Consume an SSE stream from a POST endpoint.
 *
 * Hand-rolled because EventSource cannot POST, so the stream is read off
 * fetch's body instead. Frames may be separated by CRLF, LF or CR and may
 * carry several `data:` lines — sse-starlette emits CRLF, so splitting on
 * "\n\n" alone matches nothing and no event is ever dispatched.
 */
export async function streamSSE(
  url: string,
  body: unknown,
  handlers: {
    onEvent: (event: string, data: unknown) => void;
    signal?: AbortSignal;
    method?: "GET" | "POST";
  },
): Promise<void> {
  const method = handlers.method ?? "POST";
  const res = await fetch(url, {
    method,
    ...(method === "POST"
      ? { headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) }
      : {}),
    signal: handlers.signal,
  });
  if (!res.ok || !res.body) throw new Error(`stream failed: ${res.status}`);

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  const FRAME = /\r\n\r\n|\n\n|\r\r/;
  const LINE = /\r\n|\n|\r/;

  const dispatch = (raw: string, { requireJson = false } = {}) => {
    let event = "message";
    const data: string[] = [];
    for (const line of raw.split(LINE)) {
      if (line.startsWith(":")) continue;
      if (line.startsWith("event:")) event = line.slice(6).trim();
      else if (line.startsWith("data:")) data.push(line.slice(5).replace(/^ /, ""));
    }
    if (!data.length) return;
    const payload = data.join("\n");
    try {
      handlers.onEvent(event, JSON.parse(payload));
    } catch {
      if (requireJson) return;
      handlers.onEvent(event, payload);
    }
  };

  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });

    for (;;) {
      const match = FRAME.exec(buffer);
      if (!match) break;
      dispatch(buffer.slice(0, match.index));
      buffer = buffer.slice(match.index + match[0].length);
    }
  }
  buffer += decoder.decode();
  if (buffer.trim()) dispatch(buffer, { requireJson: true });
}

/**
 * Watch ingest progress. Returns an unsubscribe — dropping it leaves the
 * EventSource open after unmount and stacks a new one on every upload.
 *
 * Job progress uses GET, so a plain EventSource is enough here.
 */
export function watchJob(
  jobId: string,
  onProgress: (p: JobProgress) => void,
  onDone: (error?: string) => void,
): () => void {
  const source = new EventSource(`/api/jobs/${jobId}/stream`);
  let finished = false;
  // The server closing a completed stream also fires a transport `error`;
  // without this guard the caller is told the ingest failed immediately after
  // being told it succeeded.
  const finish = (error?: string) => {
    if (finished) return;
    finished = true;
    source.close();
    onDone(error);
  };

  source.addEventListener("progress", (e) =>
    onProgress(JSON.parse((e as MessageEvent).data)),
  );
  source.addEventListener("done", () => finish());
  source.addEventListener("error", () => {
    if (source.readyState === EventSource.CLOSED) {
      finish("lost connection to ingest progress");
    }
  });
  return () => {
    finished = true;
    source.close();
  };
}
