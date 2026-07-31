import { useEffect, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import type { Citation, FinalAnswer } from "../api";
import { streamSSE } from "../api";
import { AlertIcon, CheckIcon, SendIcon, SparkIcon, StopIcon } from "./icons";

type Turn = {
  role: "user" | "assistant";
  content: string;
  citations?: Citation[];
  abstained?: boolean;
  grounded?: boolean | null;
  topScore?: number;
  rewrites?: number;
  streaming?: boolean;
};

const STEPS = [
  { key: "retrieve", label: "Searching" },
  { key: "rerank", label: "Ranking" },
  { key: "grade", label: "Checking relevance" },
  { key: "generate", label: "Writing" },
  { key: "verify", label: "Verifying" },
] as const;

const STAGE_TO_STEP: Record<string, (typeof STEPS)[number]["key"]> = {
  condense: "retrieve",
  retrieve: "retrieve",
  reranked: "rerank",
  grade_documents: "grade",
  graded: "grade",
  rewrite: "retrieve",
  generate: "generate",
  verify: "verify",
  verified: "verify",
  abstain: "verify",
};

function Pipeline({ active, rewriting }: { active: string | null; rewriting: boolean }) {
  const idx = active ? STEPS.findIndex((s) => s.key === active) : -1;
  return (
    <div className="animate-rise flex flex-wrap items-center gap-x-2 gap-y-1.5 px-1 py-2">
      {STEPS.map((step, i) => {
        const done = idx > i;
        const now = idx === i;
        return (
          <div key={step.key} className="flex items-center gap-2">
            <span
              className={`inline-flex items-center gap-1.5 text-xs transition-colors ${
                now
                  ? "text-[var(--accent)]"
                  : done
                    ? "text-[var(--text-3)]"
                    : "text-[var(--text-3)] opacity-45"
              }`}
            >
              <span
                className={`inline-block h-1.5 w-1.5 rounded-full ${
                  now
                    ? "dot-pulse bg-[var(--accent)]"
                    : done
                      ? "bg-[var(--ok)]"
                      : "bg-[var(--line)]"
                }`}
              />
              {step.label}
            </span>
            {i < STEPS.length - 1 && (
              <span className="h-px w-3 bg-[var(--line)]" aria-hidden="true" />
            )}
          </div>
        );
      })}
      {rewriting && (
        <span className="chip ml-1 border-[var(--warn)]/30 text-[var(--warn)]">
          no strong match — rephrasing
        </span>
      )}
    </div>
  );
}

export function Chat({
  docIds,
  onSelectCitation,
  selectedCitation,
}: {
  docIds: number[];
  onSelectCitation: (c: Citation) => void;
  selectedCitation: Citation | null;
}) {
  const [turns, setTurns] = useState<Turn[]>([]);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const [step, setStep] = useState<string | null>(null);
  const [rewriting, setRewriting] = useState(false);
  const [threadId, setThreadId] = useState<string | null>(null);
  const abortRef = useRef<AbortController | null>(null);
  const endRef = useRef<HTMLDivElement>(null);
  const boxRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [turns, step]);

  useEffect(() => {
    const el = boxRef.current;
    if (!el) return;
    el.style.height = "0px";
    el.style.height = `${Math.min(el.scrollHeight, 168)}px`;
  }, [input]);

  const ask = async (override?: string) => {
    const question = (override ?? input).trim();
    if (!question || busy) return;
    setInput("");
    setBusy(true);
    setStep(null);
    setRewriting(false);

    const history = turns.map((t) => ({ role: t.role, content: t.content }));
    setTurns((prev) => [
      ...prev,
      { role: "user", content: question },
      { role: "assistant", content: "", streaming: true },
    ]);

    const controller = new AbortController();
    abortRef.current = controller;

    const appendToken = (text: string) =>
      setTurns((prev) => {
        const last = prev[prev.length - 1];
        if (last?.role !== "assistant") return prev;
        return [...prev.slice(0, -1), { ...last, content: last.content + text }];
      });

    try {
      await streamSSE(
        "/api/chat/stream",
        { question, thread_id: threadId, doc_ids: docIds, history },
        {
          signal: controller.signal,
          onEvent: (event, data) => {
            if (event === "start") {
              setThreadId((data as { thread_id: string }).thread_id);
            } else if (event === "status") {
              const s = (data as { stage?: string }).stage;
              if (s === "rewrite") setRewriting(true);
              if (s && STAGE_TO_STEP[s]) setStep(STAGE_TO_STEP[s]);
            } else if (event === "token") {
              appendToken((data as { text: string }).text);
            } else if (event === "final") {
              const f = data as FinalAnswer;
              setTurns((prev) => {
                const last = prev[prev.length - 1];
                if (last?.role !== "assistant") return prev;
                return [
                  ...prev.slice(0, -1),
                  {
                    ...last,
                    content: f.answer || last.content,
                    citations: f.citations,
                    abstained: f.abstained,
                    grounded: f.grounded,
                    topScore: f.top_score,
                    rewrites: f.rewrites,
                    streaming: false,
                  },
                ];
              });
            } else if (event === "error") {
              appendToken(`\n\n**Error:** ${(data as { detail: string }).detail}`);
            }
          },
        },
      );
    } catch (err) {
      if ((err as Error).name !== "AbortError") {
        appendToken(`\n\n**Error:** ${(err as Error).message}`);
      }
    } finally {
      setTurns((prev) => {
        const last = prev[prev.length - 1];
        if (last?.role !== "assistant" || !last.streaming) return prev;
        return [...prev.slice(0, -1), { ...last, streaming: false }];
      });
      setBusy(false);
      setStep(null);
      setRewriting(false);
      abortRef.current = null;
    }
  };

  const suggestions = [
    "What role did Fannie Mae and Freddie Mac play in the crisis?",
    "What did the commission conclude about credit rating agencies?",
    "What were the origins of the housing bubble?",
  ];

  return (
    <div className="flex h-full min-h-0 flex-col">
      <div className="flex-1 overflow-y-auto">
        <div className="mx-auto w-full max-w-3xl px-5 py-6">
          {turns.length === 0 && (
            <div className="animate-rise mt-10 flex flex-col items-center text-center">
              <div className="mb-4 grid h-11 w-11 place-items-center rounded-full bg-[var(--accent-soft)] text-[var(--accent)]">
                <SparkIcon className="h-5 w-5" />
              </div>
              <h2 className="text-[17px] font-semibold tracking-tight">
                Ask your documents anything
              </h2>
              <p className="mt-1.5 max-w-sm text-sm text-[var(--text-2)]">
                Every answer cites the pages it came from. If the documents don’t
                contain the answer, it says so instead of guessing.
              </p>
              <div className="mt-6 flex w-full max-w-lg flex-col gap-2">
                {suggestions.map((s) => (
                  <button
                    key={s}
                    type="button"
                    onClick={() => void ask(s)}
                    className="card w-full px-3.5 py-2.5 text-left text-sm text-[var(--text-2)] transition hover:border-[var(--accent)]/40 hover:bg-[var(--surface-2)] hover:text-[var(--text)]"
                  >
                    {s}
                  </button>
                ))}
              </div>
            </div>
          )}

          <div className="flex flex-col gap-5">
            {turns.map((turn, i) =>
              turn.role === "user" ? (
                <div key={i} className="animate-rise flex justify-end">
                  <div className="max-w-[85%] rounded-2xl rounded-br-md bg-[var(--accent)] px-4 py-2.5 text-[15px] leading-relaxed text-[var(--accent-fg)] shadow-sm">
                    <div className="whitespace-pre-wrap break-words">{turn.content}</div>
                  </div>
                </div>
              ) : (
                <div key={i} className="animate-rise">
                  <div className="card px-4 py-3.5 shadow-sm">
                    {turn.content ? (
                      <div
                        className={`answer break-words ${turn.streaming ? "caret" : ""}`}
                      >
                        <ReactMarkdown remarkPlugins={[remarkGfm]}>
                          {turn.content}
                        </ReactMarkdown>
                      </div>
                    ) : (
                      <span className="text-sm text-[var(--text-3)]">Thinking…</span>
                    )}

                    {turn.citations && (
                      <div className="mt-3.5 border-t border-[var(--line-soft)] pt-3">
                        {turn.abstained ? (
                          <div className="mb-2 flex items-center gap-1.5 text-xs text-[var(--warn)]">
                            <AlertIcon className="h-3.5 w-3.5" />
                            Not found in these documents
                            {typeof turn.topScore === "number" &&
                              ` — best passage scored ${turn.topScore.toFixed(2)}`}
                            {turn.rewrites ? `, after ${turn.rewrites} retries` : ""}
                          </div>
                        ) : (
                          turn.grounded === true && (
                            <div className="mb-2 flex items-center gap-1.5 text-xs text-[var(--ok)]">
                              <CheckIcon className="h-3.5 w-3.5" />
                              Every claim verified against the source
                            </div>
                          )
                        )}
                        {turn.grounded === false && !turn.abstained && (
                          <div className="mb-2 flex items-center gap-1.5 text-xs text-[var(--warn)]">
                            <AlertIcon className="h-3.5 w-3.5" />
                            Could not fully verify this answer
                          </div>
                        )}

                        {turn.citations.length > 0 && (
                          <>
                            <div className="mb-1.5 text-[10px] font-medium uppercase tracking-wider text-[var(--text-3)]">
                              Sources
                            </div>
                            <div className="flex flex-wrap gap-1.5">
                              {turn.citations.map((c) => {
                                const on = selectedCitation?.chunk_id === c.chunk_id;
                                return (
                                  <button
                                    key={c.chunk_id + c.ref}
                                    type="button"
                                    onClick={() => onSelectCitation(c)}
                                    title={c.text.slice(0, 300)}
                                    className={`chip cursor-pointer transition ${
                                      on
                                        ? "border-[var(--accent)] bg-[var(--accent-soft)] text-[var(--accent)]"
                                        : "hover:border-[var(--accent)]/40 hover:text-[var(--text)]"
                                    }`}
                                  >
                                    <span className="font-mono opacity-60">{c.ref}</span>
                                    {c.page_label}
                                    {c.kind !== "text" && (
                                      <span className="opacity-60">· {c.kind}</span>
                                    )}
                                  </button>
                                );
                              })}
                            </div>
                          </>
                        )}
                      </div>
                    )}
                  </div>
                </div>
              ),
            )}
          </div>

          {busy && <Pipeline active={step} rewriting={rewriting} />}
          <div ref={endRef} />
        </div>
      </div>

      <div className="border-t border-[var(--line)] bg-[var(--surface)]/80 backdrop-blur">
        <div className="mx-auto w-full max-w-3xl px-5 py-3.5">
          <div className="card flex items-end gap-2 p-2 focus-within:border-[var(--accent)]/50">
            <textarea
              ref={boxRef}
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter" && !e.shiftKey) {
                  e.preventDefault();
                  void ask();
                }
              }}
              rows={1}
              placeholder="Ask about the documents…"
              className="max-h-42 flex-1 resize-none bg-transparent px-2 py-1.5 text-[15px] leading-relaxed outline-none placeholder:text-[var(--text-3)]"
            />
            {busy ? (
              <button
                type="button"
                onClick={() => abortRef.current?.abort()}
                className="btn btn-ghost"
                aria-label="Stop generating"
              >
                <StopIcon className="h-3.5 w-3.5" />
                Stop
              </button>
            ) : (
              <button
                type="button"
                onClick={() => void ask()}
                disabled={!input.trim()}
                className="btn btn-primary"
                aria-label="Send question"
              >
                Ask
                <SendIcon className="h-3.5 w-3.5" />
              </button>
            )}
          </div>
          <p className="mt-1.5 px-1 text-[11px] text-[var(--text-3)]">
            <kbd className="rounded border border-[var(--line)] px-1">Enter</kbd> to
            send · <kbd className="rounded border border-[var(--line)] px-1">Shift</kbd>
            +<kbd className="rounded border border-[var(--line)] px-1">Enter</kbd> for a
            new line
          </p>
        </div>
      </div>
    </div>
  );
}
