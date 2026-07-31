# Putting EditcoRAG on a USB stick

## Build

```bash
cd web && pnpm install && pnpm build && cd ..
cd desktop && npm install && npm run dist:mac
```

Output: `desktop/release/EditcoRAG-0.1.0-arm64.dmg` (~540 MB).

The build refuses to finish if anything is wrong: it checks that no symlink
escapes the bundle, that the interpreter resolves its own prefix, that torch,
docling, lancedb, rapidocr and server all import from inside the app, and that
the ad-hoc signature verifies.

## Copy the DMG to the stick — not the .app

**Copy `EditcoRAG-0.1.0-arm64.dmg`. Never drag `EditcoRAG.app` onto the stick.**

A USB stick that works on both Mac and Windows is formatted exFAT, which cannot
store symlinks, executable bits, or extended attributes. The app contains 23
symlinks and a code signature that depend on all three. Copied directly, it
arrives broken and macOS reports *"EditcoRAG is damaged and can't be opened"* —
with no way for the recipient to fix it.

A DMG is a single file. It copies to exFAT intact, and mounting it on the target
Mac restores a real filesystem with everything preserved.

## What they do

1. Plug in the stick, copy the `.dmg` to their Desktop
2. Double-click it, drag **EditcoRAG** into **Applications**
3. Eject the disk image, open EditcoRAG from Applications

There is no Gatekeeper prompt. Gatekeeper only assesses apps carrying a
`com.apple.quarantine` flag, and that flag is set by whatever *downloads* a
file. Nothing copied from a USB stick has it.

If you ever email or upload the DMG instead, it *will* be quarantined, and they
will need: System Settings → Privacy & Security → scroll down → **Open Anyway**,
once.

## First run

The app opens a small setup window and, in order:

1. starts its engine (a few seconds)
2. downloads the search models — **about 7 GB, over the internet, once**
3. asks for a Groq API key (free, from https://console.groq.com/keys)

Then the main window opens. Steps 2 and 3 never happen again.

**Step 2 needs a working internet connection on their machine.** If your
recipients are offline or on slow connections, see "Shipping the models too".

## Requirements

- Apple Silicon Mac (M1 or later). The bundled interpreter is arm64; an Intel
  build needs an x86_64 Python bundled on an Intel machine or via
  `uv python install cpython-3.12-macos-x86_64`.
- ~8 GB free disk (1.5 GB app + ~7 GB models)
- ~5 GB RAM free while answering; the embedder and reranker stay resident

## Shipping the models too (optional)

To make first run fully offline, copy the model cache onto the stick as well and
have the app read it from there. This turns the ~540 MB DMG into roughly 7.5 GB
but removes the download entirely. Ask if you want this wired up — it is a
`HF_HOME` override in `desktop/src/backend.js` plus a copy step in
`scripts/bundle-python.mjs`.

## Windows

`npm run dist:win` must be run **on Windows** — the bundled Python is
platform-specific, so a Windows installer cannot be produced from macOS. A free
GitHub Actions `windows-latest` runner does this if you do not have a Windows
machine.

Unsigned, their first launch shows SmartScreen: **More info → Run anyway**, once.
