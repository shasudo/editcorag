CREATE TABLE IF NOT EXISTS documents (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    sha256           TEXT    NOT NULL,
    filename         TEXT    NOT NULL,
    path             TEXT    NOT NULL,
    page_count       INTEGER NOT NULL,
    byte_size        INTEGER NOT NULL,
    title            TEXT,
    pipeline_version TEXT    NOT NULL,
    embed_model      TEXT,
    embed_dim        INTEGER,
    status           TEXT    NOT NULL DEFAULT 'pending',
    error            TEXT,
    created_at       TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
    updated_at       TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
    UNIQUE (sha256, pipeline_version)
);

CREATE TABLE IF NOT EXISTS pages (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    doc_id       INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    page_no      INTEGER NOT NULL,
    route        TEXT,
    status       TEXT    NOT NULL DEFAULT 'pending',
    attempts     INTEGER NOT NULL DEFAULT 0,
    width        REAL,
    height       REAL,
    rotation     INTEGER NOT NULL DEFAULT 0,
    n_chars      INTEGER,
    text_cover   REAL,
    image_cover  REAL,
    quality      REAL,
    n_chunks     INTEGER NOT NULL DEFAULT 0,
    error        TEXT,
    updated_at   TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
    UNIQUE (doc_id, page_no)
);

CREATE INDEX IF NOT EXISTS idx_pages_doc_status ON pages (doc_id, status);
CREATE INDEX IF NOT EXISTS idx_pages_doc_route  ON pages (doc_id, route);

CREATE TABLE IF NOT EXISTS assets (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    doc_id     INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    page_no    INTEGER NOT NULL,
    kind       TEXT    NOT NULL,
    path       TEXT    NOT NULL,
    bbox       TEXT,
    ocr_text   TEXT,
    caption    TEXT,
    created_at TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
);

CREATE INDEX IF NOT EXISTS idx_assets_doc_page ON assets (doc_id, page_no);

CREATE TABLE IF NOT EXISTS chunks (
    id         TEXT    PRIMARY KEY,
    doc_id     INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    page_no    INTEGER NOT NULL,
    page_span  TEXT,
    kind       TEXT    NOT NULL,
    section    TEXT,
    bbox       TEXT,
    asset_id   INTEGER REFERENCES assets(id) ON DELETE SET NULL,
    created_at TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
);

CREATE INDEX IF NOT EXISTS idx_chunks_doc_page ON chunks (doc_id, page_no);

CREATE TABLE IF NOT EXISTS jobs (
    id           TEXT    PRIMARY KEY,
    doc_id       INTEGER REFERENCES documents(id) ON DELETE CASCADE,
    kind         TEXT    NOT NULL,
    status       TEXT    NOT NULL DEFAULT 'queued',
    pages_total  INTEGER NOT NULL DEFAULT 0,
    pages_done   INTEGER NOT NULL DEFAULT 0,
    message      TEXT,
    error        TEXT,
    started_at   TEXT,
    finished_at  TEXT,
    updated_at   TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
);

CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs (status);
