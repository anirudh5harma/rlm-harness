CREATE TABLE IF NOT EXISTS schema_migrations (
  version INTEGER PRIMARY KEY,
  applied_at INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS core (
  key TEXT PRIMARY KEY,
  value TEXT NOT NULL,
  updated_at INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS recall (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  thread_id TEXT NOT NULL,
  ts INTEGER NOT NULL,
  role TEXT NOT NULL,
  content TEXT NOT NULL,
  tokens INTEGER NOT NULL,
  metadata_json TEXT NOT NULL DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS recall_thread_ts ON recall(thread_id, ts DESC, id DESC);
CREATE INDEX IF NOT EXISTS recall_role ON recall(role);

CREATE TABLE IF NOT EXISTS archival_meta (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  kind TEXT NOT NULL,
  source_thread TEXT,
  ts INTEGER NOT NULL,
  content TEXT NOT NULL,
  tokens INTEGER NOT NULL,
  metadata_json TEXT NOT NULL DEFAULT '{}',
  embedding_model TEXT NOT NULL,
  embedding_dim INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS archival_kind ON archival_meta(kind);
CREATE INDEX IF NOT EXISTS archival_source_thread ON archival_meta(source_thread);
CREATE INDEX IF NOT EXISTS archival_ts ON archival_meta(ts DESC, id DESC);

CREATE VIRTUAL TABLE IF NOT EXISTS archival_fts USING fts5(
  content,
  kind,
  source_thread,
  content='archival_meta',
  content_rowid='id'
);

CREATE TRIGGER IF NOT EXISTS archival_meta_ai
AFTER INSERT ON archival_meta
BEGIN
  INSERT INTO archival_fts(rowid, content, kind, source_thread)
  VALUES (new.id, new.content, new.kind, COALESCE(new.source_thread, ''));
END;

CREATE TRIGGER IF NOT EXISTS archival_meta_ad
AFTER DELETE ON archival_meta
BEGIN
  INSERT INTO archival_fts(archival_fts, rowid, content, kind, source_thread)
  VALUES ('delete', old.id, old.content, old.kind, COALESCE(old.source_thread, ''));
END;

CREATE TRIGGER IF NOT EXISTS archival_meta_au
AFTER UPDATE ON archival_meta
BEGIN
  INSERT INTO archival_fts(archival_fts, rowid, content, kind, source_thread)
  VALUES ('delete', old.id, old.content, old.kind, COALESCE(old.source_thread, ''));
  INSERT INTO archival_fts(rowid, content, kind, source_thread)
  VALUES (new.id, new.content, new.kind, COALESCE(new.source_thread, ''));
END;
