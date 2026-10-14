# Vendored Pneuma

Upstream: https://github.com/TheDataStation/pneuma at `8fd9f68a29d6e4ef8b3a5556dbbc2c2e316d2e3a` (MIT, see
`LICENSE`), exported whole except `experiments/out_benchmark/` (the paper's result files, 37 MB).

Every deviation from upstream:

- `patches/0001-registrar-metadata-insert.patch`: a failed context insert returns its error instead of crashing
  on `None`, and apostrophes in contexts are escaped. Without it the build loses or crashes on contexts.
- `patches/0002-judge-prompt-type.patch`: the judge picks the content prompt only for content documents. Upstream
  tests `split("_SEP_")[1]`, which always gives the context prompt because the lake's file names contain `_SEP_`.
- The endpoint patches, at runtime in `evaluation/nlseeker/pneuma_backend.py`: narration and judge go through
  Blend's shared chat client, embeddings through the shared embedding client, `chromadb` resolves to
  `chromadb_deterministic`. The query is tokenized with `stopwords='en'` as the corpus is (upstream's experiment
  omits it on the query; the scores are the same, since stopwords are not in the corpus vocabulary).
- The Chroma batch patch, at runtime in the same file: `generate_index` adds up to 30,000 documents per call, and
  `chromadb_deterministic` 0.5.5 refuses more than its SQLite batch limit (5,461), so every lake above that failed
  to index. The add is split into batches of that limit, in the same order.
- The registrar connection patch, at runtime in the same file: upstream's folder registration opens a new DuckDB
  connection per table, and each open loads and checkpoints the growing catalog, so the cost per table grows with
  the tables already registered (about 1.75 s per table at fetaqa's 10,330). One connection now serves the whole
  folder; each table runs upstream's own statements in upstream's order, so `storage.db` holds the same rows and
  tables. It changes only connection reuse: Pneuma's offline
  registration time, and where DuckDB checkpoints (so `storage.db`'s file size), are therefore not upstream's, and
  every run's provenance says so.
- The metadata connection patch, the same patch on the metadata file read: upstream's `__insert_metadata` opens a
  new connection per context row, with the same catalog load and checkpoint (with 10,330 tables registered, 0.85 s
  per row locally and 1.6 s on the server, about 45 h for fetaqa's 100,718 contexts; 2.6 ms per row patched). One
  connection now serves the whole file, each row running upstream's own insert in upstream's order, so
  `table_contexts` holds the same rows and ids. Again only connection reuse:
  Pneuma's metadata time and checkpoints are not upstream's, and the provenance says so. A metadata folder still
  opens one connection per file; the backend never passes one, it writes all contexts to one CSV. The summarizer and
  the index generator open per-table connections too, but always while their own outer connection holds the database
  open, so DuckDB reuses that instance and loads or checkpoints no catalog per table; they stay unpatched.

Upstream behaviour the backend maps, not a patch: the registrar stores a table id with every double quote turned
into a single quote and every backslash into a slash (`registrar.py`, `__read_table_file`), so fetaqa's
`Andrew "Jr. Boy" Jones , Discography_SEP_table_7648.csv` is registered as `Andrew 'Jr. Boy' Jones , ...`
(12 fetaqa files; no other lake has a double quote or a backslash). `pneuma_backend.py` expects those ids at
registration, keys the contexts and the shared texts on them, and maps document ids back to lake files through
the lake's own file list, refusing a lake where two files would get one id.

Both source patches are applied in this tree; `patch -R -p1 < patches/<name>` restores upstream.
