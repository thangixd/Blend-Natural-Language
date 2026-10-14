# NLSeeker evaluation suite

## Commands

```bash
make up                                       # build + start the container once
make build-lake LAKE=<lake> BLEND_GPUS=0,1 PNEUMA_GPUS=2,3   # server, offline phase: build and seal one lake's indexes
make evaluation-nlseeker BLEND_GPUS=0,1 [CONFIG=path.yaml]   # Blend alone; default CONFIG=evaluation/nlseeker/configs/default.yaml
make evaluation-nlseeker BLEND_GPUS=0,1 RESUME=<run-dir>     # resume a cancelled/crashed run (dir name under results/, .partial included)
make parity-backend BACKEND=pneuma BLEND_GPUS=0,1 PNEUMA_GPUS=2,3 CONFIG=path.yaml   # one backend, the other beside it
make parity-pilot-check BLEND_RUN=<run-dir> PNEUMA_RUN=<run-dir>   # the pilot's criteria on two runs
make parity LAKE=<lake> BLEND_GPUS=0,1 PNEUMA_GPUS=2,3   # server: both backends on one lake at once (configs/parity.yaml)
make parity-report RUNS="<blend run> <pneuma run> ..."   # local, pulled runs: one pair per lake -> data/Evaluation/NLSeeker/parity/
make eval-clean RUN=<run-dir-name>            # rm results/<RUN>, routed through the container
make eval-clean-cache                         # rm the extracted lake/bench cache (PneumaData/ is never touched)
```

`RESUME` skips datasets already marked `complete` in the run's `run_meta.json`,
wipes and reruns the one that was interrupted, and takes its configuration from
the `config.yaml` copied into the run dir, so `CONFIG` is ignored on resume.

## Offline phase

Every run loads lakes the offline phase built and builds nothing. `make build-lake LAKE=<lake>`
builds one lake once on the server, Blend then Pneuma (`python -m evaluation.nlseeker.offline build`), and seals it:

```
config/<lake>.ini                    # names the lake's database: switch lakes with plan.DB.load_config(<ini>)
data/indexes/<lake>.db               # Blend: inverted index, NL index and its table copies, in one file
data/indexes/<lake>.MANIFEST.json    # every file below, size and sha256; its first 12 hex are the lake's code
data/indexes/<lake>/
  build_blend.json, build_pneuma.json  # build records: `entry` (source sha256s, stages, storage, shared texts),
                                       # endpoints, layout, condition, NLSeeker keys
  texts_blend.json, texts_pneuma.json  # each build's narration and row blocks and contexts, before indexing
  blend.ini, pneuma.ini, metadata.csv  # the build's settings and Blend's contexts input
  pneuma/                              # Pneuma's own build: storage.db and indexes (Chroma with HNSW, bm25s)
  pneuma_shared/                       # Pneuma's index of Blend's texts (configs with attribution arms)
```

Lakes are Pneuma's five (`adventure_works`, `chembl`, `chicago_open`, `public_bi`, `fetaqa`) and the throughput
subsets `fetaqa_<tables>` (built with `configs/throughput.yaml`; both backends index the texts of Blend's fetaqa
build). A lake is built only once: the build refuses an existing database or a sealed lake, so a rebuild (which
narrates again) is a deliberate delete and a new code. A run checks the manifest and that the build used today's
source archives, then records the build's `entry` and the code in its own `run_meta.json`.

## Configs

| config | grid | maps to |
| --- | --- | --- |
| `configs/default.yaml` | k [1, 5], n [5], alpha [0.5], all five datasets | Pneuma paper Figs 6-8 (main results) |
| `configs/alpha_sweep.yaml` | k [1], n [5], alpha [0.0 .. 1.0] | Pneuma paper Fig 14 |
| `configs/n_sweep.yaml` | k [1], n [1, 5, 10, 15, 20], alpha [0.5] | Pneuma paper Fig 16 |
| `configs/smoke.yaml` | adventure_works only, BC1+BX1, questions_limit 2 | quick end-to-end check, not a paper figure |
| `configs/pilot.yaml` | adventure_works, BC2, questions_limit 5, k [1, 5], alpha [0.5], judge on/off; attribution k [1, 5], alpha [0, 1] | the parity port pilot |
| `configs/parity.yaml` | the pilot's grid on all five datasets and four families, questions_limit 1000; run per lake (`--lake`) | the parity run |

Every config names each backend's own vLLM pair under `backends:` (their URLs). GPUs are launch arguments:
`--layout BACKEND:CPUS:LLM_GPU,EMBEDDING_GPU` places this process's backend (its container's cpuset, and the GPUs
its endpoints run on, picked from those free at launch). A second `--layout` for the other backend means the two
run side by side; the runner then refuses to start if they share a core, a GPU or an endpoint URL.
One process runs one backend (`--backend`, default `blend`) into its own run directory `<ts>__<tag>-<backend>`.
`pneuma` is vanilla Pneuma (`third_party/pneuma`, see its `VENDORED.md`) in its own environment
(`evaluation/nlseeker/pneuma-env`, the `pneuma` image and service).

The optional `attribution:` section (`k`, `alpha`) adds the attribution arms: judge off, run on one build's texts
indexed in both backends, so BM25 alone must rank identically and alpha 0 compares the exact scan with HNSW only.
Blend's build is their source: Blend runs them on its own index; Pneuma's offline build indexes Blend's texts a
second time with its own registrar, index generator and retrieval (`pneuma_shared/`). The grid's other arms run on
each backend's own build. Records carry `texts: own|shared`. Both processes start together.

`--lake <lake or dataset>` runs one of the config's datasets; the run stores the narrowed config, so a resume keeps it.
Every fresh run records the sha256 of `parity_tolerance.yaml` when it starts. `report parity` (`make parity-report`)
refuses runs that started under another tolerance, pairs Blend's and Pneuma's records question by question, names
each differing top-k list's cause (judge, bm25, vector, own texts) from the attribution arms, and judges every group
by the tolerance: `parity.json` and `parity_questions.jsonl`, and prints the table.

## Measurement contract

`runner.run_backend` drives any backend the same way: every question, question-outer over the combos.
There is no warm-up call: the user warms the servers before a run; each backend loads its indexes, tokenizers
and the chat template size before the first timed question. The runner times `total`; `embedding` and `judge` come from the
instrumented clients it wraps around the backend's shared clients; `retrieval = total - embedding - judge`.
A backend returns its ranked pool of document ids; the runner maps them to Pneuma table ids, dedups
per table, cuts to k and scores. `report.pair` joins two backends' records question by question, flags
pool-size differences, and refuses a group the two ran on different questions.

## Results layout

```
data/Evaluation/NLSeeker/results/<ts>__<tag>/
  config.yaml              # byte copy of the input YAML
  run_meta.json            # backend, endpoints (URL, GPU, served models), condition (parallel, CPUs), layout,
                           # per-dataset provenance (row sampling; Pneuma: upstream, patches, HNSW), input sha256s, package versions, per-dataset status,
                           # the lake's build entry (stages, storage) and code
  summary.csv              # one row per dataset x family x k x n x alpha
  <dataset>/
    artifacts/
      nlseeker.ini          # the lake's ini with this run's endpoints and NLSeeker keys
    per_query.jsonl        # qid, family, texts, k, n, alpha, rerank, ranked pool, retrieved/answer ids, hit, rr,
                           # total, embedding, retrieval, judge (s), calls, condition
    documents.json         # per lake file, the document count by type (schema, row, contexts)
    documents_shared.json  # Pneuma with attribution arms: the same for the index of the shared texts
    texts.json             # the build's texts_<backend>.json, copied from the lake
    nl_runs.jsonl          # Blend only: one NL run record per call, with the component and Blend-HNSW probes
```

A run directory is created with a `.partial` suffix and renamed to drop it only
on completion, so a plain listing of `results/` shows which runs actually
finished. `PneumaData/` is the source archive set and is
immutable - nothing in this suite writes to it or deletes from it.
