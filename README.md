# Blend: A Unified Data Discovery System
Here you can find the code for the "Blend: A Unified Data Discovery System" paper, extended with a
natural-language seeker (a reimplementation of the retrieval half of Pneuma), a trainable ML cost
optimizer, the Blend agent (natural-language requests to Blend plans) and the evaluation suites for both. DuckDB is the only supported backend; the Vertica and Postgres paths from the paper are
kept for reference only.

## Abstract
Data discovery is an iterative and incremental process that necessitates the execution of multiple data discovery queries to identify the desired tables from large and diverse data lakes. Current methodologies concentrate on single discovery tasks such as join, correlation, or union discovery. However, in practice, a series of these approaches and their corresponding index structures are necessary to enable the user to discover the desired tables.

This paper presents Blend, a comprehensive data discovery system that empowers users to develop ad-hoc discovery tasks without the need to develop new algorithms or build a new index structure. To achieve this goal, we introduce a general index structure capable of addressing multiple discovery queries. We develop a set of lower-level operators that serve as the fundamental building blocks for more complex and sophisticated user tasks. These operators are highly efficient and enable end-to-end efficiency. 

To enhance the execution of the discovery pipeline, we rewrite the search queries into optimized SQL statements to push the data operators down to the database. We demonstrate that our holistic system is able to achieve comparable effectiveness and runtime efficiency to the individual state-of-the-art approaches specifically designed for a single task.

## Installation
Dependencies are managed with [uv](https://docs.astral.sh/uv/). `pyproject.toml` lists them, `uv.lock` pins the exact resolved versions and `.python-version` pins the interpreter (3.12).

```bash
uv sync                 # core: token seekers only
uv sync --extra nl      # + openai, PyStemmer, tokenizers, huggingface-hub - needed for the NaturalLanguage seeker
uv sync --extra index   # + fastparquet, needed only by the GitTables loader
uv sync --extra agent   # + pydantic-ai-slim, needed for the Blend agent
uv sync --extra ui      # + streamlit, for the agent's web app
uv sync --all-extras    # nl, index, test, eval (pyyaml, matplotlib), agent and ui
```

## Inference endpoints (NaturalLanguage seeker only)
The NL seeker never loads model weights locally. It talks to two OpenAI-compatible HTTP endpoints
(e.g. two vLLM containers): a chat LLM (used for table summaries and the rerank judge) and an
embedder. Both URLs go into the `[NLSeeker]` config section below. The token seekers (keyword,
join, correlation, multi-column) never touch them.

## Database configuration
Blend reads `config/config.ini` at import time, so that file must exist and connect. For DuckDB
plus the NL seeker, one ini carries both sections (template: `config/nlseeker_example.ini`):

```ini
[Database]
dbms=duckdb
path=blend_duckdb.db
index_table=my_lake

[NLSeeker]
index_name=my_lake
schema=nl_my_lake

llm_base_url=http://127.0.0.1:8001/v1
llm_api_key=vllm-local
llm_model=Qwen2.5-7B-Instruct
llm_temperature=0.0
llm_max_new_tokens=512
llm_context_length=32768
llm_concurrency=1
llm_retry_attempts=5
llm_tokenizer=Qwen/Qwen2.5-7B-Instruct

embedding_base_url=http://127.0.0.1:8002/v1
embedding_api_key=vllm-local
embedding_model=bge-base-en-v1.5
embedding_max_tokens=512
embedding_batch_size=256
embedding_tokenizer=BAAI/bge-base-en-v1.5

alpha=0.5
n=5
```

Every `[NLSeeker]` key is required. A plan can be retargeted at another lake at runtime with
`plan.DB.load_config(Path("path/to/other.ini"))` - this switches the inverted index and the NL
index together.

## Index generation
Build the inverted index and the natural-language index over one lake of CSV files, sharing
TableIds. Both indexes live as tables in the same `.db`
file: the NL side adds `<index_name>_nl_documents` and `<index_name>_nl_tokens` beside the
inverted index table, plus a bookkeeping schema (`schema=` above) for summaries and status.

```bash
uv run python -m scripts.create_index_nl_blend_duckdb \
  --lake my_lake --datalake path/to/datalake --db blend_duckdb.db --config config/config.ini
```

`--lake` names both the subdirectory under `--datalake` and the index table. Useful flags:
`--no-nl-index` / `--no-blend-index` build one side only, `--metadata` registers extra table
context for retrieval. To build only the inverted index (no NL, no endpoints needed):

```bash
uv run python -m scripts.create_index_duckdb \
  --lake my_lake --datalake path/to/datalake --db blend_duckdb.db
```

## Training the cost optimizer
The plan optimizer orders seekers by cost first and breaks ties with a learned
runtime model per token seeker type (XGBoost, `src/Operators/Seekers/<Class>_model.json`) and, for NL, with
`t_retrieval + judge * k * n * t_doc` from `config/costs/<lake>.json`. The `[Optimizer]`
section of a lake config sets the costs: `cost_basis=priority` (the default) uses fixed ranks (Keyword 3,
SC 4, Correlation 6, MC 10, NL 11), `cost_basis=measured` uses the lake's median seeker runtimes in
`config/costs/<lake>.json`, with NL counted once with and once without the judge
(`uv run python -m scripts.derive_costs` rebuilds them from `results/NLIntegration/<lake>/`).
The learned tie-breaker is on (`DBHandler.USE_ML_OPTIMIZER = True`); the models are loaded
at seeker construction, and a trained set ships with the repo. Training is offline - at query time the
models only predict. They only order Intersection inputs of equal cost, but the input that runs last ranks
the result, so a stale model can change which tables come back, not only the runtime. Retrain them with:

```bash
uv run python -m scripts.train_ml_optimizer --config config/config.ini
```

Token frequencies used by the features are queried from the index on demand - no extra artifact.

## Writing plans
Operators compose into a DAG that compiles down to SQL against the index. Query values must be
normalized like the index (lowercased, trimmed) or they will not match.

```python
from pathlib import Path
import pandas as pd

from src.Plan import Plan
from src.Operators import Seekers, Combiners

df = pd.read_csv("my_table.csv")
cities = df["City"].astype(str).str.lower().tolist()

plan = Plan()
plan.add("city_join", Seekers.SC(cities, k=50))
plan.add("nl", Seekers.NL("which tables contain customer mailing addresses", k=50))
plan.add("intersection", Combiners.Intersection(k=10), inputs=["city_join", "nl"])

plan.DB.load_config(Path("config/config.ini"))
table_ids = plan.run()
```

Seekers: `SC` (single-column join), `MC` (multi-column join), `C` (correlation), `Keyword`,
`NL` (natural language). Combiners: `Intersection`, `Union`, `Counter`, `Difference`.
`src/Tasks/` has prebuilt factories for the paper tasks plus `NLSearch` and `HybridNLSearch`
(the SC-plus-NL intersection from the example above):

```python
from src.Tasks.NLSearch import NLSearch

plan = NLSearch("which tables contain customer mailing addresses", k=10)
print(plan.run())
```

## Blend agent
The agent turns a request in natural language into a Blend plan, asks back when the request leaves a
choice open, runs the plan and shows the result tables. Its models are OpenAI-compatible chat endpoints,
one per role, set in the `[Agent]` section of the config (template: `config/nlseeker_example.ini`).

```bash
uv run python -m src.Agent --table <csv> --datalake <dir> --config <ini>
uv run streamlit run src/Agent/App.py -- --datalake <dir> --config <ini>
```

With `--datalake`, the query table can also be found with the NL seeker (`--find <description>`), and a
result can be joined, appended as rows, added as a feature, used to fill blanks, or used as the next
query table.

For a lake built with `make build-lake` (see [REPRODUCE.md](REPRODUCE.md)), pass `--config config/<lake>.ini`
and `--datalake data/Evaluation/NLSeeker/cache/lakes/<dataset>`.

## Evaluation
- `evaluation/nlseeker/`: the NL seeker against Pneuma (retrieval parity and throughput). See its README.
- `benchmarks/nl_integration/`: the NL seeker inside Blend plans (storage, pushdown, runtime, conditional
  cost, operator order, scaling, optimizer).
- `evaluation/agent/`: the agent harness and the case files in `evaluation/agent/cases/`, 10 Simple,
  10 Tasks and 10 Free form gold plans for each of Adventure Works, ChEMBL and Chicago.

Everything runs in the Docker containers of `docker-compose.yml`; `make help` lists the targets.
Results are written to `data/Evaluation/`.

[REPRODUCE.md](REPRODUCE.md) lists the data, the inference endpoints and the commands that reproduce every
experiment.

## Experiments
### Runtime Break Down

![Runtime break down experiment](images/runtime_breakdown.png)


In this section, we assess the runtime distribution for each of the
evaluated operations. We divide the execution of each search plan
into three computational components, namely, DB, which represents the amount of time spent on database query execution, Load,
which shows how much time it takes to load the database query results into the main memory, and Mem, representing the time spend
for processing the data in the main memory. Our goal in Blend
is to reduce the loading time by moving most of the computation
into the database to prevent expensive data movement. By doing
so, we can achieve a runtime efficiency that is comparable to the
stand-alone state-of-the-art baselines. Table 6 shows the results
of this experiment. As the union discovery baseline, i.e., Starmie,
leverages a large language model, this baseline does not use any
database to store the table values. In this case, the loading time
represents the average time the approach takes to generate the
embeddings from the large language model for the query tables.
In all discovery tasks except the union discovery, where DB is
not defined for baseline, Blend consumes a higher percentage
of the time during in-database query execution compared to the
baselines. On average Blend utilizes the database up to 45% more
than the baselines. This results in a drastic reduction of the loading
time. Blend’s loading time is negligible in all cases except in the
MC join discovery. Blend requires more time to load the data for
MC join because, to efficiently prune of the false positives, Blend
requires to read the posting list, including the super keys. These
posting lists, depending on the number of candidate tables, require
additional fetching time. However, the SQL query used in Blend
reduces the number of candidate tables, therefore, it reduces the
loading overhead from 68.2% in MATE to 14.3%.

### BLEND optimizer VS. Postgres and Vertica
As augmentation-by-example leverages various seekers, we evaluated the performance of our query rewriter in the execution engine compared to two baselines that only use the native DBMS optimizer: executing queries independently and then merging the results, and modeling the operator sequence with subquery formulations. According to our experiments on both commercial column store and PostgreSQL, our query rewriter is able to achieve up to 28% and 27% runtime reduction compared to the baselines mentioned above respectively.
