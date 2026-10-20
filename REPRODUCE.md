# Reproducing the evaluation

## Setup

We ran everything on one server with an AMD EPYC 7543 (32 cores), 500 GB of memory and NVIDIA A100 GPUs (40 GB).

| Model | Used for |
|---|---|
| Qwen2.5-7B-Instruct | NLSeeker and Pneuma: column descriptions and the judge (temperature 0) |
| bge-base-en-v1.5 | NLSeeker and Pneuma: embeddings |
| Qwen3.8-27B | The agent (reasoning effort xhigh) |
| Qwen3.5-9B | The agent in the 9B conditions |

NLSeeker and Pneuma each get their own pair of models and use alpha 0.5 and n 5.

Set your model endpoints in `config/<lake>.ini` and `evaluation/nlseeker/configs/*.yaml`. For the agent, pass
`MODEL`, `URL` and `TOKENIZER` to `make eval-agent`.

## Steps

### 1. Get the data

Download Pneuma's benchmark files from https://tubcloud.tu-berlin.de/s/BHEHTiqRANi552R into `PneumaData/` in the
repository root. Keep the file names.

### 2. Start the models

Start the model endpoints from the table above.

### 3. Build the indexes

Start the containers:

```bash
make up
```

They use CPU cores 0-15 and 16-31 and 200 GB of memory each. You can change this with `BLEND_CPUS`, `PNEUMA_CPUS`,
`BLEND_MEM` and `PNEUMA_MEM`.

Then either download our indexes or build your own.

**Download:** get the indexes and semantic profiles from https://tubcloud.tu-berlin.de/s/z7oBXYeE4MKQKN7 and put them
in `data/indexes/` and `data/profiles/`.

**Build:** for all five lakes,

```bash
for lake in adventure_works chembl chicago_open public_bi fetaqa; do
  make build-lake LAKE=$lake BLEND_GPUS=<gpus> PNEUMA_GPUS=<gpus>
done
```

or for one lake, `make build-lake LAKE=<lake> BLEND_GPUS=<gpus> PNEUMA_GPUS=<gpus>`.

### 4. Run the NLSeeker experiments

This runs the NLSeeker experiments and compares NLSeeker against Pneuma on every lake:

```bash
make eval-all BLEND_GPUS=<gpus> PNEUMA_GPUS=<gpus>
```

For one lake, use `make eval-lake LAKE=<lake> BLEND_GPUS=<gpus> PNEUMA_GPUS=<gpus>`.

Two extras are off by default:

- `THROUGHPUT=1` (on `build-lake` and `eval-all`) adds the throughput runs on five FeTaQA subsets.
- `SIMULATE=chicago_open` adds the judge and the simulated lakes to the scaling experiment on Chicago.

Then, for each lake, compute the paired tests and the int8 storage:

```bash
docker compose exec -T blend python -m benchmarks.nl_integration.analysis --lake <lake> \
  --pushdown <run> --order <run> --regime <run>
docker compose exec -T blend python -m benchmarks.nl_integration.int8_storage --lake <lake> --scratch /tmp/int8
```

The reports over the parity and throughput runs run on your own machine. Run `uv sync --all-extras` once, then:

```bash
make parity-report RUNS="<runs>"
make parity-diagnosis RUNS="<runs>"
make parity-efficiency-report RUNS="<runs>"
```

### 5. Run the agent

The agent only needs steps 1-3, and only three lakes: `adventure_works`, `chembl` and `chicago_open`. It also needs
the Qwen3.8-27B endpoint and NLSeeker's embedding endpoint.

| Condition | Runs | Command |
|---|---|---|
| 27B, effort xhigh | 3 | `make eval-agent RUN=27b-xhigh-r<n>` |
| 27B, effort medium | 3 | `make eval-agent RUN=27b-medium-r<n> EFFORT=medium` |
| 27B, effort low | 3 | `make eval-agent RUN=27b-low-r<n> EFFORT=low` |
| 27B, effort medium, no profile | 3 | `make eval-agent RUN=27b-medium-noprof-r<n> EFFORT=medium PROFILE=off` |
| 27B, effort xhigh, no profile | 3 | `make eval-agent RUN=27b-xhigh-noprof-r<n> PROFILE=off` |
| 9B, thinking, profile on and off | 3 each | `make eval-agent-profile RUN=9b-thinking MODEL=Qwen3.5-9B URL=<endpoint> TOKENIZER=Qwen/Qwen3.5-9B` |
| 9B, no thinking, profile on and off | 3 each | `make eval-agent-profile RUN=9b-nothink THINKING=off MODEL=Qwen3.5-9B URL=<endpoint> TOKENIZER=Qwen/Qwen3.5-9B` |

Run the 27B rows with `<n>` = 1, 2 and 3. Every repeat needs its own `RUN`, because a run skips the cases it
already has.

### 6. Summarize an agent run

```bash
docker compose exec -T blend python -m evaluation.agent.harness --summary \
  --cases evaluation/agent/cases/*_tiers.yaml --out data/Evaluation/Agent/<RUN>
docker compose exec -T blend python -m evaluation.agent.analysis \
  --runs data/Evaluation/Agent/<RUN> --cases evaluation/agent/cases/*_tiers.yaml \
  --out data/Evaluation/Agent/<RUN>/analysis
```

For the 9B profile runs, `<RUN>` is one arm and repeat, for example `9b-thinking/on-r1`. Their sign tests are already
in `data/Evaluation/Agent/9b-thinking/` and `data/Evaluation/Agent/9b-nothink/`.

Keep `--out` inside `data/Evaluation/`. That is where all results go, and it is the only output folder you can see
outside the container.

## Our results

`results/` holds our runs, grouped by lake:

- `Agent/<lake>/`: one folder per agent run. `Agent/runs.csv` lists each run's condition and repeat,
  `Agent/results.csv` summarizes one run per condition, and `Agent/9b-profile-ablation/{think,nothink}/` and
  `Agent/27b-profile-ablation/{xhigh,medium}/` have the sign tests of the semantic profile.
- `NLIntegration/<lake>/`: the NLSeeker experiments, the int8 storage, and `analysis_<lake>.json` with the paired
  tests. FeTaQA's runs are older and miss the answer tables; they are in `NLIntegration/fetaqa/answers_fetaqa.json`.
- `NLSeeker/<lake>/`: the parity runs of both systems and the diagnosis, and on FeTaQA the throughput runs.
  `NLSeeker/parity/` and `NLSeeker/efficiency/` have the reports over all lakes.

Files over 50 MB are compressed with zstd (`zstd -d <file>.zst`). We left out the parity and throughput runs' copy
of the build's texts (`texts.json`) and the agent runs' error logs.

Two summaries can be rebuilt from these files:

```bash
uv run python -m benchmarks.nl_integration.analysis --lake fetaqa --results results/NLIntegration/fetaqa \
  --pushdown <run> --order <run> --regime <run> --answers results/NLIntegration/fetaqa/answers_fetaqa.json
uv run python -m evaluation.agent.profile_ablation <out> --cases evaluation/agent/cases/*_tiers.yaml \
  --exclude awt-t10-L1 --arm on 1 <the three folders of repeat 1> ... --arm off 3 <the three folders of repeat 3>
```
