# Reproducing the evaluation

## Setup

- One server with an AMD EPYC 7543 (32 cores), 500 GB of memory and NVIDIA A100 GPUs with 40 GB each.
- Models served by vLLM:
  - NLSeeker and Pneuma: Qwen2.5-7B-Instruct for the column descriptions and the judge (temperature 0), and
    bge-base-en-v1.5 for the embeddings, one pair per system. Prefix caching off.
  - Agent: Qwen3.8-27B with the server's default reasoning effort (xhigh). It is also the NLSeeker's judge
    inside the agent's plans. Qwen3.5-9B for the 9B conditions.
- NLSeeker and Pneuma: alpha 0.5, n 5.
- The endpoints are set in `config/<lake>.ini` and `evaluation/nlseeker/configs/*.yaml`.

## Steps

1. Download Pneuma's benchmark files (<link>) into `PneumaData/` in the repository root, keeping their names.
2. Start the model endpoints.
3. Build the containers and the indexes of the five lakes. The containers take CPU cores 0-15 and 16-31 and
   200 GB of memory each; `BLEND_CPUS`, `PNEUMA_CPUS`, `BLEND_MEM` and `PNEUMA_MEM` change that.

   ```bash
   make up
   for lake in adventure_works chembl chicago_open public_bi fetaqa; do
     make build-lake LAKE=$lake BLEND_GPUS=<gpus> PNEUMA_GPUS=<gpus>
   done
   ```

4. Run the NLSeeker experiments and the parity runs against Pneuma on every lake:

   ```bash
   make eval-all BLEND_GPUS=<gpus> PNEUMA_GPUS=<gpus>
   ```

   Every lake runs the same experiments. Two additions are off by default, and the repository holds their runs
   from the thesis:

   - `THROUGHPUT=1` (on `build-lake` and `eval-all`) adds the throughput runs on five FeTaQA subsets, in
     `data/Evaluation/NLSeeker/results/*__throughput-fetaqa_*`.
   - `SIMULATE=chicago_open` adds the judge and the simulated lakes to the scaling experiment on Chicago, in
     `data/Evaluation/NLIntegration/2026-10-05T23-28-20Z__scaling`.

   The throughput runs leave out their copy of the build's texts (`texts.json`, up to 240 MB each), so
   `make verify` reports these files as missing.

   Then, per lake, the paired tests and the int8 storage:

   ```bash
   docker compose exec -T blend python -m benchmarks.nl_integration.analysis --lake <lake> \
     --pushdown <run> --order <run> --regime <run>
   docker compose exec -T blend python -m benchmarks.nl_integration.int8_storage --lake <lake> --scratch /tmp/int8
   ```

   And the reports over the parity and throughput runs, which run on the host (after `uv sync --all-extras`):

   ```bash
   make parity-report RUNS="<runs>"
   make parity-diagnosis RUNS="<runs>"
   make parity-efficiency-report RUNS="<runs>"
   ```

5. Run the agent conditions, the 27B ones first (they build the semantic profiles that every later run reuses):

   | Condition | Runs | Command |
   |---|---|---|
   | 27B, effort xhigh | 3 | `make eval-agent RUN=27b-xhigh-r1` |
   | 27B, effort medium | 3 | `make eval-agent RUN=27b-medium-r1 EFFORT=medium` |
   | 27B, effort low | 3 | `make eval-agent RUN=27b-low-r1 EFFORT=low` |
   | 27B, effort medium, no profile | 1 | `make eval-agent RUN=27b-medium-noprof EFFORT=medium PROFILE=off` |
   | 27B, effort xhigh, no profile, underspecified requests | 1 | `make eval-agent RUN=27b-xhigh-noprof PROFILE=off CASE="<ids of levels 1 to 3>"` |
   | 9B, thinking, profile on and off | 3 each | `make eval-agent-profile RUN=9b-thinking MODEL=Qwen3.5-9B URL=<endpoint> TOKENIZER=Qwen/Qwen3.5-9B` |
   | 9B, no thinking | 1 | `make eval-agent RUN=9b-nothink THINKING=off MODEL=Qwen3.5-9B URL=<endpoint> TOKENIZER=Qwen/Qwen3.5-9B` |
   | 9B, no thinking, no profile | 1 | as above with `PROFILE=off` |

6. Summarize a finished agent run:

   ```bash
   docker compose exec -T blend python -m evaluation.agent.harness --summary \
     --cases evaluation/agent/cases/*_tiers.yaml --out data/Evaluation/Agent/<RUN>
   docker compose exec -T blend python -m evaluation.agent.analysis \
     --runs data/Evaluation/Agent/<RUN> --cases evaluation/agent/cases/*_tiers.yaml --out <dir>
   ```

Results are written to `data/Evaluation/`.
