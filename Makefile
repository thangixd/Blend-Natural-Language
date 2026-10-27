COMPOSE ?= docker compose
CONFIG  ?= evaluation/nlseeker/configs/default.yaml

.DEFAULT_GOAL := help
.PHONY: help build up down shell logs evaluation-nlseeker eval-clean eval-clean-cache \
        eval-profile eval-storage status verify delete-run

help:
	@grep -E '^[a-zA-Z0-9_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
	  | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-22s\033[0m %s\n", $$1, $$2}'

build:  ## Build the Blend image from the working tree, always from scratch.
	$(COMPOSE) build --no-cache

up: build  ## Start the containers (host networking reaches the vLLMs).
	mkdir -p data/indexes data/profiles PneumaData
	$(COMPOSE) up -d

down:  ## Stop the blend container.
	$(COMPOSE) down

shell:  ## Bash inside the running container.
	$(COMPOSE) exec blend bash

logs:  ## Tail container logs.
	$(COMPOSE) logs -f blend

evaluation-nlseeker: BACKEND = blend
evaluation-nlseeker: parity-backend  ## Blend alone: CONFIG=<yaml> or RESUME=<run>, BLEND_GPUS=<llm,embedding> (see parity-backend).

define IN_CONTAINER
	@if $(COMPOSE) ps --services --filter status=running 2>/dev/null | grep -qx blend; then \
	  $(COMPOSE) exec -T blend $(1); \
	else \
	  $(COMPOSE) run --rm -T blend $(1); \
	fi
endef

eval-clean:  ## Delete one run dir: make eval-clean RUN=<run-dir name under results/>.
ifndef RUN
	$(error RUN is required, e.g. make eval-clean RUN=2026-08-05T14-30-00Z__default)
endif
	@case '$(RUN)' in *..*|*[!A-Za-z0-9_.:-]*|"") echo "invalid RUN: $(RUN)"; exit 1;; esac
	$(call IN_CONTAINER,rm -rf 'data/Evaluation/NLSeeker/results/$(RUN)')

eval-clean-cache:  ## Delete the extracted lake/bench cache (PneumaData/ is never touched).
	$(call IN_CONTAINER,rm -rf data/Evaluation/NLSeeker/cache)

check-run = @case '$(1)' in *..*|*[!A-Za-z0-9_.:/-]*|"") echo "invalid run: $(1)"; exit 1;; esac

define EVAL
	$(if $(RESUME),$(call check-run,$(RESUME)))
	$(COMPOSE) exec -T blend python -m benchmarks.nl_integration.$(1) \
	  $(if $(RESUME),--resume 'data/Evaluation/$(RESUME)',$(if $(LAKE),--lake $(LAKE)))
endef

eval-profile:  ## Lake profile. LAKE="<lakes>" (default every lake), RESUME=<run>.
	$(call EVAL,profile)

eval-storage:  ## D3 storage. LAKE="<lakes>" (default every lake), RESUME=<run>.
	$(call EVAL,storage)

status:  ## Every run: complete, partial or failed, and each lake's status.
	$(call IN_CONTAINER,python -m evaluation.nlseeker.rundir status data/Evaluation/NLIntegration data/Evaluation/NLSeeker/results)

verify:  ## Local: recompute a pulled run's checksums, print the deletion code. RUN=<run under data/Evaluation>.
	$(call check-run,$(RUN))
	uv run python -m evaluation.nlseeker.rundir verify 'data/Evaluation/$(RUN)'

delete-run:  ## Server: delete a sealed run only with its code from make verify. RUN=<run> CODE=<code>.
	$(call check-run,$(RUN))
	@case '$(CODE)' in *[!0-9a-f]*|"") echo "invalid CODE: $(CODE)"; exit 1;; esac
	$(call IN_CONTAINER,python -m evaluation.nlseeker.rundir delete 'data/Evaluation/$(RUN)' '$(CODE)')

SERVER     ?=
SERVER_DIR ?=
SSH        ?= ssh -o ClearAllForwardings=yes
RSYNC       = rsync -a --partial-dir=.rsync-partial -e '$(SSH)'
PACKAGE_fetaqa = fetaqa fetaqa_625 fetaqa_1250 fetaqa_2500 fetaqa_5000 fetaqa_10330
THROUGHPUT ?=
SIMULATE ?=
PACKAGE = $(or $(if $(THROUGHPUT),$(PACKAGE_$(LAKE))),$(LAKE))
INDEX_MANIFESTS = $(foreach lake,$(PACKAGE),'data/indexes/$(lake).MANIFEST.json')

check-remote = @case '$(SERVER)@$(SERVER_DIR)' in @*|*@|*[!A-Za-z0-9_.@/~-]*) \
  echo "set SERVER=<ssh destination> SERVER_DIR=<checkout on the server>"; exit 1;; esac
check-lake = @case '$(LAKE)' in *[!A-Za-z0-9_]*|"") echo "invalid LAKE: $(LAKE)"; exit 1;; esac
check-code = @case '$(CODE)' in *[!0-9a-f]*|"") echo "invalid CODE: $(CODE)"; exit 1;; esac

.PHONY: push-index check-index pull pull-index verify-index delete-index

push-index:  ## Local: seal LAKE's index (or package), rsync it to the server, verify it there with the local code.
	$(check-remote)
	$(check-lake)
	for lake in $(PACKAGE); do uv run python -m benchmarks.nl_integration.common seal-index "$$lake" || exit 1; done
	checked=$$(uv run python -m evaluation.nlseeker.rundir verify-index $(INDEX_MANIFESTS)) || \
	  { echo "$$checked"; exit 1; }; \
	code=$${checked#index ok: } && echo "sealed: $$code" && \
	  files=$$(uv run python -m evaluation.nlseeker.rundir index-files $(INDEX_MANIFESTS)) && \
	  printf '%s\n' "$$files" | $(RSYNC) --copy-links --files-from=- . '$(SERVER):$(SERVER_DIR)/' && \
	  $(SSH) '$(SERVER)' "cd $(SERVER_DIR) && make check-index LAKE=$(LAKE) CODE=$$code"

check-index:  ## Server: verify an index (or package) against its manifests in the container. LAKE=<lake> [CODE=<code>].
	$(check-lake)
	$(if $(CODE),$(check-code))
	$(call IN_CONTAINER,python -m evaluation.nlseeker.rundir verify-index $(INDEX_MANIFESTS) $(CODE))

pull:  ## Local: rsync a run from the server, then verify it. RUN=<run under data/Evaluation>.
	$(check-remote)
	$(call check-run,$(RUN))
	$(RSYNC) --mkpath '$(SERVER):$(SERVER_DIR)/data/Evaluation/$(RUN)/' 'data/Evaluation/$(RUN)/'
	$(if $(filter Agent/%,$(RUN)),,$(MAKE) --no-print-directory verify RUN='$(RUN)')

pull-index:  ## Local: rsync an index (or package) built on the server, then verify it. LAKE=<lake>.
	$(check-remote)
	$(check-lake)
	printf '%s\n' $(INDEX_MANIFESTS) | $(RSYNC) --ignore-existing --files-from=- '$(SERVER):$(SERVER_DIR)/' .
	uv run python -m evaluation.nlseeker.rundir index-files $(INDEX_MANIFESTS) \
	  | $(RSYNC) --ignore-existing --files-from=- '$(SERVER):$(SERVER_DIR)/' .
	$(MAKE) --no-print-directory verify-index LAKE='$(LAKE)'

verify-index:  ## Local: recompute a pulled index's (or package's) checksums, print its deletion code. LAKE=<lake>.
	$(check-lake)
	uv run python -m evaluation.nlseeker.rundir verify-index $(INDEX_MANIFESTS)

delete-index:  ## Server: delete an index (or package) only with its code from make verify-index. LAKE=<lake> CODE=<code>.
	$(check-lake)
	$(check-code)
	$(call IN_CONTAINER,python -m evaluation.nlseeker.rundir delete-index $(INDEX_MANIFESTS) '$(CODE)')

.PHONY: eval-d1
eval-d1:  ## D1 exact scan vs HNSW on every real lake. LAKE="<lakes>" (default every lake), RESUME=<run>; SIMULATE=<lake> adds the judge and the simulated lakes on that lake.
	$(call EVAL,scaling)$(if $(SIMULATE), --simulate-from '$(SIMULATE)')

.PHONY: eval-u1
eval-u1:  ## U1 SC with NL. LAKE="<lakes>" (default every lake), RESUME=<run>.
	$(call EVAL,order)

.PHONY: eval-d4
eval-d4:  ## D4 NL time. LAKE="<lakes>" (default every lake), RESUME=<run>.
	$(call EVAL,runtime)

.PHONY: eval-u2
eval-u2:  ## U2 Keyword with NL. LAKE="<lakes>" (default every lake), RESUME=<run>.
	$(call EVAL,regime)

BLEND_CPUS  ?= 0-15
PNEUMA_CPUS ?= 16-31
export BLEND_CPUS PNEUMA_CPUS
PARITY_LAYOUT = $(if $(BLEND_GPUS),--layout 'blend:$(BLEND_CPUS):$(BLEND_GPUS)') \
                $(if $(PNEUMA_GPUS),--layout 'pneuma:$(PNEUMA_CPUS):$(PNEUMA_GPUS)')
.PHONY: parity-backend parity-pilot-check

parity-backend:  ## One parity backend on lakes the offline phase built. BACKEND=blend|pneuma (also on resume: it names the container), CONFIG=<yaml> or RESUME=<run>, BLEND_GPUS=0,1 and/or PNEUMA_GPUS=2,3.
	@case '$(BACKEND)' in blend|pneuma) ;; *) echo "invalid BACKEND: $(BACKEND)"; exit 1;; esac
	@case '$(BLEND_GPUS)$(PNEUMA_GPUS)' in *[!0-9,]*|"") echo "BLEND_GPUS and/or PNEUMA_GPUS are required, e.g. BLEND_GPUS=0,1"; exit 1;; esac
	$(if $(RESUME),$(call check-run,$(RESUME)))
	$(COMPOSE) exec -T $(BACKEND) python -m evaluation.nlseeker $(PARITY_LAYOUT) \
	  $(if $(RESUME),--resume 'data/Evaluation/NLSeeker/results/$(RESUME)',--config '$(CONFIG)' --backend $(BACKEND))

parity-pilot-check:  ## The parity pilot rerun's criteria on two runs (shared-text counts, BM25 lists, timings). BLEND_RUN=<run> PNEUMA_RUN=<run> (names under results/).
	$(call check-run,$(BLEND_RUN))
	$(call check-run,$(PNEUMA_RUN))
	$(call IN_CONTAINER,python -m evaluation.nlseeker.report 'data/Evaluation/NLSeeker/results/$(BLEND_RUN)' 'data/Evaluation/NLSeeker/results/$(PNEUMA_RUN)')

.PHONY: eval-d5
eval-d5:  ## D5 NL's conditional cost. LAKE="<lakes>" (default every lake), RESUME=<run>; reads the newest D4 run unless D4=<run under data/Evaluation>.
	$(if $(D4),$(call check-run,$(D4)))
	$(call EVAL,conditional)$(if $(D4), --d4 'data/Evaluation/$(D4)')

.PHONY: eval-optimizer
eval-optimizer:  ## Theorem 1 with Blend's own seekers. LAKE="<lakes>" (default every lake), RESUME=<run>.
	$(call EVAL,optimizer)

.PHONY: eval-d2
eval-d2:  ## D2 restriction inside retrieval vs post-filter. LAKE="<lakes>" (default every lake), RESUME=<run>.
	$(call EVAL,pushdown)

PARITY_CONFIG ?= evaluation/nlseeker/configs/parity.yaml
PARITY_RESULTS = data/Evaluation/NLSeeker/results
.PHONY: parity
define PAIR
	@($(COMPOSE) exec -T blend python -m evaluation.nlseeker $(PARITY_LAYOUT) --config '$(1)' --backend blend \
	  --lake '$(LAKE)' || { failed=$$?; echo 'blend failed: stopping pneuma'; $(COMPOSE) restart pneuma; exit $$failed; }) & \
	blend=$$!; \
	$(COMPOSE) exec -T pneuma python -m evaluation.nlseeker $(PARITY_LAYOUT) --config '$(1)' \
	  --backend pneuma --lake '$(LAKE)'; pneuma=$$?; \
	wait $$blend; blend=$$?; echo "exit: blend $$blend, pneuma $$pneuma"; [ $$blend -eq 0 ] && [ $$pneuma -eq 0 ]
endef
check-gpus = @case '$(BLEND_GPUS)/$(PNEUMA_GPUS)' in *[!0-9,/]*|/*|*/) \
  echo "BLEND_GPUS and PNEUMA_GPUS are required, e.g. 0,1 and 2,3"; exit 1;; esac
check-tolerance = @grep -q '^status: approved' evaluation/nlseeker/parity_tolerance.yaml || \
  { echo "evaluation/nlseeker/parity_tolerance.yaml is not approved: no parity run starts"; exit 1; }

parity:  ## Server: parity on one built lake, both backends at once. LAKE=<lake or dataset> BLEND_GPUS=<llm,embedding> PNEUMA_GPUS=<llm,embedding> [PARITY_CONFIG=<yaml>].
	$(check-lake)
	$(check-gpus)
	$(check-tolerance)
	$(call PAIR,$(PARITY_CONFIG))

PARITY_OUT ?= data/Evaluation/NLSeeker/parity
.PHONY: parity-report
parity-report:  ## Local: the parity report over pulled runs. RUNS="<blend run> <pneuma run> ..." (one pair per lake, names under results/), PARITY_OUT=<dir>.
	@for run in $(RUNS); do case "$$run" in *..*|*[!A-Za-z0-9_.:-]*) echo "invalid run: $$run"; exit 1;; esac; done
	uv run python -m evaluation.nlseeker.report parity '$(PARITY_OUT)' $(foreach run,$(RUNS),'$(PARITY_RESULTS)/$(run)')

.PHONY: parity-diagnosis
parity-diagnosis:  ## Local: why the shared-text lists differ (BM25 ties, Pneuma's pool against the exact scan). RUNS like parity-report.
	@for run in $(RUNS); do case "$$run" in *..*|*[!A-Za-z0-9_.:-]*) echo "invalid run: $$run"; exit 1;; esac; done
	uv run python -m evaluation.nlseeker.diagnosis 'data/Evaluation/NLSeeker/diagnosis' $(foreach run,$(RUNS),'$(PARITY_RESULTS)/$(run)')

THROUGHPUT_CONFIG ?= evaluation/nlseeker/configs/throughput.yaml
.PHONY: parity-efficiency
parity-efficiency:  ## Server: throughput on one built FeTaQA subset, both backends at once. LAKE=fetaqa_<tables> BLEND_GPUS=<llm,embedding> PNEUMA_GPUS=<llm,embedding> [THROUGHPUT_CONFIG=<yaml>].
	$(check-lake)
	$(check-gpus)
	$(call PAIR,$(THROUGHPUT_CONFIG))

EFFICIENCY_OUT ?= data/Evaluation/NLSeeker/efficiency
.PHONY: parity-efficiency-report
parity-efficiency-report:  ## Local: throughput, offline stages and storage over pulled runs. RUNS="<run> ..." (parity and throughput runs of both backends, names under results/), EFFICIENCY_OUT=<dir>.
	@for run in $(RUNS); do case "$$run" in *..*|*[!A-Za-z0-9_.:-]*) echo "invalid run: $$run"; exit 1;; esac; done
	uv run python -m evaluation.nlseeker.report efficiency '$(EFFICIENCY_OUT)' $(foreach run,$(RUNS),'$(PARITY_RESULTS)/$(run)')

BACKENDS ?= blend pneuma
BUILD_CONFIG ?=
BUILD_LLM_CONCURRENCY ?= 32
BUILD_EMBEDDING_CONCURRENCY ?= 8
.PHONY: build-lake
build-lake:  ## Server: build and seal one lake's indexes (THROUGHPUT=1: with the FeTaQA subsets), both backends one after the other. LAKE=<lake> BLEND_GPUS=<llm,embedding> PNEUMA_GPUS=<llm,embedding> [BACKENDS=pneuma] [BUILD_CONFIG=<yaml>, for every lake of the package] [BUILD_LLM_CONCURRENCY=32] [BUILD_EMBEDDING_CONCURRENCY=8].
	$(check-lake)
	$(check-gpus)
	@for lake in $(PACKAGE); do \
	  if [ -f "data/indexes/$$lake.MANIFEST.json" ]; then echo "$$lake: sealed, skipping"; continue; fi; \
	  case $$lake in fetaqa_*) config='$(or $(BUILD_CONFIG),$(THROUGHPUT_CONFIG))';; \
	    *) config='$(or $(BUILD_CONFIG),$(PARITY_CONFIG))';; esac; \
	  for backend in $(BACKENDS); do \
	    $(COMPOSE) exec -T $$backend python -m evaluation.nlseeker.offline build "$$lake" --backend $$backend \
	      --config "$$config" --llm-concurrency $(BUILD_LLM_CONCURRENCY) \
	      --embedding-concurrency $(BUILD_EMBEDDING_CONCURRENCY) $(PARITY_LAYOUT) || exit 1; \
	  done; \
	  $(COMPOSE) exec -T blend python -m evaluation.nlseeker.offline seal "$$lake" || exit 1; \
	done

INTEGRATION_adventure_works = profile storage pushdown runtime conditional order regime scaling optimizer
INTEGRATION_chembl          = profile storage pushdown runtime conditional order regime scaling optimizer
INTEGRATION_chicago_open    = profile storage pushdown runtime conditional order regime scaling optimizer
INTEGRATION_public_bi       = profile storage pushdown runtime conditional order regime scaling optimizer
INTEGRATION_fetaqa          = profile storage pushdown runtime conditional order regime scaling optimizer
PARITY_LAKES = adventure_works chembl chicago_open public_bi fetaqa
SMOKE_pushdown    = --questions 5
SMOKE_runtime     = --inputs 10 --questions 5 --k 5 10 --n 3 5
SMOKE_conditional = --inputs 2
SMOKE_order       = --questions 3
SMOKE_regime      = --questions 3
SMOKE_scaling     = --questions 5 --sizes 10000,50000
SMOKE_optimizer   = --pairs 5
EVAL_RESULTS = NLIntegration NLSeeker/results

.PHONY: eval-lake eval-all
eval-lake:  ## Server: every experiment planned for one pushed lake (or package), then its run ids. LAKE=<lake> BLEND_GPUS=<llm,embedding> PNEUMA_GPUS=<llm,embedding> [SMOKE=1] [PARITY_CONFIG=<yaml>] [THROUGHPUT=1: the FeTaQA throughput subsets] [SIMULATE=<lake>: see eval-d1].
	$(check-lake)
	$(check-gpus)
	$(check-tolerance)
	$(MAKE) --no-print-directory check-index LAKE='$(LAKE)'
	@stamp=$$(mktemp); failed=; skipped=; \
	$(foreach experiment,$(INTEGRATION_$(LAKE)),echo '== $(experiment) on $(LAKE)'; \
	  case '$(experiment)'"$$failed " in (conditional*' runtime '*) \
	    echo 'D4 failed: D5 would read an older D4 run'; skipped="$$skipped $(experiment)";; \
	  (*) $(COMPOSE) exec -T blend python -m benchmarks.nl_integration.$(experiment) --lake '$(LAKE)' \
	    $(if $(SMOKE),$(SMOKE_$(experiment))) $(if $(and $(SIMULATE),$(filter scaling,$(experiment))),--simulate-from '$(SIMULATE)') || failed="$$failed $(experiment)";; esac;) \
	$(if $(filter $(LAKE),$(PARITY_LAKES)),echo '== parity on $(LAKE)'; \
	  $(MAKE) --no-print-directory parity LAKE='$(LAKE)' || failed="$$failed parity";) \
	$(foreach subset,$(filter-out $(LAKE),$(PACKAGE)),echo '== throughput on $(subset)'; \
	  $(MAKE) --no-print-directory parity-efficiency LAKE='$(subset)' || failed="$$failed throughput-$(subset)";) \
	echo "runs of $(LAKE):"; cd data/Evaluation && find $(EVAL_RESULTS) -mindepth 1 -maxdepth 1 -type d -newer "$$stamp" 2>/dev/null | sort; \
	rm -f "$$stamp"; [ -z "$$skipped" ] || echo "skipped on $(LAKE):$$skipped"; \
	[ -z "$$failed" ] || { echo "failed on $(LAKE):$$failed"; exit 1; }

eval-all:  ## Server: eval-lake on every lake, when all of them are pushed; one lake failing leaves the others. BLEND_GPUS=<llm,embedding> PNEUMA_GPUS=<llm,embedding>.
	@failed=; for lake in $(PARITY_LAKES); do \
	  $(MAKE) --no-print-directory eval-lake LAKE=$$lake || failed="$$failed $$lake"; \
	done; [ -z "$$failed" ] || { echo "failed lakes:$$failed"; exit 1; }

OWN_BUILD = ( if $(COMPOSE) exec -T $(1) python -m evaluation.nlseeker.offline build "$$lake" --backend $(1) $(3) \
    $$concurrent --config "$$config" --llm-concurrency $(BUILD_LLM_CONCURRENCY) \
      --embedding-concurrency $(BUILD_EMBEDDING_CONCURRENCY) $(PARITY_LAYOUT); then touch "$$state/$(1).done"; else failed=$$?; \
  if mkdir "$$state/stop" 2>/dev/null; then echo $(1) > "$$state/failed"; \
    [ -f "$$state/$(2).done" ] || { echo "$$lake: $(1) failed: stopping $(2)"; $(COMPOSE) restart $(2); }; fi; \
  exit $$failed; fi ) &
.PHONY: build-lake-parallel
build-lake-parallel:  ## Server: build-lake with Blend's build and Pneuma's own build at the same time. LAKE=<lake> BLEND_GPUS=<llm,embedding> PNEUMA_GPUS=<llm,embedding> [BUILD_CONFIG=<yaml>, for every lake of the package] [BUILD_LLM_CONCURRENCY=32] [BUILD_EMBEDDING_CONCURRENCY=8].
	$(check-lake)
	$(check-gpus)
	@for lake in $(PACKAGE); do \
	  if [ -f "data/indexes/$$lake.MANIFEST.json" ]; then echo "$$lake: sealed, skipping"; continue; fi; \
	  case $$lake in fetaqa_*) config='$(or $(BUILD_CONFIG),$(THROUGHPUT_CONFIG))';; \
	    *) config='$(or $(BUILD_CONFIG),$(PARITY_CONFIG))';; esac; \
	  state=$$(mktemp -d); concurrent=--concurrent; \
	  for side in blend pneuma; do \
	    if [ -f "data/indexes/$$lake/build_$$side.json" ]; then \
	      echo "$$lake: $$side's own build is done"; touch "$$state/$$side.done"; concurrent=; fi; \
	  done; \
	  if [ ! -f "$$state/blend.done" ]; then $(call OWN_BUILD,blend,pneuma,) fi; \
	  if [ ! -f "$$state/pneuma.done" ]; then $(call OWN_BUILD,pneuma,blend,--part own) fi; \
	  wait; \
	  if [ -f "$$state/failed" ]; then echo "$$lake: $$(cat "$$state/failed") failed; not sealed"; rm -rf "$$state"; \
	    exit 1; fi; \
	  rm -rf "$$state"; \
	  $(COMPOSE) exec -T pneuma python -m evaluation.nlseeker.offline build "$$lake" --backend pneuma --part shared \
	    --config "$$config" --llm-concurrency $(BUILD_LLM_CONCURRENCY) \
	    --embedding-concurrency $(BUILD_EMBEDDING_CONCURRENCY) $(PARITY_LAYOUT) || exit 1; \
	  $(COMPOSE) exec -T blend python -m evaluation.nlseeker.offline seal "$$lake" || exit 1; \
	done

AGENT_PARALLEL ?= 10
AGENT_CASES ?= evaluation/agent/cases/aw_tiers.yaml evaluation/agent/cases/chembl_tiers.yaml \
  evaluation/agent/cases/chicago_tiers.yaml
CASES ?= $(AGENT_CASES)
REPEATS ?= 3
HARNESS = $(COMPOSE) exec -T blend python -m evaluation.agent.harness
MODEL ?= Qwen3.8-27B
URL ?= http://localhost:8012/v1
TOKENIZER ?= Qwen/Qwen3.8-27B
AGENT_FLAGS = --parallel $(AGENT_PARALLEL) $(if $(CASE),--case $(CASE)) $(if $(EFFORT),--effort $(EFFORT)) \
  $(if $(filter off,$(THINKING)),--no-thinking) --model '$(MODEL)' --url '$(URL)' --tokenizer '$(TOKENIZER)' \
  $(if $(PLANNER),--planner $(PLANNER)) $(if $(SPECIFIED),--specified)
.PHONY: eval-agent eval-agent-profile
eval-agent:  ## Server: agent cases in the container. RUN=<folder under data/Evaluation/Agent> [CASES="<case files>", default the three tier files] [CASE="<ids>"] [AGENT_PARALLEL=10] [EFFORT=low|medium|xhigh] [THINKING=off] [PROFILE=off] [PLANNER=text|single_call] [SPECIFIED=1, level 0 and inferable only] [MODEL=<name> URL=<endpoint> TOKENIZER=<hf id>, default Qwen3.8-27B on :8012, also the NL seeker's judge]
	$(call check-run,$(RUN))
	$(HARNESS) --cases $(CASES) --out 'data/Evaluation/Agent/$(RUN)' $(AGENT_FLAGS) $(if $(filter off,$(PROFILE)),--no-profile)

eval-agent-profile:  ## Server: semantic profile on against off, both arms at once, REPEATS times, then runs.csv, cases.csv, sign_test.csv. RUN=<folder> [REPEATS=3] [CASES=...] [AGENT_PARALLEL=10, per arm] [EFFORT, THINKING, MODEL/URL/TOKENIZER as eval-agent]
	$(call check-run,$(RUN))
	@for repeat in $$(seq $(REPEATS)); do \
	  echo "== repeat $$repeat"; \
	  $(HARNESS) --cases $(CASES) --out "data/Evaluation/Agent/$(RUN)/on-r$$repeat" $(AGENT_FLAGS) & on=$$!; \
	  $(HARNESS) --cases $(CASES) --out "data/Evaluation/Agent/$(RUN)/off-r$$repeat" $(AGENT_FLAGS) --no-profile & off=$$!; \
	  wait $$on; on=$$?; wait $$off; off=$$?; \
	  [ $$on -eq 0 ] && [ $$off -eq 0 ] || { echo "repeat $$repeat failed: on $$on, off $$off"; exit 1; }; \
	done
	$(COMPOSE) exec -T blend python -m evaluation.agent.profile_ablation 'data/Evaluation/Agent/$(RUN)' --cases $(CASES)
