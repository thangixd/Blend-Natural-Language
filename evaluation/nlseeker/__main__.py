import argparse
import json
import os
import shutil
import sys
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

from evaluation.nlseeker.config import EvalConfig, check_layout, for_lake, load_config, parse_config, parse_layout
from evaluation.nlseeker.datasets import DATASETS, FAMILY_FIELDS
from evaluation.nlseeker.offline import check_sources, load_built
from evaluation.nlseeker.prepare import CACHE_DIR, DATA_ROOT, DATASETS_DIR, prepare_dataset, sha256_file
from evaluation.nlseeker.report import TOLERANCE, aggregate, collect_records, write_summary
from evaluation.nlseeker.rundir import RunDir, write_ini
from evaluation.nlseeker.runner import SHARED_DOCUMENTS, TEXTS, Backend, blend_backend, run_backend, write_documents

from evaluation.nlseeker.offline import Lake
from evaluation.nlseeker.prepare import PreparedDataset
from typing import Dict, Optional, Tuple

RESULTS_DIR = DATA_ROOT / 'results'


def _blend(cfg: EvalConfig, lake: Lake, prepared: PreparedDataset, directory: Path,
           artifacts: Path) -> Tuple[Backend, None]:
    ini = write_ini(lake.ini, cfg.nlseeker_for('blend'), cfg.alpha[0], cfg.n[0], artifacts / 'nlseeker.ini')
    return blend_backend(ini, directory / 'nl_runs.jsonl'), None


def _pneuma(cfg: EvalConfig, lake: Lake, prepared: PreparedDataset, directory: Path,
            artifacts: Path) -> Tuple[Backend, Optional[Backend]]:
    from evaluation.nlseeker.pneuma_backend import SharedEmbedder, pneuma_backend
    from src.NLSeeker.Clients import EmbeddingClient, LLMClient
    from src.NLSeeker.Config import NLSeekerConfig

    config = NLSeekerConfig.load(write_ini(lake.ini, cfg.nlseeker_for('pneuma'), cfg.alpha[0], cfg.n[0],
                                           artifacts / 'nlseeker.ini'))
    llm, embedder = LLMClient(config), SharedEmbedder(EmbeddingClient(config), config.embedding_tokenizer)
    own = pneuma_backend(lake.directory / 'pneuma', llm, embedder, prepared.lake_dir)
    return own, (pneuma_backend(lake.directory / 'pneuma_shared', llm, embedder, prepared.lake_dir)
                 if cfg.shared_combos() else None)


BACKENDS = {'blend': _blend, 'pneuma': _pneuma}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog='evaluation.nlseeker',
                                     description='Run the NLSeeker evaluation suite on lakes the offline phase built.')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--config', help='YAML config for a fresh run.')
    group.add_argument('--resume', help='Existing .partial run directory to resume.')
    parser.add_argument('--backend', choices=list(BACKENDS), default='blend',
                        help='the system this process runs; a resumed run keeps its own')
    parser.add_argument('--layout', action='append', required=True, metavar='BACKEND:CPUS:LLM_GPU,EMBEDDING_GPU',
                        help="this backend's cores and its endpoints' GPUs, e.g. blend:0-15:0,1; repeat it for the "
                             'backend running beside it, which must share no core, GPU or endpoint URL')
    parser.add_argument('--lake', help="run only this one of the config's datasets, by its dataset or lake name "
                                       '(a fresh run)')
    parser.add_argument('--min-free-gb', type=float, default=5.0, help='free disk space required before each dataset')
    return parser


def _ping(base_url: str, api_key: str) -> list:
    request = urllib.request.Request(f'{base_url.rstrip("/")}/models',
                                     headers={'Authorization': f'Bearer {api_key}'})
    try:
        response = urllib.request.urlopen(request, timeout=10)
    except urllib.error.HTTPError as error:
        raise ConnectionError(f'endpoint check failed: {base_url} (HTTP {error.code})') from error
    except OSError as error:
        raise ConnectionError(f'endpoint not reachable: {base_url} ({error})') from error
    try:
        body = json.loads(response.read())
    finally:
        response.close()
    return [item['id'] for item in body['data']]


def _count_questions(datasets_dir: Path, name: str, family: str, limit: Optional[int]) -> int:
    spec = DATASETS[name]
    source = FAMILY_FIELDS[family][0]
    if source == 'bc':
        text = (datasets_dir / spec.questions).read_text()
    else:
        with zipfile.ZipFile(datasets_dir / spec.zip) as zf:
            text = zf.read(f'{spec.zip_name}/bx_{spec.zip_name}.jsonl').decode()
    count = sum(1 for line in text.splitlines() if line.strip())
    return count if limit is None else min(count, limit)


def preflight(cfg: EvalConfig, backend: str, layout: Dict[str, dict], datasets_dir: Path) -> dict:
    for name in cfg.datasets:
        spec = DATASETS[name]
        for filename in (spec.tar, spec.zip, spec.questions):
            if not (datasets_dir / filename).is_file():
                raise FileNotFoundError(f'dataset source missing: {datasets_dir / filename}')
    for name in layout:
        if name not in cfg.backends:
            raise ValueError(f'the config has no endpoints for backend {name!r}')
    check_layout(cfg, layout, backend, os.sched_getaffinity(0))
    endpoints = cfg.backends[backend]
    models = {role: {'url': endpoints[f'{role}_base_url'], 'gpu': layout[backend][f'{role}_gpu'],
                     'models': _ping(endpoints[f'{role}_base_url'], cfg.nlseeker[f'{role}_api_key'])}
              for role in ('llm', 'embedding')}
    combos = cfg.combos() + cfg.shared_combos()
    total_questions = sum(
        _count_questions(datasets_dir, name, family, cfg.questions_limit) * len(combos) * cfg.repetitions
        * max(len(cfg.subsets), 1) for name in cfg.datasets for family in cfg.families)
    print(f'{len(cfg.datasets)} dataset(s) x {len(cfg.families)} family/ies x '
          f'{len(combos)} combo(s): {combos}')
    print(f'{total_questions} query/ies planned in total')
    return models


def _versions() -> dict:
    import duckdb
    return {'python': sys.version.split()[0], 'duckdb': duckdb.__version__}


def _run_dataset(cfg: EvalConfig, run: RunDir, backend: str, condition: dict, name: str, directory: Path) -> None:
    from evaluation.nlseeker.datasets import load_questions

    started = time.monotonic()
    dataset = cfg.datasets[0] if cfg.subsets else name
    lake, record, code = load_built(name if cfg.subsets else DATASETS[name].lake, backend)
    prepared = prepare_dataset(DATASETS[dataset], DATASETS_DIR, CACHE_DIR)
    entry = record['entry']
    check_sources(lake, entry, prepared)
    if record['nlseeker']['embedding_model'] != cfg.nlseeker['embedding_model']:
        raise ValueError(f'{lake.name} was embedded with {record["nlseeker"]["embedding_model"]}, the config embeds '
                         f'with {cfg.nlseeker["embedding_model"]}')
    if backend == 'pneuma' and cfg.shared_combos() and 'shared_build' not in entry:
        raise ValueError(f'{lake.name} has no Pneuma index of Blend\'s texts for the attribution arms: its build\'s '
                         f'config had none')
    run.set_dataset(name, lake={'name': lake.name, 'index_code': code, 'built_utc': record['built_utc']}, **entry)
    questions = {family: load_questions(prepared.questions_jsonl if FAMILY_FIELDS[family][0] == 'bc'
                                        else prepared.bx_jsonl, family, cfg.questions_limit)
                 for family in cfg.families}

    built, shared = BACKENDS[backend](cfg, lake, prepared, directory, run.artifacts_dir(name))
    write_documents(built, directory / 'documents.json')
    if shared:
        write_documents(shared, directory / SHARED_DOCUMENTS)
    shutil.copyfile(lake.texts(backend), directory / TEXTS)
    loaded = time.monotonic()

    run.set_dataset(name, status='running', provenance=built.provenance,
                    **({'shared_provenance': shared.provenance} if shared else {}))
    print(f'{name}: {sum(map(len, questions.values()))} questions x {len(cfg.combos())} combo(s) '
          f'+ {len(cfg.shared_combos())} on the shared texts, {cfg.repetitions} pass(es)')
    window = [time.time()]
    run_backend(built, questions, cfg.combos(), directory / 'per_query.jsonl', condition, cfg.shared_combos(), shared,
                cfg.repetitions)
    window.append(time.time())
    finished = time.monotonic()

    run.set_dataset(name, questions_window=window, duration_s=round(finished - started, 1),
                    load_s=round(loaded - started, 1), run_s=round(finished - loaded, 1))


def run_evaluation(cfg: EvalConfig, run: RunDir, backend: str, condition: dict, min_free_bytes: int) -> bool:
    names = [f'{DATASETS[cfg.datasets[0]].lake}_{size}' for size in cfg.subsets] or list(cfg.datasets)
    complete = run.run_datasets(
        names, lambda name, directory: _run_dataset(cfg, run, backend, condition, name, directory), min_free_bytes)
    done = [name for name in names if run.dataset_status(name) == 'complete']
    write_summary(aggregate(collect_records(run.path, done)), run.path)
    run.update_meta(versions=_versions(),
                    config_sha256=sha256_file(run.config_path))
    return complete


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()
    if args.resume and args.lake:
        parser.error('--lake is for a fresh run; a resumed run keeps its lake')
    layout = dict(map(parse_layout, args.layout))
    if args.resume:
        run = RunDir.open_partial(Path(args.resume))
        cfg = load_config(run.config_path)
        backend = run.meta()['backend']
        endpoints = preflight(cfg, backend, layout, DATASETS_DIR)
    else:
        config_text = Path(args.config).read_text()
        if args.lake:
            config_text = for_lake(config_text, args.lake)
        cfg = parse_config(config_text)
        backend = args.backend
        endpoints = preflight(cfg, backend, layout, DATASETS_DIR)
        run = RunDir.create(RESULTS_DIR, f'{cfg.tag}-{backend}', config_text)
        run.update_meta(tolerance_sha256=sha256_file(TOLERANCE))

    condition = {'parallel': len(layout) > 1, 'cpus': sorted(os.sched_getaffinity(0))}
    run.update_meta(backend=backend, endpoints=endpoints, condition=condition, layout=layout)

    complete = run_evaluation(cfg, run, backend, condition, int(args.min_free_gb * 2 ** 30))
    if complete:
        print(f'run complete: {run.finalize()}')
        return 0
    print(f'run incomplete, resume with: --resume {run.path}', file=sys.stderr)
    return 1


if __name__ == '__main__':
    sys.exit(main())
