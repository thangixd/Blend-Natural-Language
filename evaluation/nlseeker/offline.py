import argparse
import configparser
import fcntl
import hashlib
import json
import os
import shutil
import sys
import time
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path

import duckdb

from evaluation.nlseeker.build import export_texts
from evaluation.nlseeker.config import load_config, parse_layout
from evaluation.nlseeker.datasets import DATASETS, FAMILY_FIELDS, lake_source, load_questions, subset_order
from evaluation.nlseeker.prepare import CACHE_DIR, DATASETS_DIR, prepare_dataset, sha256_file
from evaluation.nlseeker.report import overlap
from evaluation.nlseeker.rundir import seal_index, verify_index, write_ini
from evaluation.nlseeker.runner import document_counts

from evaluation.nlseeker.build import Texts
from evaluation.nlseeker.config import EvalConfig
from evaluation.nlseeker.prepare import PreparedDataset
from src.NLSeeker.Clients import EmbeddingClient, LLMClient
from src.NLSeeker.Config import NLSeekerConfig
from typing import Callable, Dict, List, Optional, Tuple

BACKENDS = ('blend', 'pneuma')
ART_MARGIN_GB = 1.0


@dataclass(frozen=True)
class Lake:
    name: str
    database: Path
    index_table: str

    @classmethod
    def load(cls, name: str) -> 'Lake':
        ini = cls.ini_of(name)
        if not ini.is_file():
            raise FileNotFoundError(f'{ini} is missing: every lake is switched to by its ini')
        config = configparser.ConfigParser()
        config.read(ini)
        return cls(name, Path(config['Database']['path']), config['Database']['index_table'])

    @staticmethod
    def ini_of(name: str) -> Path:
        return Path('config') / f'{name}.ini'

    @property
    def ini(self) -> Path:
        return self.ini_of(self.name)

    @property
    def directory(self) -> Path:
        return self.database.with_suffix('')

    @property
    def manifest(self) -> Path:
        from benchmarks.nl_integration.common import index_manifest
        return index_manifest(self.name)

    def record(self, backend: str) -> Path:
        return self.directory / f'build_{backend}.json'

    def texts(self, backend: str) -> Path:
        return self.directory / f'texts_{backend}.json'


def prepare_lake(cfg: EvalConfig, lake: Lake) -> Tuple[PreparedDataset, Optional[Texts], dict]:
    dataset, size = lake_source(lake.name)
    if dataset not in cfg.datasets:
        raise ValueError(f'{lake.name} is built from {dataset}, which the config does not list')
    prepared = prepare_dataset(DATASETS[dataset], DATASETS_DIR, CACHE_DIR)
    if size is None:
        return prepared, None, {}
    if size not in cfg.subsets:
        raise ValueError(f'{lake.name}: {size} tables is not one of the config\'s subsets {cfg.subsets}')
    questions = [question for family in cfg.families for question in load_questions(
        prepared.questions_jsonl if FAMILY_FIELDS[family][0] == 'bc' else prepared.bx_jsonl, family,
        cfg.questions_limit)]
    files = subset_order(sorted(path.name for path in prepared.lake_dir.glob('*.csv')), questions)[:size]
    if len(files) < size:
        raise ValueError(f'{dataset} has {len(files)} tables, fewer than the subset of {size}')
    linked = CACHE_DIR / 'lakes' / lake.name
    if not linked.is_dir() or sorted(path.name for path in linked.iterdir()) != sorted(files):
        shutil.rmtree(linked, ignore_errors=True)
        linked.mkdir(parents=True)
        for file in files:
            (linked / file).symlink_to(prepared.lake_dir.resolve() / file)
    source = Lake.load(DATASETS[dataset].lake).texts('blend')
    if not source.is_file():
        raise FileNotFoundError(f'{lake.name} indexes the texts of Blend\'s build of {dataset}: {source} is missing')
    whole = json.loads(source.read_text())
    return replace(prepared, lake_dir=linked), {file: whole[file] for file in files}, {
        'tables': size, 'tables_sha256': hashlib.sha256('\n'.join(files).encode()).hexdigest(),
        'shared_texts': _source(source)}


def build_blend_lake(cfg: EvalConfig, lake: Lake, prepared: PreparedDataset, clients: Optional[Callable] = None,
                     imported: Optional[Texts] = None, entry: Optional[dict] = None, concurrent: bool = False,
                     **meta) -> dict:
    from evaluation.nlseeker.build import build_blend_database
    from evaluation.nlseeker.storage import blend_storage
    from src.NLSeeker import Schema

    _refuse_sealed(lake)
    leftovers = [lake.database] * lake.database.exists() + [
        path for top in lake.directory.glob('*') if 'pneuma' not in top.name
        for path in [top, *top.rglob('*')] if path.is_file()]
    if leftovers:
        raise FileExistsError(f'{lake.name} holds files of an earlier build ({leftovers[0]}): a lake is built again '
                              f'only on purpose; delete {lake.database} and every entry of {lake.directory}/ without '
                              f'"pneuma" in its name first')
    lake.directory.mkdir(parents=True, exist_ok=True)
    config = _config(cfg, lake, 'blend')
    started = time.time()
    stages, inverted_index_s = build_blend_database(lake.database, config, prepared, lake.index_table,
                                                    *(clients or _clients(cfg))(config),
                                                    lake.directory / 'metadata.csv', imported)
    with duckdb.connect(str(lake.database), read_only=True) as connection:
        files = dict(connection.execute(f'SELECT tableid, filename FROM "{lake.index_table}_tables"').fetchall())
        texts = export_texts(connection, config.schema, {str(table_id): file for table_id, file in files.items()})
        documents = [row[0] for row in connection.execute(f'SELECT docid FROM "{config.documents_table}"').fetchall()]
    _check_narrated(document_counts(documents, lambda document: files[Schema.parse_table_id(document)]), prepared,
                    'blend')
    lake.texts('blend').write_text(json.dumps(texts))
    storage = blend_storage(lake.database, lake.index_table, config, lake.directory, ART_MARGIN_GB)
    return _write_record(cfg, lake, 'blend', prepared, {
        **(entry or {}), 'build': {'stages': stages, 'storage': storage, 'inverted_index_s': inverted_index_s,
                                   'window': [started, time.time()], 'concurrent': concurrent,
                                   'narration_concurrency': _narration_concurrency(cfg, imported),
                                   'embedding_concurrency': _embedding_concurrency(cfg)}}, meta)


def build_pneuma_lake(cfg: EvalConfig, lake: Lake, prepared: PreparedDataset, clients: Optional[Callable] = None,
                      imported: Optional[Texts] = None, entry: Optional[dict] = None, part: str = 'both',
                      concurrent: bool = False, **meta) -> dict:
    from evaluation.nlseeker.pneuma_backend import SharedEmbedder, build_pneuma, pneuma_backend
    from evaluation.nlseeker.storage import pneuma_storage

    _refuse_sealed(lake)
    lake.directory.mkdir(parents=True, exist_ok=True)
    config = _config(cfg, lake, 'pneuma')

    def pneuma_clients():
        llm, embedder = (clients or _clients(cfg))(config)
        return llm, SharedEmbedder(embedder, config.embedding_tokenizer)

    def build(out: Path, texts: Optional[Texts]) -> Tuple[Texts, dict]:
        started = time.time()
        indexed, stages = build_pneuma(out, prepared, *pneuma_clients(), texts=texts)
        backend = pneuma_backend(out, *pneuma_clients(), prepared.lake_dir)
        _check_narrated(document_counts(backend.documents, backend.table_file), prepared, out.name)
        return indexed, {'stages': stages, 'storage': pneuma_storage(out), 'window': [started, time.time()],
                         'embedding_concurrency': _embedding_concurrency(cfg)}

    if part != 'shared':
        if part == 'own' and lake.record('pneuma').is_file():
            raise FileExistsError(f'{lake.name}: Pneuma\'s own build is done ({lake.record("pneuma")}); a lake is '
                                  f'built again only on purpose')
        indexed, own = build(lake.directory / 'pneuma', imported)
        lake.texts('pneuma').write_text(json.dumps(indexed))
        pending = {'pending': 'shared'} if imported is None and cfg.shared_combos() else {}
        record = _write_record(cfg, lake, 'pneuma', prepared,
                               {**(entry or {}), 'build': {**own, 'concurrent': concurrent, 'narration_concurrency':
                                                           _narration_concurrency(cfg, imported)}},
                               {**meta, **pending})
    elif not lake.record('pneuma').is_file():
        raise FileNotFoundError(f'{lake.name}: Pneuma\'s index on Blend\'s texts follows its own build, which is '
                                f'missing')
    else:
        record = json.loads(lake.record('pneuma').read_text())
    if part == 'own' or 'pending' not in record:
        if part == 'shared':
            print(f'{lake.name}: Pneuma\'s index on Blend\'s texts: nothing to build (done, or the config has none)')
        return record
    source = lake.texts('blend')
    if not lake.record('blend').is_file():
        raise FileNotFoundError(f'{lake.name}: Pneuma indexes the texts of Blend\'s build, so Blend builds first')
    check_sources(lake, json.loads(lake.record('blend').read_text())['entry'], prepared)
    _, record['entry']['shared_build'] = build(lake.directory / 'pneuma_shared', json.loads(source.read_text()))
    record['entry']['shared_texts'] = _source(source)
    del record['pending']
    lake.record('pneuma').write_text(json.dumps(record, indent=1))
    return record


def seal(lake: Lake) -> str:
    _refuse_sealed(lake)
    for backend in BACKENDS:
        if not lake.record(backend).is_file():
            raise FileNotFoundError(f'{lake.name}: no {backend} build; both backends build before the lake is sealed')
    records = {backend: json.loads(lake.record(backend).read_text()) for backend in BACKENDS}
    if 'pending' in records['pneuma']:
        raise FileNotFoundError(f'{lake.name}: Pneuma\'s index on Blend\'s texts is not built yet (--part shared)')
    builds = [record['entry']['build'] for record in records.values()]
    if any(build.get('concurrent') for build in builds) and all('window' in build for build in builds):
        share = overlap(*(build['window'] for build in builds))
        for backend, record in records.items():
            record['entry']['build']['overlap'] = share
            lake.record(backend).write_text(json.dumps(record, indent=1))
    files = [lake.database] + sorted(path for path in lake.directory.rglob('*') if path.is_file())
    return seal_index(lake.manifest, files)


def load_built(name: str, backend: str) -> Tuple[Lake, dict, str]:
    lake = Lake.load(name)
    problems, code = verify_index(lake.manifest)
    if problems:
        raise RuntimeError(f'{name} is not a sealed build: {"; ".join(problems)}')
    return lake, json.loads(lake.record(backend).read_text()), code


def check_sources(lake: Lake, entry: dict, prepared: PreparedDataset) -> None:
    if any(entry[key] != value for key, value in _hashes(prepared).items()):
        raise ValueError(f'{lake.name} was built from other sources than the dataset has now')


def build_config(cfg: EvalConfig, llm_concurrency: Optional[int], embedding_concurrency: Optional[int]) -> EvalConfig:
    given = {'llm_concurrency': llm_concurrency, 'embedding_concurrency': embedding_concurrency}
    return replace(cfg, nlseeker={**cfg.nlseeker, **{key: str(value) for key, value in given.items()
                                                     if value is not None}})


def _narration_concurrency(cfg: EvalConfig, imported: Optional[Texts]) -> Optional[int]:
    return None if imported is not None else int(cfg.nlseeker['llm_concurrency'])


def _embedding_concurrency(cfg: EvalConfig) -> int:
    return int(cfg.nlseeker.get('embedding_concurrency', 1))


def _refuse_sealed(lake: Lake) -> None:
    if lake.manifest.is_file():
        raise FileExistsError(f'{lake.name} is sealed ({lake.manifest}): a lake is built again only on purpose, as a '
                              f'new manifest code; delete its index first')


def _config(cfg: EvalConfig, lake: Lake, backend: str) -> NLSeekerConfig:
    named = NLSeekerConfig.load(lake.ini).embedding_model
    if named != cfg.nlseeker['embedding_model']:
        raise ValueError(f'{lake.ini} embeds with {named}, the config with {cfg.nlseeker["embedding_model"]}')
    return NLSeekerConfig.load(write_ini(lake.ini, cfg.nlseeker_for(backend), cfg.alpha[0], cfg.n[0],
                                         lake.directory / f'{backend}.ini'))


def _clients(cfg: EvalConfig) -> Callable[[NLSeekerConfig], Tuple['LLMClient', 'EmbeddingClient']]:
    from src.NLSeeker.Clients import EmbeddingClient, LLMClient
    return lambda config: (LLMClient(config), EmbeddingClient(config, _embedding_concurrency(cfg)))


def _source(path: Path) -> dict:
    return {'source': str(path), 'sha256': sha256_file(path)}


def _check_narrated(counts: Dict[str, Dict[str, int]], prepared: PreparedDataset, build: str) -> None:
    lost = sorted(path.name for path in prepared.lake_dir.glob('*.csv') if 'schema' not in counts.get(path.name, {}))
    if lost:
        raise RuntimeError(f'{build} indexed no narration for {len(lost)} table(s): {lost[:5]}')


def _hashes(prepared: PreparedDataset) -> dict:
    return {'tar_sha256': prepared.tar_sha256, 'zip_sha256': prepared.zip_sha256,
            'questions_sha256': prepared.questions_sha256}


def _write_record(cfg: EvalConfig, lake: Lake, backend: str, prepared: PreparedDataset, fields: dict,
                  meta: dict) -> dict:
    from benchmarks.nl_integration.common import _git

    record = {'lake': lake.name, 'backend': backend,
              'entry': {**_hashes(prepared), **fields},
              'nlseeker': {key: value for key, value in cfg.nlseeker.items() if 'api_key' not in key},
              'built_utc': datetime.now(timezone.utc).isoformat(), 'git_commit': _git('rev-parse', 'HEAD'), **meta}
    lake.record(backend).write_text(json.dumps(record, indent=1))
    return record


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog='evaluation.nlseeker.offline')
    commands = parser.add_subparsers(dest='command', required=True)
    build = commands.add_parser('build', help="one backend's build of a lake")
    build.add_argument('lake')
    build.add_argument('--backend', choices=BACKENDS, required=True)
    build.add_argument('--part', choices=('own', 'shared', 'both'), default='both',
                       help="Pneuma only: its own build, its index on Blend's texts, or both in one go")
    build.add_argument('--concurrent', action='store_true',
                       help="the other backend's own build was launched beside this one")
    build.add_argument('--llm-concurrency', type=int, help="narration calls at once, in place of the config's")
    build.add_argument('--embedding-concurrency', type=int, help='embedding batches sent at once (default 1)')
    build.add_argument('--config', required=True, help='the YAML whose NLSeeker keys and endpoints both builds use')
    build.add_argument('--layout', action='append', required=True, metavar='BACKEND:CPUS:LLM_GPU,EMBEDDING_GPU')
    commands.add_parser('seal', help='seal a lake both backends built').add_argument('lake')
    args = parser.parse_args(argv)
    if args.command == 'build' and args.backend == 'blend' and args.part != 'both':
        parser.error('--part splits Pneuma\'s build only')
    lake = Lake.load(args.lake)
    if args.command == 'seal':
        print(f'{lake.name} sealed: {seal(lake)}')
        return 0

    from evaluation.nlseeker.__main__ import preflight

    cfg = build_config(load_config(args.config), args.llm_concurrency, args.embedding_concurrency)
    cfg = replace(cfg, datasets=(lake_source(lake.name)[0],))
    layout = dict(map(parse_layout, args.layout))
    endpoints = preflight(cfg, args.backend, layout, DATASETS_DIR)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(CACHE_DIR / '.prepare.lock', 'w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        prepared, imported, entry = prepare_lake(cfg, lake)
    split = {'part': args.part} if args.backend == 'pneuma' else {}
    build_one = build_blend_lake if args.backend == 'blend' else build_pneuma_lake
    record = build_one(cfg, lake, prepared, imported=imported, entry=entry, concurrent=args.concurrent, **split,
                       config=args.config, endpoints=endpoints, layout=layout,
                       condition={'parallel': args.concurrent, 'cpus': sorted(os.sched_getaffinity(0))})
    print(f'{lake.name}: {args.backend}{"" if args.part == "both" else " " + args.part} built, record '
          f'{lake.record(args.backend)}')
    print(json.dumps({key: record['entry'][key] for key in ('build', 'shared_build') if key in record['entry']},
                     indent=1))
    return 0


if __name__ == '__main__':
    sys.exit(main())
