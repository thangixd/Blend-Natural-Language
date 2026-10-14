import argparse
import configparser
import hashlib
import json
import platform
import random
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import pandas as pd

from benchmarks.nl_integration.metrics import retention
from evaluation.nlseeker.datasets import pneuma_id
from evaluation.nlseeker.rundir import RunDir, manifest_code, seal_index, verify_index
from src.DBHandler import DBHandler
from src.Operators.OperatorBase import Operator

from typing import Callable, List, Optional

ROOT = Path(__file__).resolve().parents[2]
QUESTIONS_DIR = Path('PneumaData')
RESULTS = ROOT / 'data' / 'Evaluation' / 'NLIntegration'
INDEXES = ROOT / 'data' / 'indexes'

LAKES = {
    'adventure_works': ('config/adventure_works.ini', 'pneuma_adventure_works_questions_annotated.jsonl'),
    'chembl': ('config/chembl.ini', 'pneuma_chembl_10K_questions_annotated.jsonl'),
    'chicago_open': ('config/chicago_open.ini', 'pneuma_chicago_10K_questions_annotated.jsonl'),
    'public_bi': ('config/public_bi.ini', 'pneuma_public_bi_questions_annotated.jsonl'),
    'fetaqa': ('config/fetaqa.ini', 'pneuma_fetaqa_questions_annotated.jsonl'),
}


def _parser(defaults: dict) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--fake-endpoints', action='store_true',
                        help='deterministic embedder and judge instead of the inference endpoints')
    for name, value in defaults.items():
        flag = f'--{name.replace("_", "-")}'
        if isinstance(value, bool):
            parser.add_argument(flag, action='store_true', default=value)
        elif isinstance(value, list):
            parser.add_argument(flag, nargs='+', type=type(value[0]), default=value)
        else:
            parser.add_argument(flag, type=type(value), default=value)
    return parser


def run_lakes(experiment: str, run_lake: Callable[[argparse.Namespace, DBHandler], None],
              argv: Optional[List[str]] = None, **defaults) -> int:
    parser = _parser(defaults)
    parser.add_argument('--lake', nargs='+', choices=list(LAKES), default=list(LAKES))
    parser.add_argument('--results', type=Path, default=RESULTS, help='where the run directory goes')
    parser.add_argument('--resume', type=Path, help='a partial run directory to finish with its own arguments')
    parser.add_argument('--min-free-gb', type=float, default=5.0, help='free disk space required before each lake')
    argv = sys.argv[1:] if argv is None else argv
    args = parser.parse_args(argv)
    if args.resume:
        run = RunDir.open_partial(args.resume)
        args = parser.parse_args(json.loads(run.config_path.read_text())['argv'] + argv)
    else:
        run = RunDir.create(args.results, experiment, json.dumps({'argv': argv}))

    def run_one(lake: str, directory: Path) -> None:
        args.lake, args.out = lake, directory
        run_lake(args, open_lake(lake, args.fake_endpoints))

    if run.run_datasets(args.lake, run_one, int(args.min_free_gb * 2 ** 30)):
        final = run.finalize()
        print(f'sealed {final}')
        return 0
    print(f'run incomplete, resume with: --resume {run.path}', file=sys.stderr)
    return 1


def index_manifest(lake: str) -> Path:
    return INDEXES / f'{lake}.MANIFEST.json'


def seal_lake_index(lake: str) -> str:
    if index_manifest(lake).is_file():
        problems, code = verify_index(index_manifest(lake))
        if problems:
            raise RuntimeError(f'{lake} does not match {index_manifest(lake)}: {"; ".join(problems)}')
        return code
    config = configparser.ConfigParser()
    config.read(ROOT / LAKES[lake][0])
    return seal_index(index_manifest(lake), [Path(config['Database']['path'])])


def open_lake(lake: str, fake: bool = False) -> DBHandler:
    if index_manifest(lake).is_file():
        problems, _ = verify_index(index_manifest(lake))
        if problems:
            raise RuntimeError(f'{lake} index does not match {index_manifest(lake)}: {"; ".join(problems)}')
    Operator.DB.load_config(ROOT / LAKES[lake][0])
    if fake:
        use_fake_endpoints(Operator.DB)
    return Operator.DB


def use_fake_endpoints(db: DBHandler) -> None:
    documents = nl_config(db).documents_table
    dimension = db.execute_and_fetchall(f'SELECT len(embedding) FROM "{documents}" LIMIT 1')[0][0]
    retriever = _retriever(db)
    if hasattr(retriever, 'meter'):
        raise RuntimeError('fake endpoints go in before the NLMeter')

    def encode(texts):
        return [[rng.gauss(0, 1) for _ in range(dimension)] for rng in map(random.Random, texts)]

    def complete(conversations, max_new_tokens=None, desc=None):
        return ['yes' if hashlib.sha256(conversation[0]['content'].encode()).digest()[0] % 2 else 'no'
                for conversation in conversations]

    retriever.embedder.encode = encode
    retriever.llm.complete = complete


def table_ids(db: DBHandler) -> List[int]:
    return [row[0] for row in db.execute_and_fetchall('SELECT TableId FROM AllTables_tables ORDER BY TableId')]


def shape(db: DBHandler) -> dict:
    cells_per_column = [row[0] for row in db.execute_and_fetchall(
        'SELECT count(*) FROM AllTables GROUP BY TableId, ColumnId')]
    documents = db.execute_and_fetchall(f'SELECT count(*) FROM "{nl_config(db).documents_table}"')[0][0]
    columns = len(cells_per_column)
    return {'tables': db.execute_and_fetchall('SELECT count(*) FROM AllTables_tables')[0][0],
            'columns': columns, 'cells': sum(cells_per_column),
            'cells_per_column': sum(cells_per_column) / max(columns, 1),
            'median_cells_per_column': statistics.median(cells_per_column or [0]),
            'documents': documents, 'documents_per_column': documents / max(columns, 1)}


def nl_table_ids(db: DBHandler) -> List[int]:
    documents = nl_config(db).documents_table
    return [row[0] for row in db.execute_and_fetchall(f'SELECT DISTINCT tableid FROM "{documents}" ORDER BY 1')]


def nl_config(db: DBHandler):
    from src.NLSeeker.Config import NLSeekerConfig
    return NLSeekerConfig.load(db.config_path)


def questions(db: DBHandler, lake: str) -> List[dict]:
    by_pneuma_id = {pneuma_id(filename): table_id for table_id, filename in
                    db.execute_and_fetchall('SELECT TableId, filename FROM AllTables_tables')}
    rows = []
    with open(QUESTIONS_DIR / LAKES[lake][1]) as f:
        for line in f:
            row = json.loads(line)
            answer = row['meta']['table_id']
            rows.append({'qid': row['id'], 'question': row['question'], 'answer': by_pneuma_id[answer],
                         'answers': sorted({by_pneuma_id[table] for table in row.get('answer_tables', [answer])
                                            if table in by_pneuma_id}),
                         'ref_values': [str(v) for v in row['meta'].get('ref_values') or []]})
    return rows


def value_tokens(values: List[str]) -> List[str]:
    tokens = set(map(DBHandler.normalize_token, values))
    for value in values:
        try:
            number = pd.to_numeric(value)
        except ValueError:
            continue
        if isinstance(number, float):
            tokens.add(DBHandler.normalize_token(number))
    return sorted(token for token in tokens if token)


def answer_holds_values(db: DBHandler, question: dict) -> Optional[bool]:
    values = value_tokens(question['ref_values'])
    if not values:
        return None
    return bool(db.execute_and_fetchall(
        f"SELECT count(*) FROM AllTables WHERE TableId = {question['answer']} "
        f"AND CellValue IN ({db.create_sql_list_str(values)})")[0][0])


class InstrumentedClients:
    def __init__(self, embedder, llm) -> None:
        self.window: Optional[dict] = None
        encode, complete = embedder.encode, llm.complete

        def timed_encode(texts):
            start = time.perf_counter()
            try:
                return encode(texts)
            finally:
                self._add('embedding', time.perf_counter() - start, 1)

        def timed_complete(conversations, *args, **kwargs):
            start = time.perf_counter()
            try:
                return complete(conversations, *args, **kwargs)
            finally:
                self._add('judge', time.perf_counter() - start, len(conversations))

        embedder.encode = timed_encode
        llm.complete = timed_complete

    def _add(self, stage: str, seconds: float, calls: int) -> None:
        if self.window is not None:
            self.window[stage] += seconds
            self.window[f'{stage}_calls'] += calls

    def start(self) -> None:
        self.window = {'embedding': 0.0, 'judge': 0.0, 'embedding_calls': 0, 'judge_calls': 0}

    def stop(self) -> dict:
        window, self.window = self.window, None
        return window


class NLMeter:
    def __init__(self, db: DBHandler, path: Path, hnsw: Optional['HNSWProbe'] = None) -> None:
        self.db = db
        self.path = path
        self.hnsw = hnsw
        self.config = nl_config(db)
        self._pending: List[dict] = []
        self._run: Optional[dict] = None
        retriever = _retriever(db)
        if hasattr(retriever, 'meter'):
            raise RuntimeError('this retriever already has an NLMeter')
        retriever.meter = self
        self.clients = InstrumentedClients(retriever.embedder, retriever.llm)
        retrieve, rerank = retriever.retrieve, retriever._rerank
        meter = self

        class CapturingDB:
            def __init__(self, inner):
                self._inner = inner

            def execute_and_fetchall(self, query, parameters=None):
                rows = self._inner.execute_and_fetchall(query, parameters)
                meter._run.update(pooled_documents=len(rows), ranked_documents=[row[0] for row in rows],
                                  query_embedding=list(parameters[0]), pooled_rows=rows)
                return rows

            def __getattr__(self, name):
                return getattr(self._inner, name)

        def timed_retrieve(db_, query, k, n, alpha, additionals='', rerank=True):
            self._run = {'question': query, 'k': k, 'n': n, 'alpha': alpha, 'judge_on': rerank,
                         'additionals': additionals}
            self.clients.start()
            start = time.perf_counter()
            try:
                tables = retrieve(CapturingDB(db_), query, k, n, alpha, additionals, rerank)
            finally:
                total = time.perf_counter() - start
                run, self._run = self._run, None
                window = self.clients.stop()
            run.update(window, total=total, retrieval=total - window['embedding'] - window['judge'], tables=tables)
            self._pending.append(run)
            return tables

        def ranked_rerank(query, document_ids, texts):
            ranked = rerank(query, document_ids, texts)
            self._run['ranked_documents'] = ranked
            return ranked

        retriever.retrieve = timed_retrieve
        retriever._rerank = ranked_rerank

    def write(self, arm: str, **fields) -> List[dict]:
        records, self._pending = self._pending, []
        with open(self.path, 'a') as f:
            for record in records:
                record.update(self._probe(record['question'], record['additionals'],
                                          record.pop('query_embedding'), record['k'] * record['n']),
                              pooled_characters=sum(len(text) for _, text in record.pop('pooled_rows')),
                              arm=arm, **fields)
                f.write(json.dumps(record) + '\n')
        return records

    def _probe(self, question: str, additionals: str, embedding: List[float], pooled: int) -> dict:
        from src.NLSeeker import Tokenizer
        from src.NLSeeker.Retriever import bm25_sql

        documents = f'"{self.config.documents_table}"'
        bm25 = bm25_sql(self.config.documents_table, self.config.tokens_table, Tokenizer.tokenize(question),
                        additionals)
        start = time.perf_counter()
        self.db.execute_and_fetchall(f'SELECT docid FROM ({bm25}) ORDER BY score DESC, docid LIMIT {pooled}')
        bm25_probe = time.perf_counter() - start
        start = time.perf_counter()
        exact = self.db.execute_and_fetchall(exact_scan_sql(documents, len(embedding), additionals, pooled), [embedding])
        vector_probe = time.perf_counter() - start

        matched, scanned, restriction = self.db.execute_and_fetchall(
            f'SELECT (SELECT count(*) FROM ({bm25})), count(*), count(DISTINCT tableid) '
            f'FROM {documents} WHERE 1=1 {additionals}')[0]
        probes = {'bm25_probe': bm25_probe, 'vector_probe': vector_probe, 'bm25_matched': matched,
                  'documents_scanned': scanned, 'restriction_size': restriction}
        if self.hnsw:
            probes.update(self.hnsw.search(embedding, pooled, additionals, [row[0] for row in exact]))
        return probes


def exact_scan_sql(documents: str, dimension: int, additionals: str, pooled: int) -> str:
    return (f'SELECT docid FROM (SELECT docid, array_cosine_similarity(embedding, ?::FLOAT[{dimension}]) '
            f'AS score FROM {documents} WHERE 1=1 {additionals}) ORDER BY score DESC, docid LIMIT {pooled}')


HNSW_SETTINGS = "metric = 'cosine', M = 48, ef_construction = 100, ef_search = 10"


class HNSWProbe:
    def __init__(self, source: duckdb.DuckDBPyConnection, documents: str) -> None:
        self.connection = duckdb.connect()
        self.connection.execute('INSTALL vss; LOAD vss')
        reader = source.execute(f'SELECT docid, tableid, embedding FROM {documents}').to_arrow_reader()
        self.connection.execute('CREATE TABLE documents AS SELECT * FROM reader')
        start = time.perf_counter()
        self.connection.execute(f'CREATE INDEX hnsw ON documents USING HNSW (embedding) WITH ({HNSW_SETTINGS})')
        self.build_s = time.perf_counter() - start
        self.documents, self.index_bytes = self.connection.execute(
            'SELECT count, approx_memory_usage FROM pragma_hnsw_index_info()').fetchone()
        self.table_bytes = self.connection.execute(
            "SELECT sum(memory_usage_bytes) FROM duckdb_memory() WHERE tag = 'IN_MEMORY_TABLE'").fetchone()[0]

    def search(self, embedding: List[float], pooled: int, additionals: str, exact: List[str]) -> dict:
        query = (f'SELECT docid FROM documents WHERE 1=1 {additionals} '
                 f'ORDER BY array_cosine_distance(embedding, ?::FLOAT[{len(embedding)}]) LIMIT {pooled}')
        plan = ' '.join(row[1] for row in self.connection.execute(f'EXPLAIN {query}', [embedding]).fetchall())
        start = time.perf_counter()
        found = [row[0] for row in self.connection.execute(query, [embedding]).fetchall()]
        seconds = time.perf_counter() - start
        return {'hnsw_probe': seconds, 'hnsw_index_used': 'HNSW_INDEX_SCAN' in plan,
                'hnsw_recall': retention(found, exact), 'hnsw_short': len(found) < len(exact)}

    def close(self) -> None:
        self.connection.close()


class Records:
    def __init__(self, args: argparse.Namespace, experiment: str) -> None:
        self.path = args.out / f'{experiment}_{args.lake}.jsonl'
        self._file = open(self.path, 'a')
        db = Operator.DB
        manifest = index_manifest(args.lake)
        provenance = {
            'experiment': experiment, 'started_utc': datetime.now(timezone.utc).isoformat(),
            'command': ' '.join(sys.argv), 'args': {k: str(v) for k, v in vars(args).items()},
            'git_commit': _git('rev-parse', 'HEAD'),
            'git_status_measured_code': _git('status', '--porcelain', 'src', 'benchmarks/nl_integration'),
            'lake_ini': LAKES[args.lake][0], 'db_path': str(db.connection.execute('PRAGMA database_list').fetchall()),
            'nl': {k: v for k, v in vars(nl_config(db)).items() if 'api_key' not in k},
            'index_code': manifest_code(manifest) if manifest.is_file() else None,
            'index_manifest': json.loads(manifest.read_text()) if manifest.is_file() else None,
            'python': sys.version.split()[0], 'duckdb': duckdb.__version__, 'machine': platform.platform(),
        }
        (args.out / f'provenance_{experiment}_{args.lake}.json').write_text(json.dumps(provenance, indent=2))

    def write(self, record: dict) -> None:
        self._file.write(json.dumps(record) + '\n')
        self._file.flush()

    def read(self) -> List[dict]:
        with open(self.path) as f:
            return [json.loads(line) for line in f]


def write_summary(args: argparse.Namespace, experiment: str, text: str) -> None:
    print(text)


def markdown_table(header: List[str], rows: List[list]) -> str:
    def cell(value) -> str:
        if value is None:
            return '-'
        if isinstance(value, float):
            return f'{value:,.1f}' if abs(value) >= 10 else f'{value:.3f}'
        return str(value)
    lines = ['| ' + ' | '.join(header) + ' |', '|' + '---|' * len(header)]
    lines += ['| ' + ' | '.join(cell(v) for v in row) + ' |' for row in rows]
    return '\n'.join(lines) + '\n'


def _retriever(db: DBHandler):
    from src.Operators.Seekers.NaturalLanguage import _load_retriever
    return _load_retriever(db.config_path, nl_config(db))


def _git(*args) -> Optional[str]:
    try:
        return subprocess.run(['git', *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


if __name__ == '__main__':
    if sys.argv[1:2] != ['seal-index'] or len(sys.argv) != 3:
        sys.exit('usage: python -m benchmarks.nl_integration.common seal-index <lake>')
    print(seal_lake_index(sys.argv[2]))
