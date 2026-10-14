import json
import time
from collections import Counter, defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

from evaluation.nlseeker.datasets import Question, pneuma_id
from evaluation.nlseeker.metrics import hit, recall, reciprocal_rank

from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Tuple


TEXTS = 'texts.json'
SHARED_DOCUMENTS = 'documents_shared.json'


@dataclass
class Backend:
    name: str
    retrieve: Callable[[str, int, int, float, bool], List[str]]
    table_file: Callable[[str], str]
    embedder: Any
    llm: Any
    after: Callable[..., Any] = lambda **fields: None
    documents: Sequence[str] = ()
    provenance: dict = field(default_factory=dict)
    build: dict = field(default_factory=dict)


def document_type(document: str) -> str:
    if '_SEP_contexts-' in document:
        return 'contexts'
    return document.rsplit('_SEP_contents_SEP_', 1)[1].split('-')[0]


def document_counts(documents: Sequence[str], table_file: Callable[[str], str]) -> Dict[str, Dict[str, int]]:
    counts = defaultdict(Counter)
    for document in documents:
        counts[table_file(document)][document_type(document)] += 1
    return {file: dict(types) for file, types in sorted(counts.items())}


def write_documents(backend: Backend, path: Path) -> Dict[str, Dict[str, int]]:
    counts = document_counts(backend.documents, backend.table_file)
    path.write_text(json.dumps(counts, indent=1))
    return counts


class StageTimer:
    def __init__(self, embedder: Any, llm: Any) -> None:
        from benchmarks.nl_integration.common import InstrumentedClients

        self.clients = InstrumentedClients(embedder, llm)
        self.stages: Dict[str, dict] = {}

    @contextmanager
    def __call__(self, name: str) -> Iterator[None]:
        self.clients.start()
        start = time.perf_counter()
        try:
            yield
        finally:
            window = {'seconds': time.perf_counter() - start, **self.clients.stop()}
            before = self.stages.get(name, {})
            self.stages[name] = {key: before.get(key, 0) + value for key, value in window.items()}


def blend_backend(ini_path: Path, nl_runs: Path, fake: bool = False) -> Backend:
    from benchmarks.nl_integration.common import HNSWProbe, NLMeter, nl_config, use_fake_endpoints
    from src.NLSeeker import Schema
    from src.Operators.OperatorBase import Operator
    from src.Operators.Seekers import NaturalLanguage
    from src.Operators.Seekers.NaturalLanguage import _load_retriever

    db = Operator.DB
    db.load_config(ini_path)
    if fake:
        use_fake_endpoints(db)
    config = nl_config(db)
    hnsw = HNSWProbe(db.connection, f'"{config.documents_table}"')
    meter = NLMeter(db, nl_runs, hnsw)
    retriever = _load_retriever(db.config_path, config)
    if not fake:
        load_clients(retriever.embedder, retriever.llm)
    files = dict(db.execute_and_fetchall('SELECT TableId, filename FROM AllTables_tables'))
    documents = [row[0] for row in db.execute_and_fetchall(f'SELECT docid FROM "{config.documents_table}"')]
    db.execute_and_fetchall(f'SELECT max(len(embedding)), max(length) FROM "{config.documents_table}"')
    db.execute_and_fetchall(f'SELECT max(length(token)), max(tf) FROM "{config.tokens_table}"')

    def retrieve(question: str, k: int, n: int, alpha: float, judge: bool) -> List[str]:
        NaturalLanguage(question, k, n, alpha, rerank=judge).run()
        return meter._pending[-1]['ranked_documents']

    return Backend('blend', retrieve, lambda document: files[Schema.parse_table_id(document)],
                   retriever.embedder, retriever.llm, after=lambda **fields: meter.write(arm='parity', **fields),
                   documents=documents,
                   provenance={'row_sampling': 'DuckDB USING SAMPLE reservoir(5 ROWS) REPEATABLE (0)',
                               'hnsw_probe': {'documents': hnsw.documents, 'build_s': hnsw.build_s,
                                              'index_bytes': hnsw.index_bytes, 'table_bytes': hnsw.table_bytes}})


def load_clients(embedder: Any, llm: Any) -> None:
    embedder.count_tokens('')
    llm._tokenizer.count('')
    llm._template_overhead()


def run_backend(backend: Backend, questions: Dict[str, List[Question]],
                combos: List[Tuple[int, int, float, bool]], out_jsonl: Path, condition: dict,
                shared_combos: Sequence[Tuple[int, int, float, bool]] = (), shared: Optional[Backend] = None,
                repetitions: int = 1) -> List[dict]:
    from benchmarks.nl_integration.common import InstrumentedClients

    shared = shared or backend
    clients = InstrumentedClients(backend.embedder, backend.llm)
    arms = [(combo, 'own', backend) for combo in combos] + [(combo, 'shared', shared) for combo in shared_combos]
    records = []
    passes = [(rep, family, question) for rep in range(repetitions)
              for family, family_questions in questions.items() for question in family_questions]
    with open(out_jsonl, 'a') as out:
        for rep, family, question in passes:
            for (k, n, alpha, judge), texts, system in arms:
                clients.start()
                start = time.perf_counter()
                documents = system.retrieve(question.text, k, n, alpha, judge)
                total = time.perf_counter() - start
                window = clients.stop()
                system.after(qid=question.qid, family=family)

                retrieved = list(dict.fromkeys(pneuma_id(system.table_file(d)) for d in documents))[:k]
                answers = set(question.answer_tables)
                record = {
                    'backend': backend.name, 'qid': question.qid, 'family': family, 'rep': rep,
                    'k': k, 'n': n, 'alpha': alpha, 'rerank': 'on' if judge else 'off', 'texts': texts,
                    'documents': documents, 'pool_size': len(documents),
                    'retrieved': retrieved, 'answers': sorted(answers),
                    'hit': hit(retrieved, answers), 'recall': recall(retrieved, answers),
                    'rr': reciprocal_rank(retrieved, answers),
                    **window, 'total': total, 'retrieval': total - window['embedding'] - window['judge'],
                    'condition': condition,
                }
                out.write(json.dumps(record) + '\n')
                out.flush()
                records.append(record)
    return records
