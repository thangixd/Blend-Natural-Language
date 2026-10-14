import json
import re
import shutil
import sys
import types
from contextlib import nullcontext
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from evaluation.nlseeker.build import export_texts, import_texts
from evaluation.nlseeker.datasets import pneuma_id
from evaluation.nlseeker.runner import Backend, StageTimer, load_clients

PNEUMA = Path(__file__).resolve().parents[2] / 'third_party' / 'pneuma'
for _path in (PNEUMA / 'src', PNEUMA, PNEUMA / 'experiments'):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import bm25s  # noqa: E402
import chromadb_deterministic  # noqa: E402
import Stemmer  # noqa: E402
from tokenizers import Tokenizer  # noqa: E402

sys.modules.setdefault('chromadb', chromadb_deterministic)

from pneuma.pneuma import Pneuma  # noqa: E402
from pneuma.registrar import registrar  # noqa: E402
from pneuma.summarizer import summarizer  # noqa: E402
from pneuma_retriever import hybrid_retriever  # noqa: E402

from evaluation.nlseeker.build import Texts  # noqa: E402
from evaluation.nlseeker.prepare import PreparedDataset  # noqa: E402
from src.NLSeeker.Clients import EmbeddingClient, LLMClient  # noqa: E402
from typing import Dict, Iterable, List, Optional, Tuple  # noqa: E402

UPSTREAM = 'https://github.com/TheDataStation/pneuma @ 8fd9f68a29d6e4ef8b3a5556dbbc2c2e316d2e3a'
INDEX = 'parity'
HNSW_DEFAULTS = {'hnsw:construction_ef': 100, 'hnsw:search_ef': 10}


class SharedEmbedder:
    def __init__(self, client: EmbeddingClient, tokenizer: str) -> None:
        self.client = client
        self.tokenizer = Tokenizer.from_pretrained(tokenizer)

    def encode(self, sentences, **_device_and_batching):
        single = isinstance(sentences, str)
        vectors = np.asarray(self.client.encode([sentences] if single else list(sentences)), dtype=np.float32)
        return vectors[0] if single else vectors


def patch_endpoints(chat: LLMClient) -> None:
    def prompt(llm, conversations, *_, max_new_tokens=512, **_generation):
        answers = chat.complete(conversations, max_new_tokens)
        return [conversation + [{'role': 'assistant', 'content': answer}]
                for conversation, answer in zip(conversations, answers)]

    summarizer.prompt_openai_llm = prompt
    hybrid_retriever.prompt_pipeline = prompt


_add = chromadb_deterministic.Collection.add


def _add_in_batches(self, ids, embeddings, documents):
    size = self._client.get_max_batch_size()
    for start in range(0, len(ids), size):
        part = slice(start, start + size)
        _add(self, ids=ids[part], embeddings=embeddings[part], documents=documents[part])


chromadb_deterministic.Collection.add = _add_in_batches


def _on_one_connection(read):
    def read_on_one_connection(self, *args):
        if registrar.duckdb is not duckdb:
            return read(self, *args)
        with duckdb.connect(self.db_path) as connection:
            registrar.duckdb = types.SimpleNamespace(connect=lambda *_: nullcontext(connection))
            try:
                return read(self, *args)
            finally:
                registrar.duckdb = duckdb
    return read_on_one_connection


_read_table_folder = registrar.Registrar._Registrar__read_table_folder
_read_metadata_file = registrar.Registrar._Registrar__read_metadata_file
registrar.Registrar._Registrar__read_table_folder = _on_one_connection(_read_table_folder)
registrar.Registrar._Registrar__read_metadata_file = _on_one_connection(_read_metadata_file)


def _check(response: str, step: str) -> dict:
    result = json.loads(response)
    if result['status'] != 'SUCCESS':
        raise RuntimeError(f'Pneuma {step} failed: {result["message"]}')
    return result['data']


def _registered_id(path: str) -> str:
    return path.replace('\\', '/').replace('"', "'")


def _files_by_registered_name(files: Iterable[str]) -> Dict[str, str]:
    names = {}
    for file in files:
        name = _registered_id(file)
        if '/' in name or names.setdefault(name, file) != file:
            raise ValueError(f'Pneuma cannot register {file!r} apart from the lake\'s other files: its table id would '
                             f'be {name!r}')
    return names


def _table_name(document: str) -> str:
    return Path(re.split('_SEP_(?:contents_SEP_|contexts-)', document)[0]).name


def build_pneuma(out: Path, prepared: PreparedDataset, llm: LLMClient, embedder: SharedEmbedder,
                 texts: Optional[Texts] = None) -> Tuple[Texts, Dict[str, dict]]:
    patch_endpoints(llm)
    stage = StageTimer(embedder.client, llm)
    if out.exists():
        shutil.rmtree(out)
    lake = str(prepared.lake_dir.resolve())
    files = sorted(path.name for path in prepared.lake_dir.glob('*.csv'))
    _files_by_registered_name(files)
    pneuma = Pneuma(out_path=str(out), use_local_model=False, openai_api_key='unused')
    pneuma.embed_model = embedder

    table_ids = {name: _registered_id(f'{lake}/{name}') for name in files}
    with stage('register'):
        _check(pneuma.setup(), 'setup')
        _check(pneuma.add_tables(lake, creator='parity', accept_duplicates=True), 'add_tables')
        with duckdb.connect(pneuma.db_path, read_only=True) as connection:
            registered = sorted(row[0] for row in connection.sql('SELECT id FROM table_status').fetchall())
        if registered != sorted(table_ids.values()):
            raise RuntimeError(f'Pneuma registered {len(registered)} of {len(files)} tables; ids unexpected / missing: '
                               f'{sorted(set(registered) - set(table_ids.values()))[:3]} / '
                               f'{sorted(set(table_ids.values()) - set(registered))[:3]}')
        if texts is None:
            by_name = {pneuma_id(name): table_id for name, table_id in table_ids.items()}
            with open(prepared.contexts_jsonl) as f:
                contexts = [json.loads(line) for line in f]
            metadata = out / 'metadata.csv'
            pd.DataFrame([(by_name[c['table']], c['context']) for c in contexts],
                         columns=['table_id', 'value']).to_csv(metadata, index=False)
            added = _check(pneuma.add_metadata(str(metadata)), 'add_metadata')['metadata_ids']
            if len(added) != len(contexts):
                raise RuntimeError(f'Pneuma stored {len(added)} of {len(contexts)} contexts')
    if texts is None:
        with stage('narrate'):
            _check(pneuma.summarize(), 'summarize')
    else:
        with stage('import'), duckdb.connect(pneuma.db_path) as connection:
            import_texts(connection, 'main', table_ids, texts)

    with duckdb.connect(pneuma.db_path, read_only=True) as connection:
        statuses = connection.sql('SELECT status, count(*) FROM table_status GROUP BY status').fetchall()
        indexed = export_texts(connection, 'main', {table_id: name for name, table_id in table_ids.items()})
    if statuses != [('TableStatus.SUMMARIZED', len(files))]:
        raise RuntimeError(f'Pneuma summarized {statuses}, expected all {len(files)} tables')
    if texts is not None and indexed != texts:
        raise RuntimeError('Pneuma\'s bookkeeping holds other texts than the ones it was given')
    with stage('index'):
        _check(pneuma.generate_index(INDEX), 'generate_index')
    return indexed, stage.stages


def pneuma_backend(out: Path, llm: LLMClient, embedder: SharedEmbedder, lake_dir: Path) -> Backend:
    files = _files_by_registered_name(path.name for path in lake_dir.glob('*.csv'))
    patch_endpoints(llm)
    load_clients(embedder.client, llm)
    collection = chromadb_deterministic.PersistentClient(str(out / 'indexes' / 'vector')).get_collection(INDEX)
    retriever = bm25s.BM25.load(str(out / 'indexes' / 'fulltext' / INDEX), load_corpus=True)
    documents = collection.get(include=[])['ids']
    collection.query(query_embeddings=collection.get(limit=1, include=['embeddings'])['embeddings'], n_results=1)
    if sorted(documents) != sorted(datum['metadata']['table'] for datum in retriever.corpus):
        raise RuntimeError('Pneuma\'s vector and full-text indexes hold different documents')

    stemmer = Stemmer.Stemmer('english')
    dictionary_id_bm25 = {datum['metadata']['table']: i for i, datum in enumerate(retriever.corpus)}
    retrievers = {judge: hybrid_retriever.HybridRetriever(
        None, hybrid_retriever.RerankingMode.LLM if judge else hybrid_retriever.RerankingMode.NONE)
        for judge in (True, False)}

    def retrieve(question: str, k: int, n: int, alpha: float, judge: bool) -> List[str]:
        pool = k * n
        embedding = embedder.encode(question).tolist()
        tokens = bm25s.tokenize(question, stopwords='en', stemmer=stemmer, show_progress=False)
        vectors = collection.query(query_embeddings=[embedding], n_results=pool)
        bm25 = retriever.retrieve(tokens, k=pool, show_progress=False)
        nodes = retrievers[judge].retrieve(retriever, collection, bm25, vectors, pool, question, alpha,
                                           tokens, embedding, dictionary_id_bm25)
        return [node[0] for node in nodes]

    provenance = {
        'upstream': UPSTREAM,
        'patches': sorted(path.name for path in (PNEUMA / 'patches').glob('*.patch'))
        + ['endpoint, Chroma batch, registrar and metadata connection patches: evaluation/nlseeker/pneuma_backend.py'],
        'registration': 'one DuckDB connection per table folder and per metadata file, not per table and per context '
                        '(runtime patch): the same storage.db contents, but not upstream\'s registration and metadata '
                        'times or checkpoints (file size)',
        'hnsw': {**HNSW_DEFAULTS, **collection.metadata},
        'row_sampling': 'pandas DataFrame.sample(n=min(rows, 5), random_state=0)',
        'query_tokens': "bm25s.tokenize(question, stopwords='en', stemmer=english)",
    }
    return Backend('pneuma', retrieve, lambda document: files[_table_name(document)], embedder.client, llm,
                   documents=documents, provenance=provenance)
