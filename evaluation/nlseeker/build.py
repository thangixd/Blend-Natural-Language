import json
import time
from pathlib import Path

import duckdb
import pandas as pd

from evaluation.nlseeker.datasets import pneuma_id
from evaluation.nlseeker.prepare import PreparedDataset
from evaluation.nlseeker.runner import StageTimer
from src.NLSeeker import Schema

from src.NLSeeker.Clients import EmbeddingClient, LLMClient
from src.NLSeeker.Config import NLSeekerConfig
from typing import Dict, List, Optional, Tuple

SUMMARY_TYPES = {'schema': Schema.COLUMN_NARRATION, 'row': Schema.ROW_SAMPLE}

Texts = Dict[str, Dict[str, List[str]]]


def write_metadata_csv(contexts_jsonl: Path, assignment: List[Tuple[int, str]], out_csv: Path) -> Path:
    tableid_by_pneuma = {pneuma_id(Path(path).name): table_id for table_id, path in assignment}
    rows = []
    with open(contexts_jsonl) as f:
        for line in f:
            record = json.loads(line)
            rows.append((tableid_by_pneuma[record['table']], record['context']))
    pd.DataFrame(rows, columns=['table_id', 'value']).to_csv(out_csv, index=False)
    return out_csv


def build_blend_database(database: Path, config: NLSeekerConfig, prepared: PreparedDataset, index_table: str,
                         llm: LLMClient, embedder: EmbeddingClient, metadata_csv: Path,
                         texts: Optional[Texts] = None) -> Tuple[Dict[str, dict], float]:
    from scripts.create_index_duckdb import USABLE_CORES, assign_table_ids, create_index
    from scripts.create_index_nl_blend_duckdb import build_nl_index
    from src.NLSeeker.IndexGenerator import IndexGenerator
    from src.NLSeeker.Registrar import Registrar

    assignment = assign_table_ids(str(prepared.lake_dir / '*.csv'))
    stage = StageTimer(embedder, llm)
    dbcon = duckdb.connect(str(database))
    try:
        start = time.perf_counter()
        create_index(dbcon, assignment, index_table, workers=USABLE_CORES)
        inverted_index_s = time.perf_counter() - start
        if texts is None:
            with stage('register'):
                write_metadata_csv(prepared.contexts_jsonl, assignment, metadata_csv)
            build_nl_index(dbcon, config, assignment, str(metadata_csv), accept_duplicates=True, llm=llm,
                           embedder=embedder, stage=stage)
        else:
            with stage('register'):
                registrar = Registrar(dbcon, config.schema)
                registrar.setup()
                registrar.add_tables(assignment, creator='blend', accept_duplicates=True)
            with stage('import'):
                import_texts(dbcon, config.schema, {Path(path).name: str(table_id) for table_id, path in assignment},
                             texts)
            with stage('index'):
                IndexGenerator(dbcon, config.schema, embedder, config.documents_table,
                               config.tokens_table).generate_index(config.index_name, replace=True)
    finally:
        dbcon.close()
    return stage.stages, inverted_index_s


def export_texts(connection, schema: str, files: Dict[str, str]) -> Texts:
    texts = {file: {'schema': [], 'row': [], 'contexts': []} for file in sorted(files.values())}
    for kind, summary_type in SUMMARY_TYPES.items():
        for table_id, summary in connection.execute(
                f'SELECT table_id, summary FROM "{schema}".table_summaries WHERE summary_type = ? ORDER BY id',
                [summary_type]).fetchall():
            texts[files[table_id]][kind].append(json.loads(summary)['payload'])
    for table_id, context in connection.execute(
            f'SELECT table_id, context FROM "{schema}".table_contexts ORDER BY id').fetchall():
        texts[files[table_id]]['contexts'].append(json.loads(context)['payload'])
    return texts


def import_texts(connection, schema: str, table_ids: Dict[str, str], texts: Texts) -> None:
    if table_ids.keys() != texts.keys():
        raise ValueError(f'texts and bookkeeping differ in lake files: {sorted(table_ids.keys() ^ texts.keys())[:5]}')
    summaries = [(table_ids[file], json.dumps({'payload': text}), summary_type)
                 for file, kinds in sorted(texts.items())
                 for kind, summary_type in SUMMARY_TYPES.items() for text in kinds[kind]]
    contexts = [(table_ids[file], json.dumps({'payload': text}))
                for file, kinds in sorted(texts.items()) for text in kinds['contexts']]
    with Schema.transaction(connection):
        if summaries:
            connection.executemany(
                f'INSERT INTO "{schema}".table_summaries (table_id, summary, summary_type) VALUES (?, ?, ?)',
                summaries)
        if contexts:
            connection.executemany(f'INSERT INTO "{schema}".table_contexts (table_id, context) VALUES (?, ?)',
                                   contexts)
        connection.executemany(f'UPDATE "{schema}".table_status SET status = ? WHERE id = ?',
                               [(Schema.SUMMARIZED, table_id) for table_id in table_ids.values()])
