import json

import pandas as pd

from src.NLSeeker import Schema, Tokenizer

from duckdb import DuckDBPyConnection
from src.NLSeeker.Clients import EmbeddingClient
from typing import Dict, Iterable, List, Optional, Tuple


EMBEDDING_CHUNK_SIZE = 2000


class IndexGenerator:
    def __init__(self, connection: DuckDBPyConnection, schema: str, embedder: EmbeddingClient,
                 documents_table: str, tokens_table: str) -> None:
        self.connection = connection
        self.embedder = embedder
        self.documents_table = documents_table
        self.tokens_table = tokens_table
        Schema.use_schema(connection, schema)

    def generate_index(self, index_name: str, table_ids: Optional[Iterable[int]] = None,
                       replace: bool = False) -> int:
        if replace:
            self._drop_index(index_name)

        if table_ids is None:
            rows = self.connection.execute(
                'SELECT id FROM table_status ORDER BY CAST(id AS INTEGER)').fetchall()
            table_ids = [int(row[0]) for row in rows]
        else:
            table_ids = list(table_ids)

        documents = []
        missing = []
        summaries = self._payloads(
            'SELECT table_id, summary_type, summary FROM table_summaries ORDER BY id')
        contexts = self._payloads("SELECT table_id, '', context FROM table_contexts ORDER BY id")
        for table_id in table_ids:
            table_documents = self._documents(table_id, summaries.get(str(table_id), {}),
                                              contexts.get(str(table_id), {}).get('', []))
            if not table_documents:
                missing.append(table_id)
            documents.extend(table_documents)
        if missing:
            raise ValueError(f'Tables {missing} have no summaries or contexts; run the summarizer first.')
        if not documents:
            raise ValueError('No summaries or contexts to index; run the summarizer first.')

        self._write_documents(documents)
        self._write_tokens(documents)
        self._record(index_name, self.documents_table, table_ids)
        self._record(index_name, self.tokens_table, table_ids)
        return len(documents)

    def _drop_index(self, index_name: str) -> None:
        self.connection.execute(f'DROP TABLE IF EXISTS main."{self.documents_table}"')
        self.connection.execute(f'DROP TABLE IF EXISTS main."{self.tokens_table}"')

        index_ids = [row[0] for row in self.connection.execute(
            'SELECT id FROM indexes WHERE name = ?', [index_name]).fetchall()]
        for index_id in index_ids:
            self.connection.execute('DELETE FROM index_table_mappings WHERE index_id = ?', [index_id])
            self.connection.execute('DELETE FROM indexes WHERE id = ?', [index_id])

    def _documents(self, table_id: int, summaries: Dict[str, List[str]],
                   contexts: List[str]) -> List[Tuple[str, str]]:
        documents = []
        for index, text in enumerate(summaries.get(Schema.COLUMN_NARRATION, [])):
            documents.append((Schema.content_schema_id(table_id, index), text))

        for index, text in enumerate(summaries.get(Schema.ROW_SAMPLE, [])):
            documents.append((Schema.content_row_id(table_id, index), text))

        merged = Schema.block(contexts, Schema.CONTEXT_JOINER, self.embedder.content_budget,
                              self.embedder.count_tokens)
        for index, text in enumerate(merged):
            documents.append((Schema.context_id(table_id, index), text))

        return documents

    def _payloads(self, sql: str) -> Dict[str, Dict[str, List[str]]]:
        payloads = {}
        for table_id, kind, payload in self.connection.execute(sql).fetchall():
            payloads.setdefault(table_id, {}).setdefault(kind, []).append(json.loads(payload)['payload'])
        return payloads

    def _write_documents(self, documents: List[Tuple[str, str]]) -> None:
        dimension = None
        for start in range(0, len(documents), EMBEDDING_CHUNK_SIZE):
            chunk = documents[start:start + EMBEDDING_CHUNK_SIZE]
            texts = [text for _, text in chunk]
            embeddings = self.embedder.encode(texts)

            if dimension is None:
                dimension = len(embeddings[0])
                self.connection.execute(f"""CREATE TABLE main."{self.documents_table}" (
                    docid VARCHAR PRIMARY KEY,
                    tableid INTEGER NOT NULL,
                    text VARCHAR NOT NULL,
                    length INTEGER NOT NULL,
                    embedding FLOAT[{dimension}] NOT NULL
                    )""")

            frame = pd.DataFrame({
                'docid': [document_id for document_id, _ in chunk],
                'tableid': [Schema.parse_table_id(document_id) for document_id, _ in chunk],
                'text': texts,
                'length': [len(Tokenizer.tokenize(text)) for text in texts],
                'embedding': embeddings,
            })
            self._insert(f'main."{self.documents_table}"', frame,
                         f'docid, tableid, text, length, CAST(embedding AS FLOAT[{dimension}])')

        self.connection.execute(
            f'CREATE INDEX "{self.documents_table}_to_tableid" '
            f'ON main."{self.documents_table}" (tableid)')

    def _write_tokens(self, documents: List[Tuple[str, str]]) -> None:
        rows = [(document_id, token, frequency)
                for document_id, text in documents
                for token, frequency in Tokenizer.term_frequencies(text).items()]
        frame = pd.DataFrame(rows, columns=['docid', 'token', 'tf'])

        self.connection.execute(f"""CREATE TABLE main."{self.tokens_table}" (
            docid VARCHAR NOT NULL,
            token VARCHAR NOT NULL,
            tf INTEGER NOT NULL
            )""")
        self._insert(f'main."{self.tokens_table}"', frame, 'docid, token, tf')
        self.connection.execute(
            f'CREATE INDEX "{self.tokens_table}_to_token" ON main."{self.tokens_table}" (token)')

    def _insert(self, table: str, frame: pd.DataFrame, projection: str) -> None:
        self.connection.register('index_df', frame)
        try:
            self.connection.execute(f'INSERT INTO {table} SELECT {projection} FROM index_df')
        finally:
            self.connection.unregister('index_df')

    def _record(self, index_name: str, location: str, table_ids: List[int]) -> None:
        index_id = self.connection.execute(
            'INSERT INTO indexes (name, location) VALUES (?, ?) RETURNING id',
            [index_name, location],
        ).fetchone()[0]
        self.connection.execute(
            'INSERT INTO index_table_mappings (index_id, table_id) SELECT ?, unnest(?::VARCHAR[])',
            [index_id, [str(table_id) for table_id in table_ids]],
        )
