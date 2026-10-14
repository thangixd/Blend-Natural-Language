from pathlib import Path

import duckdb

from benchmarks.nl_integration.storage import art_bytes, block_bytes

from src.NLSeeker.Config import NLSeekerConfig

CATEGORIES = ('documents and tokens', 'vectors and their index', 'bookkeeping', 'table copies', 'other')
BOOKKEEPING = frozenset({'table_status', 'table_contexts', 'table_summaries', 'indexes', 'index_table_mappings',
                         'query_history'})
INVERTED = 'inverted index'


def _sized(parts: dict, total: int, **extra) -> dict:
    sizes = {category: parts.get(category, 0) for category in CATEGORIES[:-1]}
    sizes['other'] = total - sum(sizes.values()) - parts.get(INVERTED, 0)
    return {'bytes': sizes, INVERTED: parts.get(INVERTED, 0), 'total': total, **extra}


def blend_storage(database: Path, index_table: str, config: NLSeekerConfig, scratch: Path, min_free_gb: float) -> dict:
    def part(schema: str, table: str, column: str) -> str:
        if schema == 'main' and table in (index_table, f'{index_table}_tables'):
            return INVERTED
        if schema == 'main' and table == config.documents_table:
            return 'vectors and their index' if column == 'embedding' else 'documents and tokens'
        if schema == 'main' and table == config.tokens_table:
            return 'documents and tokens'
        if schema == config.schema:
            return 'bookkeeping' if table in BOOKKEEPING else 'table copies'
        return 'other'

    with duckdb.connect(str(database), read_only=True) as connection:
        parts = block_bytes(connection, part)
        tables = dict(connection.execute('SELECT index_name, table_name FROM duckdb_indexes()').fetchall())
    art = art_bytes(database, scratch, min_free_gb)
    for index, size in art.items():
        table = tables[index]
        category = 'vectors and their index' if table == config.documents_table else part('main', table, '')
        parts[category] = parts.get(category, 0) + size
    return _sized(parts, database.stat().st_size, art_bytes=art)


def pneuma_storage(out: Path) -> dict:
    with duckdb.connect(str(out / 'storage.db'), read_only=True) as connection:
        parts = block_bytes(connection, lambda schema, table, column:
                            'bookkeeping' if table in BOOKKEEPING else 'table copies')
    for directory, category in (('fulltext', 'documents and tokens'), ('vector', 'vectors and their index')):
        files = (out / 'indexes' / directory).rglob('*')
        parts[category] = sum(path.stat().st_size for path in files if path.is_file())
    indexes = parts['documents and tokens'] + parts['vectors and their index']
    return _sized(parts, (out / 'storage.db').stat().st_size + indexes)
