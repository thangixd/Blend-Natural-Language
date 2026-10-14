import shutil
import sys
import tempfile
from pathlib import Path

import duckdb

from benchmarks.nl_integration.common import Records, markdown_table, nl_config, run_lakes, shape, write_summary
from evaluation.nlseeker.rundir import check_free_space
from src.DBHandler import DBHandler

from typing import Callable, Dict

OTHER = 'other (metadata, shared blocks)'


def run_lake(args, db: DBHandler) -> None:
    config = nl_config(db)
    connection = db.connection

    block_size, used_blocks = connection.execute('SELECT block_size, used_blocks FROM pragma_database_size()').fetchone()
    tables = connection.execute('SELECT schema_name, table_name, estimated_size FROM duckdb_tables()').fetchall()
    bookkeeping = {'table_status', 'table_contexts', 'table_summaries', 'indexes', 'index_table_mappings'}

    def group(schema, table):
        if schema == 'main' and table in (db.index_table, f'{db.index_table}_tables'):
            return 'inverted index'
        if schema == 'main' and table in (config.documents_table, config.tokens_table):
            return 'NL retrieval'
        if schema == config.schema:
            return 'NL bookkeeping' if table in bookkeeping else 'NL table copies'
        return 'other tables'

    rows = {}
    for schema, table, estimated_rows in tables:
        name = group(schema, table)
        rows[name] = rows.get(name, 0) + estimated_rows
    sizes = block_bytes(connection, lambda schema, table, column: 'NL embeddings'
                        if table == config.documents_table and column == 'embedding' else group(schema, table))
    art = art_bytes(Path(connection.execute(
        'SELECT path FROM duckdb_databases() WHERE database_name = current_database()').fetchone()[0]),
        args.out, args.min_free_gb)
    sizes[OTHER] = used_blocks * block_size - sum(sizes.values()) - sum(art.values())
    lake = shape(db)
    nl = sizes.get('NL retrieval', 0) + sizes.get('NL embeddings', 0)
    record = {'block_size': block_size, 'used_blocks': used_blocks, 'bytes': sizes, 'art_bytes': art, 'rows': rows,
              'cells': lake['cells'], 'columns': lake['columns'], 'documents': lake['documents'],
              'bytes_per_unit': {'inverted index per cell': sizes.get('inverted index', 0) / max(lake['cells'], 1),
                                 'ART per cell': sum(art.values()) / max(lake['cells'], 1),
                                 'NL per document': nl / max(lake['documents'], 1),
                                 'NL per column': nl / max(lake['columns'], 1)},
              'levers': {'drop the table copies': sizes.get('NL table copies', 0),
                         'int8 embeddings': sizes.get('NL embeddings', 0) * 3 / 4}}
    Records(args, 'storage').write(record)
    write_summary(args, 'storage', summarize(record, args))


def block_bytes(connection, part: Callable[[str, str, str], str]) -> Dict[str, int]:
    block_size = connection.execute('SELECT block_size FROM pragma_database_size()').fetchone()[0]
    blocks = {}
    for schema, table in connection.execute('SELECT schema_name, table_name FROM duckdb_tables()').fetchall():
        qualified = '.'.join('"' + name.replace('"', '""') + '"' for name in (schema, table))
        segments = connection.execute(
            'SELECT column_name, block_id, additional_block_ids FROM pragma_storage_info(?) WHERE block_id >= 0',
            [qualified]).fetchall()
        for column, block_id, additional in segments:
            blocks.setdefault(part(schema, table, column), set()).update([block_id, *additional])
    return {name: len(ids) * block_size for name, ids in blocks.items()}


def art_bytes(database: Path, scratch: Path, min_free_gb: float) -> dict:
    check_free_space(scratch, database.stat().st_size + int(min_free_gb * 2 ** 30))
    with tempfile.TemporaryDirectory(dir=scratch) as directory:
        copy = Path(directory) / database.name
        shutil.copyfile(database, copy)
        connection = duckdb.connect(str(copy))
        try:
            def used():
                return connection.execute('SELECT used_blocks FROM pragma_database_size()').fetchone()[0]

            block_size = connection.execute('SELECT block_size FROM pragma_database_size()').fetchone()[0]
            sizes = {}
            for schema, name in connection.execute(
                    'SELECT schema_name, index_name FROM duckdb_indexes() ORDER BY 1, 2').fetchall():
                before = used()
                connection.execute(f'DROP INDEX "{schema}"."{name}"')
                connection.execute('CHECKPOINT')
                sizes[name] = max(before - used(), 0) * block_size
            return sizes
        finally:
            connection.close()


def summarize(record: dict, args) -> str:
    sizes = record['bytes']
    inverted = sizes.get('inverted index', 0)
    total = record['used_blocks'] * record['block_size']

    def row(part, size):
        return [part, size / 2 ** 20, size / inverted if inverted else None, size / total]

    parts = [row(part, size) for part, size in sizes.items()] + [row(f'ART {name}', size)
                                                                for name, size in record['art_bytes'].items()]
    text = (f'# Storage on {args.lake}\n\n{record["used_blocks"]} used blocks of {record["block_size"]} bytes. '
            f'{record["cells"]:,} cells in {record["columns"]:,} columns, {record["documents"]:,} NL documents. '
            'NL is the retrieval tables plus the embeddings; the table copies and bookkeeping are counted '
            'apart. ART sizes are whole blocks.\n\n')
    text += markdown_table(['Part', 'MiB', 'Share of inverted index', 'Share of file'], parts)
    text += '\n' + markdown_table(['Unit', 'Bytes'], [[unit, size] for unit, size in record['bytes_per_unit'].items()])
    text += '\n' + markdown_table(['Lever', 'MiB saved', 'Share of file'],
                                  [[lever, size / 2 ** 20, size / total] for lever, size in record['levers'].items()])
    return text


def main(argv=None) -> int:
    return run_lakes('storage', run_lake, argv)


if __name__ == '__main__':
    sys.exit(main())
