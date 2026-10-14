import argparse
import multiprocessing
import os
from collections import defaultdict, deque
from concurrent.futures import ProcessPoolExecutor
from glob import glob
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import numpy as np
import pyarrow as pa
import duckdb
from src.DBHandler import DBHandler
from src.utils import calculate_xash


BATCH_SIZE = 1_000_000
USABLE_CORES = len(os.sched_getaffinity(0))


def create_table(dbcon, table_name):
    print('Preparing index creation by creating tables in DB.')
    dbcon.execute(f'DROP TABLE IF EXISTS "{table_name}";')
    dbcon.execute(f'DROP TABLE IF EXISTS "{table_name}_tables";')

    dbcon.execute(f"""CREATE TABLE "{table_name}" (
        tokenized VARCHAR,
        tableid INTEGER,
        colid INTEGER,
        rowid INTEGER,
        super_key VARCHAR,
        quadrant BOOLEAN
        );""")

    dbcon.execute(f"""CREATE TABLE "{table_name}_tables" (
        tableid INTEGER,
        filename VARCHAR
        );""")

    print('Preparation of index creation is finished!')


def create_indexes(dbcon, table_name):
    print('Creating indexes.')
    dbcon.execute(f'CREATE INDEX "{table_name}_to_tokenized" ON "{table_name}" (tokenized);')
    dbcon.execute(f'CREATE INDEX "{table_name}_to_tableid" ON "{table_name}" (tableid);')
    print('Creation of indexes is finished!')


def save_data_to_duckdb(dbcon, batch, table_name):
    dbcon.register('batch_rows', batch)
    try:
        dbcon.execute(f'INSERT INTO "{table_name}" SELECT * FROM batch_rows')
    finally:
        dbcon.unregister('batch_rows')


def assign_table_ids(lake_path):
    file_paths = sorted(glob(f'{lake_path}'))
    if not file_paths:
        raise FileNotFoundError(f'No files matched {lake_path}')

    return list(enumerate(file_paths))


def table_rows(table_id, file_path, sep):
    file_content_df = pd.read_csv(file_path, sep=sep, low_memory=False)
    numeric_cols = file_content_df.select_dtypes(include='number').columns
    numeric_cols = [file_content_df.columns.get_loc(col) for col in numeric_cols]

    file_content = file_content_df.values
    number_of_rows = file_content.shape[0]
    number_of_cols = file_content.shape[1]

    superkeys = defaultdict(int)
    tokens = []
    quadrants = []
    for col_counter in range(number_of_cols):
        is_numeric_col = col_counter in numeric_cols
        if is_numeric_col:
            mean = np.nanmean(file_content[:, col_counter])
        for row_counter in range(number_of_rows):
            tokenized = DBHandler.normalize_token(file_content[row_counter][col_counter])
            tokens.append(tokenized or None)
            quadrants.append(file_content[row_counter][col_counter] >= mean if is_numeric_col else None)
            superkeys[row_counter] = superkeys[row_counter] | calculate_xash(str(tokenized))

    row_ids = np.tile(np.arange(number_of_rows, dtype=np.int32), number_of_cols)
    keys = pa.array([f"{superkeys[row]:0128b}" for row in range(number_of_rows)], pa.string())
    return pa.table({
        'tokenized': pa.array(tokens, pa.string()),
        'tableid': np.full(len(tokens), table_id, dtype=np.int32),
        'colid': np.repeat(np.arange(number_of_cols, dtype=np.int32), number_of_rows),
        'rowid': row_ids,
        'super_key': pa.DictionaryArray.from_arrays(row_ids, keys),
        'quadrant': pa.array(quadrants, pa.bool_()),
    })


def in_order(function, items, workers):
    if workers <= 1:
        yield from (function(*item) for item in items)
        return
    with ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context('spawn')) as pool:
        pending = deque()
        for item in items:
            pending.append(pool.submit(function, *item))
            if len(pending) >= 2 * workers:
                yield pending.popleft().result()
        while pending:
            yield pending.popleft().result()


def create_index(dbcon, assignment, table_name, sep=',', batch_size=BATCH_SIZE, workers=1):
    create_table(dbcon, table_name)

    print(f'Inserting data into index from {len(assignment)} files.')
    dbcon.execute(f'INSERT INTO "{table_name}_tables" SELECT unnest(?::INTEGER[]), unnest(?::VARCHAR[])',
                  [[table_id for table_id, _ in assignment], [Path(path).name for _, path in assignment]])

    batch, batch_rows = [], 0
    for rows in tqdm(in_order(table_rows, [(table_id, path, sep) for table_id, path in assignment],
                              min(workers, len(assignment))),
                     total=len(assignment)):
        batch.append(rows)
        batch_rows += len(rows)
        if batch_rows >= batch_size:
            save_data_to_duckdb(dbcon, pa.concat_tables(batch), table_name)
            batch, batch_rows = [], 0

    if batch:
        save_data_to_duckdb(dbcon, pa.concat_tables(batch), table_name)

    print('Insertion of the index is Finished!')
    create_indexes(dbcon, table_name)


def parse_args():
    repo_root = Path(__file__).parent.parent

    parser = argparse.ArgumentParser(description='Build a BLEND index in DuckDB from a CSV data lake.')
    parser.add_argument('--lake', default='adventure_works',
                        help='Lake directory inside --datalake, also used as the index table name.')
    parser.add_argument('--datalake', default=str(repo_root / 'datalake'), help='Directory holding the lakes.')
    parser.add_argument('--db', default=str(repo_root / 'blend_duckdb.db'), help='DuckDB database file to write.')
    parser.add_argument('--sep', default=',', help='CSV separator.')
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE,
                        help='Index rows to buffer before writing to DuckDB.')
    parser.add_argument('--workers', type=int, default=USABLE_CORES,
                        help='Lake files tokenized at once, each in its own process (default: the usable cores).')

    return parser.parse_args()


def main():
    args = parse_args()

    lake_dir = Path(args.datalake) / args.lake
    if not lake_dir.is_dir():
        raise NotADirectoryError(f'Lake directory not found: {lake_dir}')

    dbcon = duckdb.connect(database=args.db, read_only=False)
    try:
        assignment = assign_table_ids(str(lake_dir / '*.csv'))
        create_index(dbcon, assignment, args.lake, sep=args.sep, batch_size=args.batch_size,
                     workers=args.workers)
    finally:
        dbcon.close()

    print(f'\nIndex written to {args.db}. To query it, set in config/config.ini:')
    print(f'  dbms=duckdb\n  path={args.db}\n  index_table={args.lake}')


if __name__ == '__main__':
    main()
