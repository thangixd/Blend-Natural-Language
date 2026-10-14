import argparse
import configparser
import json
import sys
from pathlib import Path

import duckdb

from benchmarks.nl_integration.common import LAKES, RESULTS, ROOT

CASTS = {'float32': 'embedding',
         'int8': 'list_transform(embedding, x -> CAST(round(x * 127) AS TINYINT))::TINYINT[768]'}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--lake', choices=list(LAKES), required=True)
    parser.add_argument('--scratch', type=Path, required=True, help='where the two temporary files go')
    parser.add_argument('--out', type=Path, default=RESULTS / 'analysis')
    args = parser.parse_args(argv)
    config = configparser.ConfigParser()
    config.read(ROOT / LAKES[args.lake][0])
    source, index = config['Database']['path'], config['NLSeeker']['index_name']
    args.scratch.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, expression in CASTS.items():
        path = args.scratch / f'{args.lake}_{name}.db'
        path.unlink(missing_ok=True)
        with duckdb.connect(str(path)) as connection:
            connection.execute(f"ATTACH '{source}' AS source (READ_ONLY)")
            connection.execute(f'CREATE TABLE e AS SELECT {expression} AS embedding '
                               f'FROM source."{index}_nl_documents"')
            connection.execute('DETACH source')
            connection.execute('CHECKPOINT')
            blocks = connection.execute("SELECT count(DISTINCT block_id) FROM pragma_storage_info('e') "
                                        "WHERE segment_type <> 'VALIDITY'").fetchone()[0]
            size = connection.execute('SELECT block_size FROM pragma_database_size()').fetchone()[0]
        path.unlink()
        rows.append({'type': name, 'blocks': blocks, 'block_size': size, 'mib': round(blocks * size / 2 ** 20, 2)})
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / f'int8_{args.lake}.json').write_text(json.dumps(rows, indent=1))
    print(json.dumps(rows, indent=1))
    return 0


if __name__ == '__main__':
    sys.exit(main())
