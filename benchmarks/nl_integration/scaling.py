import hashlib
import random
import resource
import statistics
import sys
import time
from pathlib import Path

import duckdb
import numpy as np
import pyarrow as pa

from benchmarks.nl_integration.common import (HNSWProbe, NLMeter, Records, _retriever, exact_scan_sql,
                                              markdown_table, nl_config, questions, run_lakes, write_summary)
from evaluation.nlseeker.rundir import check_free_space
from src.DBHandler import DBHandler
from src.Operators.Seekers import NaturalLanguage

from typing import Iterator, List, Tuple

SIZES = '10000,50000,100000,250000,500000,1000000,1000000t'
FRACTIONS = (0.0001, 0.001, 0.01, 0.1, 0.5, 0.9)
RULE_TABLES = 1_000_000
RULE_SHARE = 0.20


def run_lake(args, db: DBHandler) -> None:
    records = Records(args, 'scaling')
    sample = questions(db, args.lake)
    sample = random.Random(args.seed).sample(sample, min(args.questions, len(sample)))

    hnsw = HNSWProbe(db.connection, f'"{nl_config(db).documents_table}"')
    records.write({'part': 'real lake', 'documents': hnsw.documents, 'build_s': hnsw.build_s,
                   'index_bytes': hnsw.index_bytes, 'table_bytes': hnsw.table_bytes})
    meter = NLMeter(db, args.out / f'nl_runs_scaling_{args.lake}.jsonl', hnsw)
    judge = args.lake == args.simulate_from
    NaturalLanguage('warm-up', k=1, n=1, rerank=judge).run()
    meter.write(arm='warm-up')
    runs = []
    for question in sample:
        NaturalLanguage(question['question'], k=args.k, rerank=judge).run()
        runs += meter.write(arm='real lake', qid=question['qid'])
    hnsw.close()

    if args.lake == args.simulate_from and args.sizes:
        simulate(args, db, sample, records)
    write_summary(args, 'scaling', summarize(records.read(), runs, args))


def simulate(args, db: DBHandler, sample: List[dict], records: Records) -> None:
    source = db.connection.execute(f'SELECT docid, tableid, embedding FROM "{nl_config(db).documents_table}" '
                                   'ORDER BY tableid, docid').to_arrow_table()
    table_ids = source['tableid'].to_numpy()
    dimension = source.schema.field('embedding').type.list_size
    vectors = source['embedding'].combine_chunks().flatten().to_numpy().reshape(len(table_ids), dimension)
    sizes = sorted(_size_in_documents(size, table_ids) for size in args.sizes.split(','))
    embeddings = _retriever(db).embedder.encode([question['question'] for question in sample])
    pooled = args.k * nl_config(db).n

    check_free_space(args.out, sizes[-1] * dimension * 4 + int(args.min_free_gb * 2 ** 30))
    path = args.out / 'simulated.db'
    connection = duckdb.connect(str(path))
    try:
        connection.execute(f'CREATE TABLE documents (docid VARCHAR, tableid INTEGER, embedding FLOAT[{dimension}])')
        connection.execute('CREATE INDEX documents_to_tableid ON documents (tableid)')
        copies = _copies(source['docid'].to_pylist(), table_ids, vectors, args.seed, args.noise)
        pending = ([], table_ids[:0], vectors[:0])
        table_digest, vector_digest, written = hashlib.sha256(), hashlib.sha256(), 0
        for size in sizes:
            while written < size:
                if not len(pending[1]):
                    pending = next(copies)
                take = size - written
                chunk = tuple(part[:take] for part in pending)
                pending = tuple(part[take:] for part in pending)
                docids, tableids, embedding = chunk
                table_digest.update(tableids.astype('<i4').tobytes())
                vector_digest.update(embedding.astype('<f4').tobytes())
                batch = pa.table({'docid': docids, 'tableid': tableids.astype(np.int32),
                                  'embedding': pa.FixedSizeListArray.from_arrays(embedding.ravel(), dimension)})
                connection.execute('INSERT INTO documents SELECT * FROM batch')
                written += len(tableids)
            connection.execute('CHECKPOINT')
            checksum = hashlib.sha256((table_digest.hexdigest() + vector_digest.hexdigest()).encode()).hexdigest()
            _measure(args, connection, size, checksum, sample, embeddings, pooled, records)
    finally:
        connection.close()
        for leftover in (path, Path(f'{path}.wal')):
            leftover.unlink(missing_ok=True)


def _measure(args, connection: duckdb.DuckDBPyConnection, size: int, checksum: str, sample: List[dict],
             embeddings: List[List[float]], pooled: int, records: Records) -> None:
    hnsw = HNSWProbe(connection, 'documents')
    tables = [row[0] for row in connection.execute('SELECT DISTINCT tableid FROM documents ORDER BY 1').fetchall()]
    records.write({'part': 'simulated size', 'documents': size, 'tables': len(tables), 'build_s': hnsw.build_s,
                   'index_bytes': hnsw.index_bytes, 'table_bytes': hnsw.table_bytes,
                   'max_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                   'seed': args.seed, 'noise': args.noise, 'checksum': checksum})
    restrictions = [(None, len(tables), '')]
    for fraction in FRACTIONS:
        allowed = random.Random(f'{args.seed}-{size}-{fraction}').sample(tables, max(1, round(fraction * len(tables))))
        restrictions.append((fraction, len(allowed), f' AND TableId IN ({DBHandler.create_sql_list_numeric(allowed)}) '))

    def exact_scan(embedding, additionals):
        return [row[0] for row in connection.execute(
            exact_scan_sql('documents', len(embedding), additionals, pooled), [embedding]).fetchall()]

    exact_plans = {fraction: 'INDEX_SCAN' in ' '.join(row[1] for row in connection.execute(
        f'EXPLAIN {exact_scan_sql("documents", len(embeddings[0]), additionals, pooled)}', [embeddings[0]]).fetchall())
        for fraction, _, additionals in restrictions}
    hnsw.search(embeddings[0], pooled, '', exact_scan(embeddings[0], ''))
    for question, embedding in zip(sample, embeddings):
        for fraction, allowed, additionals in restrictions:
            start = time.perf_counter()
            exact = exact_scan(embedding, additionals)
            exact_s = time.perf_counter() - start
            records.write({'part': 'simulated', 'documents': size, 'qid': question['qid'], 'fraction': fraction,
                           'allowed_tables': allowed, 'exact_s': exact_s, 'exact_rows': len(exact),
                           'exact_index_used': exact_plans[fraction],
                           **hnsw.search(embedding, pooled, additionals, exact)})
    hnsw.close()


def _size_in_documents(size: str, table_ids: np.ndarray) -> int:
    if not size.endswith('t'):
        return int(size)
    per_copy = np.unique(table_ids)
    copies, rest = divmod(int(size[:-1]), len(per_copy))
    return copies * len(table_ids) + int(np.isin(table_ids, per_copy[:rest]).sum())


def _copies(docids: List[str], table_ids: np.ndarray, vectors: np.ndarray, seed: int,
            noise: float) -> Iterator[Tuple[list, np.ndarray, np.ndarray]]:
    rng = np.random.default_rng(seed)
    stride = int(table_ids.max()) + 1
    suffixes = [docid[docid.index('_'):] for docid in docids]
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    copy = 1
    while True:
        jitter = rng.standard_normal(vectors.shape).astype(np.float32)
        jitter *= noise * norms / np.linalg.norm(jitter, axis=1, keepdims=True)
        copied = vectors + jitter
        copied /= np.linalg.norm(copied, axis=1, keepdims=True)
        fresh = copy * stride + table_ids
        yield [f'{table_id}{suffix}' for table_id, suffix in zip(fresh, suffixes)], fresh, copied
        copy += 1


def summarize(rows: list, runs: List[dict], args) -> str:
    def median_ms(records, field):
        low, median, high = np.percentile([record[field] * 1000 for record in records], [25, 50, 75])
        return f'{median:,.1f} [{low:,.1f}, {high:,.1f}]'

    (build,) = [row for row in rows if row['part'] == 'real lake']
    text = (f'# Scaling on {args.lake}\n\n{len(runs)} questions, k {args.k}, judge '
            f'{"on" if runs[0]["judge_on"] else "off"}, seed {args.seed}. HNSW is '
            'an in-memory vss index with Pneuma\'s settings; recall is the share of the exact scan\'s pool it '
            'finds. Times are medians [25th, 75th percentile]. NL total spans Retriever.retrieve(), query embedding '
            'and judge included; the exact scan and HNSW times cover executing and fetching their one query, '
            'outside that total.\n\n')
    text += markdown_table(
        ['Documents', 'HNSW build s', 'Index MiB', 'NL total ms', 'Judge ms', 'Exact scan ms', 'HNSW ms',
         'HNSW recall', 'Index used'],
        [[build['documents'], build['build_s'], build['index_bytes'] / 2 ** 20,
          median_ms(runs, 'total'), median_ms(runs, 'judge'), median_ms(runs, 'vector_probe'),
          median_ms(runs, 'hnsw_probe'),
          statistics.mean(run['hnsw_recall'] for run in runs),
          sum(run['hnsw_index_used'] for run in runs) / len(runs)]])

    sizes = [row for row in rows if row['part'] == 'simulated size']
    if not sizes:
        return text
    searches = [row for row in rows if row['part'] == 'simulated']
    by_qid = {run['qid']: run for run in runs}

    def cell(size, fraction=None):
        return [row for row in searches if row['documents'] == size['documents'] and row['fraction'] == fraction]

    def saving(size):
        return statistics.median((row['exact_s'] - row['hnsw_probe']) /
                                 (by_qid[row['qid']]['total'] - by_qid[row['qid']]['vector_probe'] + row['exact_s'])
                                 for row in cell(size))

    text += (f'\n## Simulated lake\n\nCopies of {args.lake}\'s embeddings, noise {args.noise} of each vector\'s '
             f'norm, seed {args.seed}; deleted after the run. The checksum covers the TableIds and vectors up to '
             'each size.\n\n')
    text += markdown_table(
        ['Documents', 'Tables', 'HNSW build s', 'Index MiB', 'In-memory copy MiB', 'Peak RSS MiB', 'Exact scan ms',
         'HNSW ms', 'HNSW recall', 'Share of NL total saved', 'Checksum'],
        [[size['documents'], size['tables'], size['build_s'], size['index_bytes'] / 2 ** 20,
          size['table_bytes'] / 2 ** 20, size['max_rss_bytes'] / 2 ** 20, median_ms(cell(size), 'exact_s'), median_ms(cell(size), 'hnsw_probe'),
          statistics.mean(row['hnsw_recall'] for row in cell(size)), saving(size), size['checksum'][:12]]
         for size in sizes])

    restricted = []
    for size in sizes:
        for fraction in FRACTIONS:
            arm = cell(size, fraction)
            restricted.append([size['documents'], f'{fraction:.2%}', arm[0]['allowed_tables'],
                               arm[0]['exact_index_used'],
                               sum(row['hnsw_index_used'] for row in arm) / len(arm),
                               sum(row['hnsw_short'] for row in arm) / len(arm),
                               statistics.mean(row['hnsw_recall'] for row in arm),
                               median_ms(arm, 'exact_s'), median_ms(arm, 'hnsw_probe')])
    text += ('\n### Under a restriction\n\nOn the indexed copy DuckDB picks HNSW or the exact scan itself; a short '
             'result has fewer documents than the exact scan, which breaks Blend\'s top-k contract.\n\n')
    text += markdown_table(['Documents', 'Fraction', 'Allowed tables', 'Exact scan uses the ART', 'HNSW chosen',
                            'Short', 'Recall',
                            'Exact scan ms', 'Indexed copy ms'], restricted)

    at_rule = [size for size in sizes if size['tables'] >= RULE_TABLES]
    size = at_rule[0] if at_rule else sizes[-1]
    share = saving(size)
    if not at_rule:
        verdict = f'not decided: the largest size has {size["tables"]:,} tables, the rule needs {RULE_TABLES:,}'
    elif share >= RULE_SHARE:
        verdict = 'implement the HNSW trigger'
    else:
        verdict = 'keep the exact scan; HNSW stays a documented upgrade path'
    recall = statistics.mean(row['hnsw_recall'] for row in cell(size))
    return text + (f'\n**20 % rule:** at {size["tables"]:,} tables HNSW saves {share:.1%} of NL\'s total latency '
                   f'(judge on, upper bound) at a recall of {recall:.2f}: {verdict}.\n')


def main(argv=None) -> int:
    return run_lakes('scaling', run_lake, argv, questions=100, k=10, sizes=SIZES, noise=0.1,
                     simulate_from='')


if __name__ == '__main__':
    sys.exit(main())
