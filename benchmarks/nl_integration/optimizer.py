import random
import sys

from benchmarks.nl_integration.common import Records, markdown_table, run_lakes, write_summary
from scripts.train_ml_optimizer import TableSampler
from src.DBHandler import DBHandler
from src.Operators.Combiners.Intersection import Intersection
from src.Operators.Seekers import Keyword, SingleColumnOverlap
from src.Optimization import changeOptimizationOrder

KINDS = ('SC+SC', 'KW+SC')


def values(df, column) -> list:
    return [value for value in df[column].dropna().tolist() if value != ''][:100]


def pair(kind: str, sampler: TableSampler, k: int):
    table_id = sampler.rng.choice(sampler.table_ids)
    df = sampler.table(table_id)
    if len(df.columns) < 2:
        return None
    first_column, second_column = sampler.rng.sample(list(df.columns), 2)
    first_values, second_values = values(df, first_column), values(df, second_column)
    if not first_values or not second_values:
        return None
    keywords = None if kind == 'SC+SC' else sampler.rng.sample(first_values, min(3, len(first_values)))
    first = SingleColumnOverlap(first_values, k) if keywords is None else Keyword(keywords, k)
    inputs = {'first_column': first_column, 'second_column': second_column, 'keywords': keywords}
    return table_id, inputs, first, SingleColumnOverlap(second_values, k)


def intersect(first, second, k: int) -> set:
    changeOptimizationOrder(first, second)
    plan = Intersection(k=k)
    plan.set_inputs([first, second])
    return set(plan.run())


def run_lake(args, db: DBHandler) -> None:
    records = Records(args, 'optimizer')
    sampler = TableSampler(db, random.Random(args.seed))
    for kind in KINDS:
        measured = attempts = 0
        while measured < args.pairs and attempts < 5 * args.pairs:
            attempts += 1
            built = pair(kind, sampler, args.k)
            if built is None:
                continue
            table_id, inputs, first, second = built
            own = set(first.run()) & set(second.run())
            records.write({'kind': kind, 'table': table_id, **inputs, 'own': sorted(own),
                           'forward': sorted(intersect(first, second, args.k)),
                           'reverse': sorted(intersect(second, first, args.k))})
            measured += 1
    write_summary(args, 'optimizer', summarize(records.read(), args))


def summarize(rows: list, args) -> str:
    text = (f'# Theorem 1 with Blend\'s seekers on {args.lake}\n\nk {args.k}, seed {args.seed}. Own: each seeker alone '
            'at k, intersected afterwards. Forward and reverse: Blend\'s Intersection with either input first, which '
            'passes the first input\'s tables into the second input\'s query. Lost: pairs where an own table is '
            'missing from forward or reverse.\n\n')
    table = []
    for kind in KINDS:
        group = [row for row in rows if row['kind'] == kind]
        if group:
            table.append([kind, len(group), sum(row['forward'] != row['own'] for row in group),
                          sum(row['reverse'] != row['own'] for row in group),
                          sum(row['forward'] != row['reverse'] for row in group),
                          sum(not set(row['own']) <= set(row['forward']) & set(row['reverse']) for row in group)])
    return text + markdown_table(['Pair', 'Pairs', 'Forward differs from own', 'Reverse differs from own',
                                  'Forward differs from reverse', 'Lost'], table)


def main(argv=None) -> int:
    return run_lakes('optimizer', run_lake, argv, pairs=100, k=10)


if __name__ == '__main__':
    sys.exit(main())
