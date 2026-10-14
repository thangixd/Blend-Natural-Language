import random
import statistics
import sys
import time

from benchmarks.nl_integration.common import (NLMeter, Records, markdown_table, nl_table_ids, questions, run_lakes,
                                              write_summary)
from benchmarks.nl_integration.metrics import answer_found, distinct, divergence, intersect_without_pushdown
from benchmarks.nl_integration.runtime import INNER_K
from evaluation.nlseeker.metrics import reciprocal_rank
from src.DBHandler import DBHandler
from src.Operators.Combiners import Intersection
from src.Operators.Seekers import NaturalLanguage, SingleColumnOverlap
from src.Optimization import changeOptimizationOrder

from src.Operators.Seekers.SeekerBase import Seeker
from typing import Callable, List, Optional, Set, Tuple

WHOLE_LAKE = 1_000_000
SC_VALUES = 100
STEPS = ('inner cut', 'normalization', 'final cut')

PARTNER_SQL = """
WITH answer AS (
    SELECT DISTINCT ColumnId AS answer_column, CellValue FROM AllTables
    WHERE TableId = {answer} AND CellValue <> ''),
shared AS (
    SELECT a.TableId, a.ColumnId, answer.answer_column, count(DISTINCT a.CellValue) AS shared
    FROM AllTables a JOIN answer ON a.CellValue = answer.CellValue
    WHERE a.TableId <> {answer}
    GROUP BY 1, 2, 3),
sizes AS (
    SELECT a.TableId, a.ColumnId, count(DISTINCT a.CellValue) AS size
    FROM AllTables a JOIN (SELECT DISTINCT TableId, ColumnId FROM shared) c
      ON a.TableId = c.TableId AND a.ColumnId = c.ColumnId
    WHERE a.CellValue <> ''
    GROUP BY 1, 2)
SELECT shared.TableId, shared.ColumnId, shared.answer_column, shared.shared, sizes.size
FROM shared JOIN sizes ON shared.TableId = sizes.TableId AND shared.ColumnId = sizes.ColumnId
WHERE sizes.size >= {min_values} AND shared.shared >= {containment} * sizes.size
ORDER BY shared.shared * 1.0 / sizes.size DESC, sizes.size DESC, 1, 2, 3
LIMIT 1
"""


def arm_names(label: str) -> tuple:
    return (f'{label} alone', 'NL alone', 'NL alone at 2k', f'{label} then NL', f'NL then {label}', 'Without pushdown')


ARMS = arm_names('SC')


def run_arms(args, meter: NLMeter, question: dict, seeker: Callable[[int], Seeker], label: str,
             nl_tables: Set[int], exclude: str = '') -> dict:
    qid = question['qid']

    def nl(k):
        return NaturalLanguage(question['question'], k=k, rerank=bool(args.rerank))

    def timed(run):
        start = time.perf_counter()
        result = run()
        return result, time.perf_counter() - start

    def intersect(first, last):
        changeOptimizationOrder(first, last)
        combiner = Intersection(args.k)
        combiner.set_inputs([first, last])
        return timed(lambda: combiner.run(exclude))

    alone, last, first = f'{label} alone', f'{label} then NL', f'NL then {label}'
    candidates = set(seeker(WHOLE_LAKE).run(exclude))
    top, top_s = timed(lambda: seeker(args.k).run(exclude))
    inner, inner_s = timed(lambda: seeker(args.inner_k).run(exclude))
    nl_top, nl_top_s = timed(lambda: nl(args.k).run(exclude))
    meter.write(arm='NL alone', qid=qid)
    nl_2k, nl_2k_s = timed(lambda: nl(2 * args.k).run(exclude))
    meter.write(arm='NL alone at 2k', qid=qid)
    nl_inner, nl_inner_s = timed(lambda: nl(args.nl_inner_k).run(exclude))
    meter.write(arm='Without pushdown', qid=qid)
    nl_last, nl_last_s = intersect(seeker(args.inner_k), nl(args.k))
    meter.write(arm=last, qid=qid)
    nl_last_repeat, _ = intersect(seeker(args.inner_k), nl(args.k))
    meter.write(arm=f'{last}, repeat', qid=qid)
    nl_first, nl_first_s = intersect(nl(args.nl_inner_k), seeker(args.k))
    (first_leg_run,) = meter.write(arm=first, qid=qid)
    nl_first_leg = first_leg_run['tables']
    on_common = sorted(set(inner) & set(nl_first_leg))
    nl_on_common = nl(args.nl_inner_k).run(
        f'{exclude} AND TableId IN ({DBHandler.create_sql_list_numeric(on_common)}) ')[:args.k] if on_common else []
    meter.write(arm='NL on both first legs', qid=qid)

    arms = {alone: (distinct(top), top_s), 'NL alone': (nl_top, nl_top_s), 'NL alone at 2k': (nl_2k, nl_2k_s),
            last: (nl_last, nl_last_s), first: (distinct(nl_first), nl_first_s),
            'Without pushdown': (intersect_without_pushdown(inner, nl_inner, args.k), inner_s + nl_inner_s)}
    true_intersection = candidates & nl_tables
    return {
        'candidates': sorted(candidates), 'true_intersection': len(true_intersection),
        'arms': {arm: {'tables': tables, 'seconds': seconds} for arm, (tables, seconds) in arms.items()},
        'inner_rows': inner, 'nl_inner': nl_inner, 'nl_first_leg': nl_first_leg, 'nl_on_common': nl_on_common,
        'nl_last_repeat': nl_last_repeat,
        'divergence': divergence(nl_last, nl_first, nl_on_common, nl_first_leg, inner, args.k),
        'unsound': sorted({table for arm in (last, first, 'Without pushdown') for table in arms[arm][0]}
                          - true_intersection)}


def join_partner(args, db: DBHandler, question: dict, rng: random.Random) -> Optional[Tuple[dict, List[str]]]:
    partners = db.execute_and_fetchall(PARTNER_SQL.format(
        answer=question['answer'], containment=args.containment, min_values=args.min_values))
    if not partners:
        return None
    partner, column, answer_column, shared, size = partners[0]
    values = [row[0] for row in db.execute_and_fetchall(
        f"SELECT DISTINCT CellValue FROM AllTables WHERE TableId = {partner} AND ColumnId = {column} "
        "AND CellValue <> '' ORDER BY 1")]
    values = rng.sample(values, min(SC_VALUES, len(values)))
    return {'partner': partner, 'partner_column': column, 'answer_column': answer_column,
            'containment': shared / size, 'partner_values': size, 'sc_values': len(values)}, values


def run_lake(args, db: DBHandler) -> None:
    records = Records(args, 'order')
    meter = NLMeter(db, args.out / f'nl_runs_order_{args.lake}.jsonl')
    rng = random.Random(args.seed)
    nl_tables = set(nl_table_ids(db))

    NaturalLanguage('warm-up', k=1, n=1, rerank=bool(args.rerank)).run()
    meter.write(arm='warm-up')
    measured = 0
    pool = questions(db, args.lake)
    rng.shuffle(pool)
    for question in pool:
        if measured == args.questions:
            break
        found = join_partner(args, db, question, rng)
        if not found:
            records.write({'qid': question['qid'], 'answer': question['answer'], 'skipped': 'no partner'})
            continue
        measured += 1
        partner, values = found
        records.write({
            'qid': question['qid'], 'answer': question['answer'], 'answers': question['answers'], **partner,
            **run_arms(args, meter, question, lambda k: SingleColumnOverlap(values, k), 'SC', nl_tables,
                       f' AND TableId <> {partner["partner"]} ')})

    rows = records.read()
    write_summary(args, 'order', summarize(rows, args))
    measured_rows = [row for row in rows if 'skipped' not in row]
    if measured_rows:
        plot(measured_rows, args.out / f'order_{args.lake}.png', ARMS,
             f'U1 on {args.lake}, {len(measured_rows)} questions')


def _arm_quality(rows: list, arm: str):
    def tables(row):
        return row['arms'][arm]['tables']

    found = [row for row in rows if answer_found(row, tables(row))]
    returned = [row for row in rows if tables(row)]
    nl_found = [row for row in rows if answer_found(row, row['arms']['NL alone']['tables'])]
    return [len(found) / len(rows),
            statistics.mean(reciprocal_rank(tables(row), set(row['answers'])) for row in rows),
            statistics.mean(len(set(tables(row)) & set(row['candidates'])) / len(tables(row)) for row in returned)
            if returned else None,
            sum(answer_found(row, tables(row)) for row in nl_found) / len(nl_found) if nl_found else None,
            statistics.mean(len(tables(row)) for row in rows), 1 - len(returned) / len(rows),
            statistics.median(row['arms'][arm]['seconds'] for row in rows) * 1000]


def arms_table(rows: list, label: str, share: str) -> str:
    return markdown_table(['Arm', 'Answer returned', 'MRR', share, 'Answer retention', 'Mean tables', 'Empty',
                           'Median ms'], [[arm] + _arm_quality(rows, arm) for arm in arm_names(label)])


def order_sections(rows: list, args, label: str) -> str:
    def share(predicate):
        return sum(map(predicate, rows)) / len(rows)

    def last(row):
        return row['arms'][f'{label} then NL']['tables']

    def first(row):
        return row['arms'][f'NL then {label}']['tables']

    text = (f'\n## Order\n\nInner-k loss: a labelled answer table is among {label}\'s untruncated candidates but '
            f'none is among its first {args.inner_k} rows, or none is among NL\'s first {args.nl_inner_k} tables. '
            'Unsound tables lie outside the true intersection in a composed arm.\n\n')
    text += markdown_table(
        [f'Median {label} candidates', 'Median true intersection', 'True intersection > k', f'{label} inner-k loss',
         'NL inner-k loss', 'Identical lists', 'Identical sets', 'Unsound tables'],
        [[statistics.median(len(row['candidates']) for row in rows),
          statistics.median(row['true_intersection'] for row in rows),
          share(lambda r: r['true_intersection'] > args.k),
          share(lambda r: answer_found(r, r['candidates']) and not answer_found(r, r['inner_rows'])),
          share(lambda r: not answer_found(r, r['nl_inner'])),
          share(lambda r: last(r) == first(r)), share(lambda r: set(last(r)) == set(first(r))),
          sum(len(row['unsound']) for row in rows)]])

    diverged = [row for row in rows if row['divergence']['diverged']]
    noise = share(lambda r: set(last(r)) != set(r['nl_last_repeat']))
    text += (f'\n## Why the orders diverge\n\nFrom {label} then NL to NL then {label} in steps: the inner cut (each '
             'order only sees its first leg\'s tables; this step also deepens NL\'s pool to its first leg\'s k), the '
             f'normalization effect (NL restricted to the tables both first legs kept, against NL then {label}\'s '
             'own unrestricted first leg on them, at the same pool depth) and the final cut (NL\'s against '
             f'{label}\'s ranking on those tables). Counted among the diverging questions; one question can have '
             f'several. The noise floor is {label} then NL run twice: the diverging share, '
             f'{len(diverged) / len(rows):.0%}, is {"inside" if len(diverged) / len(rows) <= noise else "above"} '
             f'its {noise:.0%}. The first two steps compare separate NL runs, so their counts carry judge noise too; '
             'the floor is measured at k, so it may understate the noise of NL\'s deeper first-leg pool.\n\n')
    return text + markdown_table(
        ['Questions', 'Diverging sets'] + [f'... by {step}' for step in STEPS]
        + ['Noise floor: sets differ', 'Noise floor: lists differ'],
        [[len(rows), len(diverged)] + [sum(row['divergence'][step] for row in diverged) for step in STEPS]
         + [noise, share(lambda r: last(r) != r['nl_last_repeat'])]])


def summarize(records: list, args) -> str:
    rows = [row for row in records if 'skipped' not in row]
    text = (f'# U1: SC with NL on {args.lake}\n\nk {args.k}, first leg k {args.inner_k} (SC) or {args.nl_inner_k} '
            f'(NL), judge {"on" if args.rerank else "off"}, seed {args.seed}. SC queries a join partner: a column '
            f'of another table with at least {args.min_values} distinct values, of which the answer table holds at '
            f'least {args.containment:.0%} (at most {SC_VALUES} sampled). The partner\'s table is excluded from '
            'every arm. The true intersection is SC\'s untruncated candidates that have NL documents. An arm returns '
            'the answer when it returns any table Pneuma labels for the question, as parity scores a hit.\n\n'
            f'{len(rows)} questions measured; {len(records) - len(rows)} skipped without a join partner.\n')
    if not rows:
        return text
    text += (f'Median containment {statistics.median(row["containment"] for row in rows):.2f}, median partner '
             f'size {statistics.median(row["partner_values"] for row in rows)} values.\n\n'
             '## Arms\n\nThe headline is NL alone\'s joinable share: the share of its tables that SC finds on the '
             'partner, which is 1 for every composed arm by construction. Answer retention is the share of the '
             'questions where NL alone returns a labelled answer table in which the arm returns one too. SC '
             'alone\'s numbers are circular, because the partner is chosen so that the answer table holds its '
             'values, so no arm is measured against them. NL alone at 2k returns up to twice k tables.\n\n')
    partner_returned = sum(row['partner'] in result['tables'] for row in rows for result in row['arms'].values())
    return (text + arms_table(rows, 'SC', 'Joinable share') + order_sections(rows, args, 'SC')
            + f'\nThe partner\'s table was returned {partner_returned} times.\n')


def plot(rows: list, path, arms: tuple, title: str) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4))
    for arm in arms:
        returned, *_, milliseconds = _arm_quality(rows, arm)
        ax.scatter(milliseconds, returned)
        ax.annotate(arm, (milliseconds, returned), textcoords='offset points', xytext=(4, 4), fontsize=8)
    ax.set_xscale('log')
    ax.set_xlabel('median ms per question')
    ax.set_ylabel('answer returned')
    ax.set_ylim(-0.05, 1.05)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(argv=None) -> int:
    return run_lakes('order', run_lake, argv, questions=100, k=10, inner_k=INNER_K,
                     nl_inner_k=50, rerank=1, containment=0.9, min_values=10)


if __name__ == '__main__':
    sys.exit(main())
