import argparse
import importlib
import json
import statistics
import sys
from pathlib import Path

from benchmarks.nl_integration.common import LAKES, RESULTS, nl_config, nl_table_ids, open_lake, questions
from evaluation.agent.analysis import holm, mid_p


def _value(text: str):
    for cast in (int, float):
        try:
            return cast(text)
        except ValueError:
            pass
    return {'True': True, 'False': False, 'None': None}.get(text, text)


def load(run: Path, experiment: str, lake: str, answers: dict) -> tuple:
    directory = run / lake
    saved = json.loads((directory / f'provenance_{experiment}_{lake}.json').read_text())['args']
    args = argparse.Namespace(**{name: _value(text) for name, text in saved.items()})
    rows = [json.loads(line) for line in open(directory / f'{experiment}_{lake}.jsonl')]
    for row in rows:
        if 'answers' not in row and 'skipped' not in row:
            row['answers'] = answers()[row['qid']]
    return rows, args


def pct(count: int, total: int):
    return round(100 * count / total, 1) if total else None


def paired(rows: list, arm: str, others: list, tables, hit) -> list:
    tests = [(sum(hit(r, tables(r, arm)) and not hit(r, tables(r, o)) for r in rows),
              sum(hit(r, tables(r, o)) and not hit(r, tables(r, arm)) for r in rows)) for o in others]
    p = [mid_p(*t) for t in tests]
    return [f'  {arm} vs {o}: gained {g} lost {l} p {q:.4f} holm {h:.4f}'
            for o, (g, l), q, h in zip(others, tests, p, holm(p))]


def pushdown_lines(rows: list) -> list:
    hit = lambda r, tables: bool(set(r['answers']) & set(tables))
    tables = lambda r, a: r['arms'][a]
    arms = ['Restricted', 'Post-filter', 'Deep post-filter', 'Filtered full ranking']
    lines = ['## Pushdown']
    for restriction in ('SC', 'Keyword'):
        every = [r for r in rows if 'arms' in r and r['restriction'] == restriction]
        group = [r for r in every if hit(r, r['allowed'])]
        if not group:
            continue
        restricted = lambda r: tables(r, 'Restricted')
        lines += [f'### {restriction}: {len(group)} of {len(every)} with an answer table in the restriction',
                  f'  median restriction size {statistics.median(len(r["allowed"]) for r in group)}',
                  f'  hit % {({a: pct(sum(hit(r, tables(r, a)) for r in group), len(group)) for a in arms})}',
                  f'  fewer tables % {({a: pct(sum(len(tables(r, a)) < len(restricted(r)) for r in group), len(group)) for a in arms[1:]})}',
                  *paired(group, 'Restricted', arms[1:], tables, hit)]
        kept = [len(set(tables(r, 'Post-filter')) & set(restricted(r))) / len(restricted(r))
                for r in group if restricted(r)]
        full = lambda r: tables(r, 'Filtered full ranking')
        lines += [f'  post-filter keeps a mean of {round(100 * statistics.mean(kept), 1)}% of the restricted tables',
                  f'  full ranking: other sets {sum(set(full(r)) != set(restricted(r)) for r in group)}, '
                  f'other order {sum([t for t in full(r) if t in restricted(r)] != [t for t in restricted(r) if t in full(r)] for r in group)}, '
                  f'repeat differs {sum(r["restricted_repeat"] != restricted(r) for r in group)}']
    return lines


def plan_lines(rows: list, experiment: str, first: str) -> list:
    hit = lambda r, tables: bool(set(r['answers']) & set(tables))
    tables = lambda r, a: r['arms'][a]['tables']
    rows = [r for r in rows if 'arms' in r and (experiment == 'order' or r.get('held'))]
    blend, reverse = f'{first} then NL', f'NL then {first}'
    arms = [f'{first} alone', 'NL alone', 'NL alone at 2k', blend, reverse, 'Without pushdown']

    def rank(r, a):
        return next((1 / (i + 1) for i, t in enumerate(tables(r, a)) if t in r['answers']), 0)

    nl_found = [r for r in rows if hit(r, tables(r, 'NL alone'))]
    lost = [r for r in nl_found if not hit(r, tables(r, blend))]
    differ = [r for r in rows if set(tables(r, blend)) != set(tables(r, reverse))]
    disjoint = [r for r in differ if not set(r['inner_rows']) & set(r['nl_first_leg'])]
    lines = [f'## {experiment}: {len(rows)} questions',
             f'  hit % {({a: pct(sum(hit(r, tables(r, a)) for r in rows), len(rows)) for a in arms})}',
             *paired(rows, blend, [a for a in arms if a != blend], tables, hit),
             f'  {blend} keeps {len(nl_found) - len(lost)} of {len(nl_found)} questions NL alone finds; of the '
             f'{len(lost)} lost, {sum(not set(r["answers"]) & set(tables(r, "NL alone")) & set(r["inner_rows"]) for r in lost)} '
             f'have NL alone\'s answer outside the first seeker\'s rows',
             f'  MRR NL alone {round(statistics.mean(rank(r, "NL alone") for r in rows), 3)}, '
             f'{blend} {round(statistics.mean(rank(r, blend) for r in rows), 3)}',
             f'  orders differ {len(differ)}, first seekers share no table {len(disjoint)} '
             f'(without pushdown returns {sorted({len(tables(r, "Without pushdown")) for r in disjoint})} tables, '
             f'{blend} {sorted({len(tables(r, blend)) for r in disjoint})})',
             f'  divergence steps {({s: sum(r["divergence"][s] for r in differ) for s in ("inner cut", "normalization", "final cut")})}',
             f'  repeat sets differ {sum(set(r["nl_last_repeat"]) != set(tables(r, blend)) for r in rows)}, '
             f'unsound tables {sum(len(r["unsound"]) for r in rows)}']
    if experiment == 'order':
        gained = [r for r in rows if hit(r, tables(r, blend)) and not hit(r, tables(r, 'NL alone'))]
        lines += [f'  restriction holds the generating table {sum(r["answer"] in set(r["inner_rows"]) for r in rows)}',
                  f'  gained {len(gained)}, {blend} returns the generating table for '
                  f'{sum(r["answer"] in tables(r, blend) for r in gained)}, as the only answer table for '
                  f'{sum(set(r["answers"]) & set(tables(r, blend)) == {r["answer"]} for r in gained)}']
    return lines


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--lake', choices=list(LAKES), required=True)
    for experiment in ('pushdown', 'order', 'regime'):
        parser.add_argument(f'--{experiment}', required=True, help='run directory name under --results')
    parser.add_argument('--results', type=Path, default=RESULTS)
    args = parser.parse_args(argv)
    cache = {}

    def db():
        if 'db' not in cache:
            cache['db'] = open_lake(args.lake)
        return cache['db']

    def answers():
        if 'answers' not in cache:
            cache['answers'] = {q['qid']: q['answers'] for q in questions(db(), args.lake)}
        return cache['answers']

    lines = [f'# Paired tests and counts on {args.lake}', '',
             *(f'{e}: {getattr(args, e)}' for e in ('pushdown', 'order', 'regime')), '']
    for experiment in ('pushdown', 'order', 'regime'):
        run = args.results / getattr(args, experiment)
        rows, run_args = load(run, experiment, args.lake, answers)
        run_args.out = run / args.lake
        module = importlib.import_module(f'benchmarks.nl_integration.{experiment}')
        if experiment == 'pushdown':
            config = nl_config(db())
            documents = db().execute_and_fetchall(f'SELECT count(*) FROM "{config.documents_table}"')[0][0]
            text = module.summarize(rows, run_args, len(nl_table_ids(db())), documents)
            lines += pushdown_lines(rows)
        else:
            text = module.summarize(rows, run_args)
            lines += plan_lines(rows, experiment, 'SC' if experiment == 'order' else 'Keyword')
        print(text)
        lines.append('')
    print('\n'.join(lines))
    return 0


if __name__ == '__main__':
    sys.exit(main())
