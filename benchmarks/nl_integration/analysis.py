import argparse
import json
import statistics
import sys
from pathlib import Path

from benchmarks.nl_integration.common import LAKES, RESULTS, open_lake, questions
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
    return [{'arm': arm, 'against': o, 'gained': g, 'lost': l, 'p_mid': round(q, 4), 'p_holm': round(h, 4)}
            for o, (g, l), q, h in zip(others, tests, p, holm(p))]


def pushdown_summary(rows: list) -> dict:
    hit = lambda r, tables: bool(set(r['answers']) & set(tables))
    tables = lambda r, a: r['arms'][a]
    arms = ['Restricted', 'Post-filter', 'Deep post-filter', 'Filtered full ranking']
    summary = {'questions': len({r['qid'] for r in rows}),
               'skipped': len({r['qid'] for r in rows if r.get('skipped') == 'no restriction'})}
    for restriction in ('SC', 'Keyword'):
        every = [r for r in rows if 'arms' in r and r['restriction'] == restriction]
        group = [r for r in every if hit(r, r['allowed'])]
        if not group:
            continue
        restricted = lambda r: tables(r, 'Restricted')
        kept = [len(set(tables(r, 'Post-filter')) & set(restricted(r))) / len(restricted(r))
                for r in group if restricted(r)]
        full = lambda r: tables(r, 'Filtered full ranking')
        summary[restriction] = {
            'questions': len(every), 'with_answer_allowed': len(group),
            'median_restriction_size': statistics.median(len(r['allowed']) for r in group),
            'found_pct': {a: pct(sum(hit(r, tables(r, a)) for r in group), len(group)) for a in arms},
            'fewer_tables_pct': {a: pct(sum(len(tables(r, a)) < len(restricted(r)) for r in group), len(group))
                                 for a in arms[1:]},
            'tests': paired(group, 'Restricted', arms[1:], tables, hit),
            'post_filter_keeps_pct': round(100 * statistics.mean(kept), 1),
            'full_ranking_other_sets': sum(set(full(r)) != set(restricted(r)) for r in group),
            'full_ranking_other_order': sum([t for t in full(r) if t in restricted(r)]
                                            != [t for t in restricted(r) if t in full(r)] for r in group),
            'restricted_repeat_differs': sum(r['restricted_repeat'] != restricted(r) for r in group)}
    return summary


def plan_summary(rows: list, experiment: str, first: str) -> dict:
    hit = lambda r, tables: bool(set(r['answers']) & set(tables))
    tables = lambda r, a: r['arms'][a]['tables']
    measured = [r for r in rows if 'arms' in r]
    skipped = len(rows) - len(measured)
    rows = [r for r in measured if experiment == 'order' or r.get('held')]
    blend, reverse = f'{first} then NL', f'NL then {first}'
    arms = [f'{first} alone', 'NL alone', 'NL alone at 2k', blend, reverse, 'Without pushdown']

    def rank(r, a):
        return next((1 / (i + 1) for i, t in enumerate(tables(r, a)) if t in r['answers']), 0)

    nl_found = [r for r in rows if hit(r, tables(r, 'NL alone'))]
    lost = [r for r in nl_found if not hit(r, tables(r, blend))]
    differ = [r for r in rows if set(tables(r, blend)) != set(tables(r, reverse))]
    disjoint = [r for r in differ if not set(r['inner_rows']) & set(r['nl_first_leg'])]
    summary = {
        'questions': len(rows), 'skipped': skipped, 'not_counted': len(measured) - len(rows),
        'arms': {a: {'found_pct': pct(sum(hit(r, tables(r, a)) for r in rows), len(rows)),
                     'mrr': round(statistics.mean(rank(r, a) for r in rows), 3),
                     'mean_tables': round(statistics.mean(len(tables(r, a)) for r in rows), 2),
                     'empty': sum(not tables(r, a) for r in rows),
                     'median_ms': round(statistics.median(r['arms'][a]['seconds'] for r in rows) * 1000, 1)}
                 for a in arms},
        'tests': paired(rows, blend, [a for a in arms if a != blend], tables, hit),
        'nl_alone_found': len(nl_found), 'kept_of_nl_alone_found': len(nl_found) - len(lost),
        'lost_outside_first_rows': sum(not set(r['answers']) & set(tables(r, 'NL alone')) & set(r['inner_rows'])
                                       for r in lost),
        'inner_k_loss': {first: sum(hit(r, r['candidates']) and not hit(r, r['inner_rows']) for r in rows),
                         'NL': sum(not hit(r, r['nl_inner']) for r in rows)},
        'orders_differ': len(differ), 'first_seekers_disjoint': len(disjoint),
        'divergence_steps': {s: sum(r['divergence'][s] for r in differ)
                             for s in ('inner cut', 'normalization', 'final cut')},
        'repeat_sets_differ': sum(set(r['nl_last_repeat']) != set(tables(r, blend)) for r in rows),
        'unsound_tables': sum(len(r['unsound']) for r in rows)}
    if experiment == 'order':
        gained = [r for r in rows if hit(r, tables(r, blend)) and not hit(r, tables(r, 'NL alone'))]
        summary |= {'restriction_holds_generating_table': sum(r['answer'] in set(r['inner_rows']) for r in rows),
                    'gained_over_nl_alone': len(gained),
                    'gained_with_generating_table': sum(r['answer'] in tables(r, blend) for r in gained),
                    'gained_generating_table_only': sum(set(r['answers']) & set(tables(r, blend)) == {r['answer']}
                                                        for r in gained)}
    return summary


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--lake', choices=list(LAKES), required=True)
    for experiment in ('pushdown', 'order', 'regime'):
        parser.add_argument(f'--{experiment}', required=True, help='run directory name under --results')
    parser.add_argument('--results', type=Path, default=RESULTS)
    parser.add_argument('--answers', type=Path, help='answer tables per question, for runs that did not record them')
    parser.add_argument('--out', type=Path, help='where analysis_<lake>.json goes, default --results')
    args = parser.parse_args(argv)
    cache = {}

    def answers():
        if 'answers' not in cache:
            cache['answers'] = json.loads(args.answers.read_text()) if args.answers else \
                {q['qid']: q['answers'] for q in questions(open_lake(args.lake), args.lake)}
        return cache['answers']

    summary = {'lake': args.lake, 'runs': {e: getattr(args, e) for e in ('pushdown', 'order', 'regime')}}
    for experiment in ('pushdown', 'order', 'regime'):
        rows, _ = load(args.results / getattr(args, experiment), experiment, args.lake, answers)
        summary[experiment] = pushdown_summary(rows) if experiment == 'pushdown' else \
            plan_summary(rows, experiment, 'SC' if experiment == 'order' else 'Keyword')
    out = (args.out or args.results) / f'analysis_{args.lake}.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=1))
    print(out)
    return 0


if __name__ == '__main__':
    sys.exit(main())
