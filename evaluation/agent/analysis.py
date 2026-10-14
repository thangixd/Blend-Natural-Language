import argparse
import csv
import json
from math import comb
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import yaml


from typing import Any, Dict, List, Sequence, Tuple, Union

SEED, DRAWS = 20261001, 10000
DEGRADED = (1, 2, 3)
STRUCTURAL = ('combiner', 'direction', 'removal', 'part_count')


def groups(level: Any) -> List[str]:
    return [f'level {level}' if level != 'inferable' else 'inferable'] + ['degraded'] * (level in DEGRADED)


def _cases(rows: List[dict]) -> Dict[Tuple[str, str, int], Dict[str, List[dict]]]:
    found = {}
    for row in rows:
        if 'failed' not in row:
            for group in groups(row['level']):
                found.setdefault((row['lake'], group, row['budget']), {}).setdefault(row['case'], []).append(row)
    return found


def mid_p(gained: int, lost: int) -> float:
    pairs, fewer = gained + lost, min(gained, lost)
    return (2 * sum(comb(pairs, k) for k in range(fewer + 1)) - comb(pairs, fewer)) / 2 ** pairs


def holm(p_values: List[float]) -> List[float]:
    order = sorted(range(len(p_values)), key=lambda index: p_values[index])
    adjusted, running = [0.0] * len(p_values), 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (len(p_values) - rank) * p_values[index]))
        adjusted[index] = running
    return adjusted


def _stats(cases: Dict[str, List[dict]]) -> Dict[str, Any]:
    plans = {}
    for found in cases.values():
        plans.setdefault(found[0]['gold_plan_id'], []).append(np.mean([row['correct'] for row in found]))
    sums = np.array([sum(plans[plan]) for plan in sorted(plans)])
    counts = np.array([len(plans[plan]) for plan in sorted(plans)])
    drawn = np.random.default_rng(SEED).integers(0, len(plans), size=(DRAWS, len(plans)))
    low, high = np.quantile(sums[drawn].sum(axis=1) / counts[drawn].sum(axis=1), [0.025, 0.975])
    repeats = {}
    for row in (row for found in cases.values() for row in found):
        repeats.setdefault(row['repeat'], []).append(row['correct'])
    rates = [np.mean(outcomes) for outcomes in repeats.values()]
    return {'cases': int(counts.sum()), 'correct': float(sums.sum()), 'rate': float(sums.sum() / counts.sum()),
            'low': float(low), 'high': float(high), 'repeats': len(repeats), 'repeat_low': float(min(rates)),
            'repeat_high': float(max(rates))}


def correctness(rows: List[dict]) -> List[dict]:
    failed = {}
    for row in rows:
        if 'failed' in row:
            for group in groups(row['level']):
                failed.setdefault((row['lake'], group), set()).add(row['case'])
    return [{'lake': lake, 'group': group, 'budget': budget} | _stats(cases) |
            {'failed': len(failed.get((lake, group), ()))}
            for (lake, group, budget), cases in sorted(_cases(rows).items(), key=str)]


MISSING = 'not in the case files'


def correctness_by(rows: List[dict], entries: List[dict]) -> List[dict]:
    about = {(entry['lake'], case['id']): {
        'tier': entry['tier'], 'class': entry['class'],
        'natural_language': 'with' if any(node.get('fill', {}).get('seeker') == 'natural_language'
                                          for node in _nodes(entry).values()) else 'without'}
        for entry in entries for case in entry['cases']}
    table = []
    for (lake, group, budget), cases in sorted(_cases(rows).items(), key=str):
        for by in ('tier', 'class', 'natural_language'):
            split = {}
            for case, found in cases.items():
                split.setdefault(about.get((lake, case), {}).get(by, MISSING), {})[case] = found
            table += [{'lake': lake, 'group': group, 'by': by, 'value': value, 'budget': budget} | _stats(chosen)
                      for value, chosen in sorted(split.items())]
    return table


def mcnemar(rows: List[dict]) -> List[dict]:
    cases = _cases(rows)

    def majority(found: List[dict]) -> bool:
        return 2 * sum(row['correct'] for row in found) > len(found)

    table = []
    for lake, group in sorted({(lake, group) for lake, group, _ in cases}):
        start = cases.get((lake, group, 0), {})
        tests = []
        for budget in range(1, 4):
            later = cases.get((lake, group, budget), {})
            paired = [(majority(start[case]), majority(later[case])) for case in sorted(start.keys() & later.keys())]
            gained, lost = paired.count((False, True)), paired.count((True, False))
            tests.append({'lake': lake, 'group': group, 'budget': budget, 'cases': len(paired), 'gained': gained,
                          'lost': lost, 'p_mid': mid_p(gained, lost)})
        for test, adjusted in zip(tests, holm([test['p_mid'] for test in tests])):
            table.append(test | {'p_holm': adjusted})
    return table


def _asked(row: dict) -> int:
    return len(row['questions']) + len(row['finishing_questions'])


def cost(rows: List[dict]) -> List[dict]:
    table = []
    for (lake, group, budget), cases in sorted(_cases(rows).items(), key=str):
        found = [row for runs in cases.values() for row in runs]
        asked = list(map(_asked, found))
        model_seconds, input_tokens, output_tokens = np.mean(
            [[sum(step[name] for step in row['steps']) for name in ('seconds', 'input_tokens', 'output_tokens')]
             for row in found], axis=0)
        table.append({'lake': lake, 'group': group, 'budget': budget, 'cases': len(cases),
                      'questions': float(np.mean(asked)), 'asked': float(np.mean([count > 0 for count in asked])),
                      'seconds': float(np.mean([row['seconds'] for row in found])),
                      'model_seconds': float(model_seconds), 'input_tokens': float(input_tokens),
                      'output_tokens': float(output_tokens)})
    return table


def questions_curve(rows: List[dict]) -> List[dict]:
    return [{'lake': lake, 'group': group, 'budget': budget,
             'questions': float(np.mean([_asked(row) for runs in cases.values() for row in runs]))} | _stats(cases)
            for (lake, group, budget), cases in sorted(_cases(rows).items(), key=str) if group in ('level 0', 'degraded')]


def _nodes(entry: dict) -> Dict[str, dict]:
    nodes, waiting = {}, [entry['gold_plan']['root']]
    while waiting:
        node = waiting.pop()
        nodes[node['id']] = node
        waiting += node.get('inputs', [])
    return nodes


def structural_values(entry: dict) -> Dict[str, List[Tuple[str, str]]]:
    nodes = _nodes(entry)

    def kind(id: str) -> str:
        return nodes[id].get('kind') or nodes[id]['fill']['seeker']

    return {case['id']: [(field['field'], f"keep {kind(field['intended']['kept'])}, remove "
                                          f"{kind(field['intended']['removed'])}; the kept side named "
                                          f"{'first' if field['named_first'] == 'kept' else 'second'}"
                          if field['field'] == 'direction' else str(field['intended']))
                         for field in case['open_fields'] if field['field'] in STRUCTURAL]
            for case in entry['cases']}


def dominant(rows: List[dict], entries: List[dict]) -> List[dict]:
    values = {(entry['lake'], case): fields for entry in entries for case, fields in structural_values(entry).items()}
    runs = {}
    for row in rows:
        if 'failed' not in row and row['budget'] == 0:
            runs.setdefault((row['lake'], row['case']), []).append(row['correct'])
    found = {}
    for (lake, case), outcomes in sorted(runs.items()):
        for field, value in values.get((lake, case), []):
            found.setdefault((lake, field, value), []).append(np.mean(outcomes))
    table = []
    for (lake, field, value), scores in sorted(found.items()):
        of_field = sum(len(others) for (other_lake, other, _), others in found.items()
                       if (other_lake, other) == (lake, field))
        table.append({'lake': lake, 'field': field, 'intended': value, 'cases': len(scores),
                      'share': len(scores) / of_field, 'correct': float(sum(scores)),
                      'rate': float(np.mean(scores))})
    return table


def read_cases(paths: Sequence[Union[str, Path]]) -> List[dict]:
    return [entry for path in paths for entry in yaml.safe_load(Path(path).read_text())]


def load(folder: Path) -> List[dict]:
    rows = [row for path in sorted((folder / 'rows').glob('*.json')) for row in json.loads(path.read_text())]
    if {row['condition'] for row in rows} - {'main'}:
        raise ValueError('the analysis covers the main curve; these rows mix conditions')
    return rows


def _csv(path: Path, table: List[dict]) -> None:
    with path.open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(table[0]) if table else [])
        writer.writeheader()
        writer.writerows(table)


def write(rows: List[dict], out: Path, entries: Sequence[dict] = ()) -> None:
    out.mkdir(parents=True, exist_ok=True)
    _csv(out / 'correctness.csv', correctness(rows))
    _csv(out / 'mcnemar.csv', mcnemar(rows))
    _csv(out / 'cost.csv', cost(rows))
    _csv(out / 'questions_curve.csv', questions_curve(rows))
    if entries:
        _csv(out / 'dominant.csv', dominant(rows, list(entries)))
        _csv(out / 'correctness_by.csv', correctness_by(rows, list(entries)))
    figure(out / 'correctness.csv', out / 'budget_curve')
    figure(out / 'questions_curve.csv', out / 'questions_curve', x='questions')


CURVES = {'level 0': '#2a78d6', 'degraded': '#eb6834', 'inferable': '#1baf7a'}


def figure(table: Path, out: Path, x: str = 'budget') -> None:
    with table.open() as file:
        found = list(csv.DictReader(file))
    lakes = sorted({row['lake'] for row in found})
    fig, axes = plt.subplots(1, len(lakes), figsize=(4.5 * len(lakes), 3.6), sharey=True, squeeze=False)
    for axis, lake in zip(axes[0], lakes):
        for shift, (group, color) in zip((-0.06, 0.0, 0.06), CURVES.items()):
            points = sorted((int(row['budget']), float(row[x]), float(row['rate']), float(row['low']),
                             float(row['high'])) for row in found if (row['lake'], row['group']) == (lake, group))
            if points:
                budgets, xs, rates, lows, highs = zip(*points)
                xs = np.add(xs, shift if x == 'budget' else 0)
                axis.errorbar(xs, rates, yerr=[np.subtract(rates, lows), np.subtract(highs, rates)], color=color,
                              linewidth=2, elinewidth=1, marker='o', markersize=6, capsize=3, label=group)
                if x != 'budget':
                    labels = {}
                    for budget, at, rate in zip(budgets, xs, rates):
                        labels.setdefault((at, rate), []).append(str(budget))
                    for at, rate in labels:
                        axis.annotate('b' + ','.join(labels[at, rate]), (at, rate), textcoords='offset points',
                                      xytext=(4, 4), fontsize=7, color='#52514e')
        axis.set(title=lake, xlabel='Budget (clarification rounds)' if x == 'budget' else 'Questions asked per case',
                 ylim=(-0.03, 1.05))
        if x == 'budget':
            axis.set_xticks(range(4))
        axis.grid(axis='y', color='#e4e3df', linewidth=0.8)
        axis.spines[['top', 'right']].set_visible(False)
    axes[0][0].set_ylabel('Plan correctness')
    axes[0][-1].legend(frameon=False, loc='lower right')
    fig.tight_layout()
    fig.savefig(out.with_suffix('.pdf'))
    fig.savefig(out.with_suffix('.png'), dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--runs', type=Path, required=True, help="The harness's --out directory.")
    parser.add_argument('--cases', nargs='*', default=[], help='The case files, for the intended values.')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    write(load(args.runs), args.out, read_cases(args.cases))


if __name__ == '__main__':
    main()
