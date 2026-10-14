import argparse
import json
from math import comb
from pathlib import Path

from evaluation.agent.analysis import _csv, read_cases

from typing import Dict, List, Optional

ARMS = ('on', 'off')
TIERS = ('simple', 'tasks', 'free form')


def cases(paths: List[str]) -> Dict[str, dict]:
    return {case['id']: {'lake': entry['lake'], 'tier': entry['tier'], 'level': case['level']}
            for entry in read_cases(paths) for case in entry['cases']}


def outcomes(folder: Path) -> Dict[str, Optional[List[bool]]]:
    found = {}
    for path in sorted((folder / 'rows').glob('*.json')):
        rows = json.loads(path.read_text())
        found[rows[0]['case']] = None if 'failed' in rows[0] else [
            bool(row['correct']) for row in sorted(rows, key=lambda row: row['budget'])]
    return found


def allowed(level) -> int:
    return 0 if level in (0, 'inferable') else int(level)


def sign_test(better: int, worse: int) -> float:
    n, k = better + worse, min(better, worse)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n) if n else 1.0


def tables(run: Path, known: Dict[str, dict]) -> Dict[str, List[dict]]:
    runs, passed = [], {case: {arm: 0 for arm in ARMS} for case in known}
    for folder in sorted(path for path in run.iterdir() if path.is_dir() and path.name.split('-')[0] in ARMS):
        arm, repeat = folder.name.split('-r')
        found = outcomes(folder)
        ok = {case: bool(found.get(case)) and found[case][allowed(info['level'])] for case, info in known.items()}
        for case, value in ok.items():
            passed[case][arm] += value
        runs.append({'arm': arm, 'repeat': int(repeat), 'cases': len(known), 'pass': sum(ok.values()),
                     **{f'b{budget}': sum(bool(found.get(case)) and found[case][budget] for case in known)
                        for budget in range(4)},
                     **{tier: sum(ok[case] for case, info in known.items() if info['tier'] == tier) for tier in TIERS},
                     'degraded': sum(ok[case] for case, info in known.items() if info['level'] in (1, 2, 3)),
                     'without_output': sum(case not in found for case in known),
                     'failed': sum(case in found and found[case] is None for case in known)})
    per_case = [{'case': case, **known[case], 'on': passed[case]['on'], 'off': passed[case]['off']} for case in known]
    better = sum(row['on'] > row['off'] for row in per_case)
    worse = sum(row['on'] < row['off'] for row in per_case)
    return {'runs': runs, 'cases': per_case,
            'sign_test': [{'on_better': better, 'off_better': worse, 'equal': len(per_case) - better - worse,
                           'p': sign_test(better, worse)}]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('run', type=Path)
    parser.add_argument('--cases', nargs='+', required=True)
    args = parser.parse_args()
    for name, table in tables(args.run, cases(args.cases)).items():
        _csv(args.run / f'{name}.csv', table)
        print(f'{name}: {args.run / f"{name}.csv"}')


if __name__ == '__main__':
    main()
