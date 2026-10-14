import argparse
import json
from math import comb
from pathlib import Path

from evaluation.agent.analysis import _csv, read_cases

from typing import Dict, List, Optional, Tuple

ARMS = ('on', 'off')
TIERS = ('simple', 'tasks', 'free form')


def cases(paths: List[str]) -> Dict[str, dict]:
    return {case['id']: {'lake': entry['lake'], 'tier': entry['tier'], 'level': case['level']}
            for entry in read_cases(paths) for case in entry['cases']}


def outcomes(folders: List[Path]) -> Dict[str, Optional[List[bool]]]:
    found = {}
    for path in sorted(path for folder in folders for path in (folder / 'rows').glob('*.json')):
        rows = json.loads(path.read_text())
        found[rows[0]['case']] = None if 'failed' in rows[0] else [
            bool(row['correct']) for row in sorted(rows, key=lambda row: row['budget'])]
    return found


def allowed(level) -> int:
    return 0 if level in (0, 'inferable') else int(level)


def sign_test(better: int, worse: int) -> float:
    n, k = better + worse, min(better, worse)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n) if n else 1.0


def groups(run: Path) -> Dict[Tuple[str, int], List[Path]]:
    return {(folder.name.split('-r')[0], int(folder.name.split('-r')[1])): [folder]
            for folder in sorted(run.iterdir()) if folder.is_dir() and folder.name.split('-')[0] in ARMS}


def tables(arms: Dict[Tuple[str, int], List[Path]], known: Dict[str, dict]) -> Dict[str, List[dict]]:
    runs, passed = [], {case: {arm: 0 for arm in ARMS} for case in known}
    for (arm, repeat), folders in sorted(arms.items()):
        found = outcomes(folders)
        ok = {case: bool(found.get(case)) and found[case][allowed(info['level'])] for case, info in known.items()}
        for case, value in ok.items():
            passed[case][arm] += value
        runs.append({'arm': arm, 'repeat': repeat, 'cases': len(known), 'pass': sum(ok.values()),
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
    parser.add_argument('run', type=Path, help='A folder with on-r<n> and off-r<n> runs, and where the tables go.')
    parser.add_argument('--cases', nargs='+', required=True)
    parser.add_argument('--arm', nargs='+', action='append', metavar=('ARM REPEAT', 'FOLDER'),
                        help='on|off, the repeat and its run folders, in place of the run folder\'s own runs.')
    parser.add_argument('--exclude', nargs='*', default=[], help='Case ids left out.')
    args = parser.parse_args()
    arms = {(arm, int(repeat)): [Path(folder) for folder in folders] for arm, repeat, *folders in args.arm} \
        if args.arm else groups(args.run)
    known = {case: info for case, info in cases(args.cases).items() if case not in args.exclude}
    args.run.mkdir(parents=True, exist_ok=True)
    for name, table in tables(arms, known).items():
        _csv(args.run / f'{name}.csv', table)
        print(f'{name}: {args.run / f"{name}.csv"}')


if __name__ == '__main__':
    main()
