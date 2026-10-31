import argparse
import configparser
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LAKES = ['adventure_works', 'chembl', 'chicago_open', 'fetaqa', 'public_bi']
TOKEN_SEEKERS = {'KW': 'Keyword', 'SC': 'SingleColumnOverlap', 'C': 'Correlation', 'MC': 'MultiColumnOverlap'}
PLAN_K = 10


def derive(lake: str) -> dict:
    (run,) = sorted((ROOT / 'results' / 'NLIntegration' / lake).glob(f'*__runtime/{lake}'))
    token_path, nl_path = run / f'runtime_{lake}.jsonl', run / f'nl_runs_runtime_{lake}.jsonl'
    config = configparser.ConfigParser()
    config.read(ROOT / 'config' / f'{lake}.ini')
    n = config.getint('NLSeeker', 'n')

    rows = [json.loads(line) for line in token_path.read_text().splitlines()]
    costs = {name: round(statistics.median(row['runtime_s'] for row in rows
                                           if row['seeker'] == seeker and row.get('finished', True)) * 1000, 1)
             for seeker, name in TOKEN_SEEKERS.items()}

    runs = [json.loads(line) for line in nl_path.read_text().splitlines()]
    runs = [run for run in runs if run['arm'] != 'warm-up']
    parts = [part for part in runs if (part['k'], part['n']) == (PLAN_K, n)]
    judge = statistics.median(part['total'] for part in parts if part['judge_on']) * 1000
    nojudge = statistics.median(part['total'] for part in parts if not part['judge_on']) * 1000
    costs['NaturalLanguage_judge'] = round(judge)
    costs['NaturalLanguage_nojudge'] = round(nojudge, 1)
    # NaturalLanguage.ml_cost parameters in seconds, over all k and n.
    costs['NaturalLanguage_t_retrieval'] = round(statistics.median(run['total'] for run in runs if not run['judge_on']), 4)
    costs['NaturalLanguage_t_doc'] = round(statistics.median(run['judge'] / run['judge_calls'] for run in runs
                                                             if run['judge_on']), 4)
    costs['source'] = [str(path.relative_to(ROOT)) for path in (token_path, nl_path)]
    return costs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--lakes', nargs='+', default=LAKES)
    args = parser.parse_args()
    for lake in args.lakes:
        costs = derive(lake)
        (ROOT / 'config' / 'costs' / f'{lake}.json').write_text(json.dumps(costs, indent=2) + '\n')
        print(lake, {name: value for name, value in costs.items() if name != 'source'})


if __name__ == '__main__':
    main()
