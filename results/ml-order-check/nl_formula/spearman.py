import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from xgboost import XGBRegressor

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
LAKES = ['adventure_works', 'chembl', 'chicago_open', 'fetaqa', 'public_bi']

model = XGBRegressor()
model.load_model(HERE / 'NaturalLanguage_model.json')


def rho(predicted, measured):
    return '' if np.ptp(predicted) == 0 else round(spearmanr(predicted, measured).statistic, 2)


rows = []
for lake in LAKES:
    costs = json.loads((ROOT / 'config' / 'costs' / f'{lake}.json').read_text())
    t_retrieval, t_doc = costs['NaturalLanguage_t_retrieval'], costs['NaturalLanguage_t_doc']
    (path,) = (ROOT / 'results' / 'NLIntegration' / lake).glob(f'*__runtime/{lake}/nl_runs_runtime_{lake}.jsonl')
    runs = [json.loads(line) for line in path.read_text().splitlines()]
    runs = [run for run in runs if run['arm'] != 'warm-up']
    total = np.array([run['total'] for run in runs])
    judge = np.array([run['judge_on'] for run in runs])
    formula = np.array([t_retrieval + run['judge_on'] * run['k'] * run['n'] * t_doc for run in runs])
    xgb = model.predict(np.array([[len(run['question'].split()), run['k'] * run['n'], int(run['judge_on'])]
                                  for run in runs], dtype=float))
    for group, mask in (('judge', judge), ('no judge', ~judge), ('all', np.ones(len(runs), bool))):
        rows.append({'lake': lake, 'runs': group, 'n': int(mask.sum()), 't_retrieval_s': t_retrieval, 't_doc_s': t_doc,
                     'spearman_formula': rho(formula[mask], total[mask]), 'spearman_xgb': rho(xgb[mask], total[mask]),
                     'median_measured_s': round(float(np.median(total[mask])), 3),
                     'median_abs_error_formula_s': round(float(np.median(abs(formula[mask] - total[mask]))), 3),
                     'median_abs_error_xgb_s': round(float(np.median(abs(xgb[mask] - total[mask]))), 3),
                     'min_formula_s': round(float(formula[mask].min()), 4), 'min_xgb_s': round(float(xgb[mask].min()), 4),
                     'xgb_negative': int((xgb[mask] < 0).sum())})

with open(HERE / 'spearman.csv', 'w', newline='') as handle:
    writer = csv.DictWriter(handle, list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
