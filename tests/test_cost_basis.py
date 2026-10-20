import json
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import pandas as pd

from src.Operators.Combiners import Intersection
from src.Operators.OperatorBase import Operator
from src.Operators.Seekers import Correlation, Keyword, MultiColumnOverlap, NaturalLanguage, SingleColumnOverlap

LAKES = ['adventure_works', 'chembl', 'chicago_open', 'fetaqa', 'public_bi']


def _seekers():
    return {'Keyword': Keyword(['red']), 'SingleColumnOverlap': SingleColumnOverlap(['red']),
            'Correlation': Correlation(['a', 'b'], [1, 2]),
            'MultiColumnOverlap': MultiColumnOverlap(pd.DataFrame({'a': ['red'], 'b': ['paris']})),
            'NaturalLanguage_judge': NaturalLanguage('drug names'),
            'NaturalLanguage_nojudge': NaturalLanguage('drug names', rerank=False)}


def _execution_order(*inputs):
    calls = []
    for operator in inputs:
        operator.run = lambda additionals='', operator=operator: calls.append(operator) or [1]
        operator.create_sql_query = lambda db, additionals='', operator=operator: calls.append(operator) or ''
    intersection = Intersection(10)
    intersection.set_inputs(list(inputs))
    intersection.create_sql_query(Operator.DB)
    return calls


def test_priority_keeps_the_fixed_costs():
    ini = (ROOT / 'config' / 'chembl.ini').read_text()
    for basis in ('[Optimizer]\ncost_basis=priority\n', ''):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'chembl.ini'
            path.write_text(ini.replace('[Optimizer]\ncost_basis=measured\n', basis))
            Operator.DB.load_config(path)
            costs = {name: seeker.cost() for name, seeker in _seekers().items()}
            assert costs == {'Keyword': 3, 'SingleColumnOverlap': 4, 'Correlation': 6, 'MultiColumnOverlap': 10,
                             'NaturalLanguage_judge': 11, 'NaturalLanguage_nojudge': 11}


def test_measured_costs_come_from_the_lake_file():
    for lake in LAKES:
        Operator.DB.load_config(ROOT / 'config' / f'{lake}.ini')
        expected = json.loads((ROOT / 'config' / 'costs' / f'{lake}.json').read_text())
        for name, seeker in _seekers().items():
            assert seeker.cost() == expected[name], (lake, name)


def test_sc_runs_before_nl_with_judge_on_every_lake():
    for lake in LAKES:
        Operator.DB.load_config(ROOT / 'config' / f'{lake}.ini')
        sc, nl = SingleColumnOverlap(['red']), NaturalLanguage('drug names')
        assert _execution_order(nl, sc) == [sc, nl], lake


if __name__ == '__main__':
    for name, test in list(globals().items()):
        if name.startswith('test_'):
            test()
            print(f'ok {name}')
