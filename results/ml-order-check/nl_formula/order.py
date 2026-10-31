"""Order of every intersection with an NL input in the agent rows and intended plans, sorted by
(cost(), ml_cost(db)) as Intersection and Compiler.ranking_input do, with the old NL model and with the formula."""
import collections
import csv
import json
import os
import sys
from pathlib import Path

import yaml
from xgboost import XGBRegressor

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from src.Agent.Compiler import COMBINERS, _ordered, seeker, walk
from src.Agent.Plans import Part, SearchPlan
from src.Agent.Profile import QueryTable
from src.DBHandler import DBHandler
from src.NLSeeker.Config import NLSeekerConfig
from src.Operators.Seekers import NaturalLanguage

LAKES = {'chembl': 'config/chembl.ini', 'chicago': 'config/chicago_open.ini',
         'adventure_works': 'config/adventure_works.ini'}

model = XGBRegressor()
model.load_model(HERE / 'NaturalLanguage_model.json')
formula = NaturalLanguage.ml_cost


def xgb(self, db):
    n = self.n if self.n is not None else NLSeekerConfig.load(db.config_path).n
    return float(model.predict([[len(self.query.split()), self.k * n, int(self.rerank)]])[0])


def has_nl(node):
    return node.fill.seeker == 'natural_language' if isinstance(node, Part) else any(map(has_nl, node.inputs))


def nl_intersections(plan):
    return [node for node in walk(plan.root)
            if not isinstance(node, Part) and node.kind == 'intersection' and any(map(has_nl, node.inputs))]


entries = {entry['gold_plan_id']: entry for path in ROOT.glob('evaluation/agent/cases/*_tiers.yaml')
           for entry in yaml.safe_load(path.read_text())}
dbs, tables = {}, {}

plans = collections.Counter()
for path in ROOT.glob('results/Agent/**/rows/*.json'):
    for row in json.loads(path.read_text()):
        if row.get('plan'):
            plans[(row['lake'], row['gold_plan_id'], json.dumps(row['plan'], sort_keys=True))] += 1
for gold, entry in entries.items():
    plan = {'version': 1, 'k': {'value': 10, 'status': 'certain'}, 'root': entry['gold_plan']['root']}
    plans[(entry['lake'], gold, json.dumps(plan))] += 0

counts, changed, pairs = collections.Counter(), [], set()
for (lake, gold, text), rows in plans.items():
    plan = SearchPlan.model_validate(json.loads(text))
    nodes = nl_intersections(plan)
    if not nodes:
        continue
    if lake not in dbs:
        dbs[lake] = DBHandler()
        dbs[lake].load_config(LAKES[lake])
    if gold not in tables:
        tables[gold] = QueryTable.load(entries[gold]['query_table'])
    db, operators = dbs[lake], {}
    for node in walk(plan.root):
        if isinstance(node, Part):
            operators[node.id] = seeker(node, node.k or 1, tables[gold])[0]
            operators[node.id].DB = db
        else:
            operators[node.id] = COMBINERS[node.kind]()
            operators[node.id].set_inputs([operators[child.id] for child in _ordered(node)])
    kind = 'agent' if rows else 'intended'
    for node in nodes:
        orders = []
        for cost in (xgb, formula):
            NaturalLanguage.ml_cost = cost
            orders.append([child.id for child in sorted(
                _ordered(node), key=lambda child: (operators[child.id].cost(), operators[child.id].ml_cost(db)))])
        NaturalLanguage.ml_cost = formula
        counts[(kind, 'rows')] += rows
        counts[(kind, 'plans')] += 1
        if orders[0] != orders[1]:
            counts[(kind, 'changed rows')] += rows
            counts[(kind, 'changed plans')] += 1
            changed.append([lake, gold, rows, node.id, ' < '.join(orders[0]), ' < '.join(orders[1])])
        nl_parts = [child for child in node.inputs if isinstance(child, Part) and child.fill.seeker == 'natural_language']
        if len(nl_parts) > 1:
            for part in nl_parts:
                operator = operators[part.id]
                pairs.add((lake, gold, operator.query, operator.k, round(xgb(operator, db), 4), round(formula(operator, db), 4)))

with open(HERE / 'order_changes.csv', 'w', newline='') as handle:
    writer = csv.writer(handle)
    writer.writerow(['lake', 'gold_plan_id', 'rows', 'intersection', 'old order', 'new order'])
    writer.writerows(changed)

with open(HERE / 'order_counts.csv', 'w', newline='') as handle:
    writer = csv.writer(handle)
    writer.writerow(['plans', 'rows', 'distinct plans', 'changed rows', 'changed distinct plans'])
    for kind in ('agent', 'intended'):
        writer.writerow([kind, *(counts[(kind, key)] for key in ('rows', 'plans', 'changed rows', 'changed plans'))])

with open(HERE / 'nl_pairs.csv', 'w', newline='') as handle:
    writer = csv.writer(handle)
    writer.writerow(['lake', 'gold_plan_id', 'query', 'k', 'ml_cost_xgb_s', 'ml_cost_formula_s'])
    writer.writerows(sorted(pairs))
