import collections
import csv
import json
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
LAKE = {'chicago': 'chicago_open'}
NAME = {'keyword': 'Keyword', 'single_column': 'SingleColumnOverlap', 'multi_column': 'MultiColumnOverlap',
        'correlation': 'Correlation', 'natural_language': 'NaturalLanguage_judge'}
SHORT = {'keyword': 'KW', 'single_column': 'SC', 'multi_column': 'MC', 'correlation': 'C', 'natural_language': 'NL'}
COSTS = {lake: json.loads((ROOT / 'config' / 'costs' / f'{lake}.json').read_text())
         for lake in ['adventure_works', 'chembl', 'chicago_open']}


def has_nl(node):
    return node['fill']['seeker'] == 'natural_language' if 'fill' in node else any(map(has_nl, node['inputs']))


def cost(node, costs):
    if 'fill' in node:
        return costs[NAME[node['fill']['seeker']]]
    inputs = [cost(child, costs) for child in node['inputs']]
    return inputs[1] if node['kind'] == 'difference' else min(inputs) if node['kind'] == 'intersection' else sum(inputs)


def intersections(node):
    if 'inputs' in node:
        if node['kind'] == 'intersection':
            yield node
        for child in node['inputs']:
            yield from intersections(child)


def short(node):
    if 'fill' in node:
        seeker = node['fill']['seeker']
        text = f", {node['fill']['description']['value']!r}" if seeker == 'natural_language' else ''
        return f"{SHORT[seeker]}(k={node.get('k')}{text})"
    return f"{node['kind']}({', '.join(map(short, node['inputs']))})"


def scan(plans):
    counts, ties = collections.Counter(), collections.Counter()
    for run, lake, case, root in plans:
        counts['plans'] += 1
        for node in intersections(root):
            counts['intersections'] += 1
            nl_inputs = sum(map(has_nl, node['inputs']))
            counts['with NL input'] += nl_inputs > 0
            counts['with >=2 NL inputs'] += nl_inputs > 1
            groups = collections.defaultdict(list)
            for child in node['inputs']:
                groups[cost(child, COSTS[lake])].append(child)
            if any(len(group) > 1 and any(map(has_nl, group)) for group in groups.values()):
                counts['cost tie involving NL'] += 1
                ties[(lake, case, run, ' ∩ '.join(map(short, node['inputs'])))] += 1
    return counts, ties


agent = [(path.parent.parent.name.rsplit('-r', 1)[0], LAKE.get(row['lake'], row['lake']), row['case'], row['plan']['root'])
         for path in ROOT.glob('results/Agent/**/rows/*.json') for row in json.loads(path.read_text()) if row.get('plan')]
intended = [('intended', LAKE.get(entry['lake'], entry['lake']), entry['gold_plan_id'], entry['gold_plan']['root'])
            for path in ROOT.glob('evaluation/agent/cases/*_tiers.yaml') for entry in yaml.safe_load(path.read_text())]

with open(HERE / 'ties.csv', 'w', newline='') as handle:
    writer = csv.writer(handle)
    writer.writerow(['plans', 'count', 'intersections', 'with NL input', 'with >=2 NL inputs', 'cost tie involving NL'])
    for label, plans in (('agent rows', agent), ('intended plans', intended)):
        counts, ties = scan(plans)
        if label == 'agent rows':
            agent_ties = ties
        writer.writerow([label, counts['plans'], counts['intersections'], counts['with NL input'],
                         counts['with >=2 NL inputs'], counts['cost tie involving NL']])

with open(HERE / 'tie_cases.csv', 'w', newline='') as handle:
    writer = csv.writer(handle)
    writer.writerow(['lake', 'case', 'condition', 'intersection', 'rows'])
    writer.writerows([*key, rows] for key, rows in sorted(agent_ties.items()))
