import copy
import json
import random
import statistics
import sys
import time
from pathlib import Path

from scipy.optimize import nnls

from benchmarks.nl_integration.common import (NLMeter, Records, markdown_table, nl_config, questions, run_lakes,
                                              write_summary)
from benchmarks.nl_integration.metrics import distinct, fit, regret
from benchmarks.nl_integration.runtime import INNER_K, NL_SEEKER, TOKEN_SEEKERS, Sampler, Worker, input_shape
from scripts.train_ml_optimizer import build_correlation, build_mc, build_sc
from src.DBHandler import DBHandler
from src.Operators.Combiners import Intersection
from src.Operators.Seekers import Keyword, NaturalLanguage
from src.Optimization import changeOptimizationOrder

from typing import Optional

ORDERS = ('NL last', 'NL first')
LISTED_INPUTS = 10


def d4_run(results: Path, lake: str) -> Path:
    for run in sorted(results.glob('*__runtime'), reverse=True):
        if (run / lake / f'runtime_{lake}.jsonl').is_file():
            return run
    raise FileNotFoundError(f'no sealed D4 run with {lake} under {results}; run make eval-d4 first')


def _lines(path: Path) -> list:
    with open(path) as f:
        return [json.loads(line) for line in f]


def d4_model(run: Path, lake: str, inner_k: int) -> dict:
    nl = [part for part in _lines(run / lake / f'nl_runs_runtime_{lake}.jsonl')
          if part['arm'] == NL_SEEKER]
    if not nl:
        raise ValueError(f'{run} has no judge-on NL runs on {lake}')
    inputs = [row for row in _lines(run / lake / f'runtime_{lake}.jsonl')
              if row['seeker'] in TOKEN_SEEKERS and row.get('finished', True)]
    present = [name for name in TOKEN_SEEKERS if any(row['seeker'] == name for row in inputs)]
    if not present:
        raise ValueError(f'{run} has no token seeker inputs ({", ".join(TOKEN_SEEKERS)}) on {lake}')
    if any(row.get('inner_k') != inner_k for row in inputs) or any('pooled_characters' not in part for part in nl):
        raise ValueError(f'{run} has no token seekers timed at inner k {inner_k} or no pooled characters on {lake}')
    sizes = {part['judge_calls'] for part in nl}
    if len(sizes) < 2:
        raise ValueError(f'{run} has judge-on NL runs at one pool size ({", ".join(map(str, sizes))}) on {lake}; '
                         'the judge\'s time per call and per character need two, so run D4 over two k x n cells')
    (call_s, char_s), _ = nnls([[part['judge_calls'], part['pooled_characters']] for part in nl],
                               [part['judge'] for part in nl])
    return {'d4_run': str(run), 'nl_runs': len(nl),
            'embedding_s': statistics.median(part['embedding'] for part in nl),
            'sql_s': statistics.median(part['retrieval'] for part in nl),
            'judge_call_s': float(call_s), 'judge_char_s': float(char_s),
            'partner_s': {name: statistics.median(row['inner_runtime_s'] for row in inputs if row['seeker'] == name)
                          for name in present},
            'no_inputs': [name for name in TOKEN_SEEKERS if name not in present],
            'inputs': inputs}


def nl_seconds(model: dict, pool: dict) -> float:
    return (model['embedding_s'] + model['sql_s'] + pool['pool'] * model['judge_call_s']
            + pool['characters'] * model['judge_char_s'])


def run_lake(args, db: DBHandler) -> None:
    run = Path(args.d4) if args.d4 else d4_run(args.results, args.lake)
    model = d4_model(run, args.lake, args.inner_k)
    records = Records(args, 'conditional')
    records.write({key: value for key, value in model.items() if key != 'inputs'})
    meter = NLMeter(db, args.out / f'nl_runs_conditional_{args.lake}.jsonl')
    rng = random.Random(args.seed)
    n = nl_config(db).n
    documents = f'"{nl_config(db).documents_table}"'

    def pool(k: int, tables: Optional[list] = None) -> dict:
        where = f'WHERE tableid IN ({db.create_sql_list_numeric(tables)})' if tables else ''
        allowed, characters = (0, 0) if tables == [] else db.execute_and_fetchall(
            f'SELECT count(*), coalesce(sum(length(text)), 0) FROM {documents} {where}')[0]
        size = min(k * n, allowed)
        return {'pool': size, 'allowed': allowed, 'characters': characters * size / allowed if allowed else 0}

    sampler = Sampler(db, rng)
    numeric = db.execute_and_fetchall('SELECT DISTINCT TableId, ColumnId FROM AllTables WHERE Quadrant IS NOT NULL')

    def keyword(answer):
        values = [row[0] for row in db.execute_and_fetchall(
            f"SELECT DISTINCT CellValue FROM AllTables WHERE TableId = {answer} AND CellValue <> '' ORDER BY 1")]
        return Keyword(rng.sample(values, min(rng.randint(1, 5), len(values)))) if values else None

    def correlation(answer):
        pairs = [pair for pair in numeric if pair[0] == answer]
        return build_correlation(sampler, pairs) if pairs else None

    builders = {'KW': keyword, 'SC': lambda answer: build_sc(sampler), 'MC': lambda answer: build_mc(sampler),
                'C': correlation}

    def run_order(order, first, last, question, partner):
        legs = {}
        run_first = first.run

        def timed_first(additionals=''):
            start = time.perf_counter()
            legs['tables'] = run_first(additionals)
            legs['first_s'] = time.perf_counter() - start
            return legs['tables']

        first.run = timed_first
        changeOptimizationOrder(first, last)
        combiner = Intersection(args.k)
        combiner.set_inputs([first, last])
        start = time.perf_counter()
        tables = combiner.run()
        seconds = time.perf_counter() - start
        parts = meter.write(arm=order, qid=question['qid'], partner=partner)
        nl_s = parts[0]['total'] if parts else 0.0
        return {'seconds': seconds, 'first_s': legs['first_s'], 'nl_s': nl_s,
                'partner_s': legs['first_s'] if order.startswith('NL last') else seconds - legs['first_s'],
                'pooled': parts[0]['pooled_documents'] if parts else 0,
                'pooled_characters': parts[0]['pooled_characters'] if parts else 0, 'first_tables': legs['tables'],
                'tables': tables}

    first = pool(args.nl_inner_k)
    first['nl_s'] = first['seconds'] = nl_seconds(model, first)

    def predict(partner_s: float, partner_tables: list) -> dict:
        last = pool(args.k, partner_tables)
        last['nl_s'] = nl_seconds(model, last) if partner_tables else 0.0
        last['seconds'] = partner_s + last['nl_s']
        return {'NL last': last, 'NL first': first}

    trial = Worker(db.config_path, [args.inner_k], int(args.memory_limit_gb * 2 ** 30), args.time_limit_min * 60)
    NaturalLanguage('warm-up', k=1, n=1).run()
    meter.write(arm='warm-up')
    asked = questions(db, args.lake)
    rng.shuffle(asked)
    for partner in model['partner_s']:
        measured, limited = 0, 0
        for question in asked:
            if args.inputs in (measured, limited):
                break
            sampler.anchor = question['answer']
            seeker = builders[partner](question['answer'])
            if seeker is None:
                records.write({'qid': question['qid'], 'partner': partner, 'skipped': 'no input'})
                continue
            seeker.k = args.inner_k
            limit = trial.time(copy.deepcopy(seeker))
            if isinstance(limit, str):
                records.write({'qid': question['qid'], 'partner': partner, 'skipped': f'past the {limit} limit',
                               **input_shape(seeker, db, question['answer'])})
                limited += 1
                continue
            measured += 1
            copy.deepcopy(seeker).run()
            orders = {order: run_order(order, *inputs(), question, partner) for order, inputs in (
                ('NL last', lambda: (copy.deepcopy(seeker), NaturalLanguage(question['question']))),
                ('NL first', lambda: (NaturalLanguage(question['question'], k=args.nl_inner_k), copy.deepcopy(seeker))),
                ('NL last, repeat', lambda: (copy.deepcopy(seeker), NaturalLanguage(question['question']))))}
            partner_tables = distinct(orders['NL last']['first_tables'])
            predicted = predict(model['partner_s'][partner], partner_tables)
            records.write({'qid': question['qid'], 'answer': question['answer'], 'partner': partner,
                           **input_shape(seeker, db, question['answer']), 'partner_tables': len(partner_tables),
                           'partner_documents': predicted['NL last']['allowed'], 'orders': orders,
                           'predicted': predicted,
                           'regret': regret({order: orders[order]['seconds'] for order in ORDERS}),
                           'noise': abs(orders['NL last']['seconds'] - orders['NL last, repeat']['seconds'])
                           / min(orders[order]['seconds'] for order in ORDERS)})
        print(f'{partner}: {measured} pairs, {limited} past a limit')
    trial.close()

    for row in model['inputs']:
        row['predicted'] = predict(row['inner_runtime_s'], row['inner_tables'])
    write_summary(args, 'conditional', summarize(records.read(), model, args))


def summarize(rows: list, model: dict, args) -> str:
    pairs = [row for row in rows if 'orders' in row]
    no_input = sum(row.get('skipped') == 'no input' for row in rows)
    limited = sum('skipped' in row for row in rows) - no_input
    text = (f'# NL\'s conditional cost on {args.lake}\n\n{len(pairs)} pairs; {no_input} times a question\'s answer table '
            f'gave a partner no input; {limited} partners passed the memory or time limit of a trial run in a child '
            f'process ({args.memory_limit_gb:g} GiB, {args.time_limit_min:g} min) and were skipped. Each pair runs '
            f'Intersection(k {args.k}) in both orders with the judge on: NL '
            f'last runs the partner at inner k {args.inner_k}, NL first runs NL at inner k {args.nl_inner_k}. Each order '
            f'is timed around Intersection.run() after one untimed run of the partner, in the sequence NL last, NL '
            f'first, NL last again; the repeat is the noise floor. Seed {args.seed}.\n\n')

    text += (f'## Model\n\nFrom D4 ({model["d4_run"]}, {model["nl_runs"]} judge-on NL runs, prefix cache off): '
             't(NL) = embedding + SQL + the pool\'s judge calls x time per call + its characters x time per judged '
             'character, both fitted by least squares on those runs with neither below zero, where the pool is k x n '
             'capped at the NL documents the restriction allows (for NL last, those of the tables the pair\'s own '
             'partner leg returned) and a capped pool reads the allowed documents\' mean length; t(partner) is D4\'s '
             f'median for its seeker at inner k {args.inner_k}; t(partner | NL) is left out.\n\n')
    for name in model['no_inputs']:
        text += f'{name} left out: D4 measured no {name} inputs on {args.lake}.\n\n'
    for key, term in (('judge_call_s', 'call'), ('judge_char_s', 'judged character')):
        if not model[key]:
            text += (f'Least squares put the time per {term} below zero, so it is held at zero and the judge term is '
                     'the other one alone.\n\n')
    text += markdown_table(['Embedding ms', 'SQL ms', 'ms per judge call', 'ms per judged character']
                           + [f't({name}) ms' for name in model['partner_s']],
                           [[model['embedding_s'] * 1000, model['sql_s'] * 1000, model['judge_call_s'] * 1000,
                             model['judge_char_s'] * 1000]
                            + [seconds * 1000 for seconds in model['partner_s'].values()]])

    if pairs:
        nl_legs = [(row['predicted'][order]['nl_s'], row['orders'][order]['nl_s']) for row in pairs for order in ORDERS
                   if order == 'NL first' or row['partner_tables']]
        totals = [(row['predicted'][order]['seconds'], row['orders'][order]['seconds'])
                  for row in pairs for order in ORDERS]
        partners = [(model['partner_s'][row['partner']], row['orders']['NL last']['partner_s']) for row in pairs]
        characters = [(row['predicted'][order]['characters'], row['orders'][order]['pooled_characters'])
                      for row in pairs for order in ORDERS if row['orders'][order]['pooled']]
        fits = [[name, len(values)] + list(fit(*zip(*values)).values())
                for name, values in (('NL leg', nl_legs), ('Partner first', partners), ('Order total', totals))
                if values]
        if characters:
            measured = fit(*zip(*characters))
            fits.append(['Pooled characters', len(characters), measured['r2'], measured['median_pct_error'], None])
        picked = sum(_best(row['predicted']) == _best(row['orders']) for row in pairs)
        text += ('\n## Held-out fit\n\nOn the pairs, which D4 never ran. The NL leg is measured over '
                 'Retriever.retrieve(); the partner first is its leg in NL last; the order total over '
                 f'Intersection.run(). The model picks the faster order in {picked} of {len(pairs)} pairs.\n\n')
        text += markdown_table(['Prediction', 'Runs', 'R²', 'Median % error', 'Mean ms error'], fits)

        text += ('\n## Regret\n\nThe extra runtime of an order over the pair\'s best order, normalized by the best '
                 'order\'s runtime. Random order is the mean of both orders\' regrets; the model\'s choice runs the '
                 'order it predicts faster. The noise floor is NL last\'s two runs apart, on the same scale.\n\n')
        table = []
        for partner in TOKEN_SEEKERS + ['All']:
            group = [row for row in pairs if partner in ('All', row['partner'])]
            if not group:
                continue
            last = [row['regret']['NL last'] for row in group]
            table.append([partner, len(group), sum(value == 0 for value in last), statistics.mean(last),
                          statistics.median(last), max(last),
                          statistics.mean(statistics.mean(row['regret'].values()) for row in group),
                          statistics.mean(row['regret'][_best(row['predicted'])] for row in group),
                          statistics.median(row['noise'] for row in group)])
        text += markdown_table(['Partner', 'Pairs', 'NL last best', 'NL always last: mean', 'median', 'max',
                                'Random order: mean', 'Model\'s choice: mean', 'Noise floor: median'], table)

        slower = [row for row in pairs if row['regret']['NL last'] > 0]
        beyond = sum(row['regret']['NL last'] > row['noise'] for row in slower)
        text += (f'\n## Pairs where NL first was faster\n\n{len(slower)} of {len(pairs)}, {beyond} of them by more '
                 'than their noise floor.\n\n')
        if slower:
            text += markdown_table(
                ['Question', 'Partner', 'Values', 'Columns', 'Candidate rows', 'Partner tables', 'Partner documents',
                 'NL last ms', 'NL first ms'],
                [[row['qid'], row['partner'], row['values'], row['columns'], row['candidate_rows'],
                  row['partner_tables'], row['partner_documents'], row['orders']['NL last']['seconds'] * 1000,
                  row['orders']['NL first']['seconds'] * 1000] for row in slower])

    risky = sorted((row for row in model['inputs'] if _best(row['predicted']) == 'NL first'),
                   key=lambda row: row['predicted']['NL first']['seconds'] - row['predicted']['NL last']['seconds'])
    counts = ', '.join(f'{name} {sum(row["seeker"] == name for row in risky)} of '
                       f'{sum(row["seeker"] == name for row in model["inputs"])}' for name in model['partner_s'])
    text += ('\n## D4\'s inputs the model sends NL first\n\nEach D4 input as a partner: its own median at inner k '
             f'{args.inner_k}, then NL restricted to the tables it returned there. Per seeker: {counts}.\n\n')
    if risky:
        text += markdown_table(['Seeker', f'Median ms at k {args.inner_k}', 'Values', 'Columns', 'Source table',
                                'Candidate rows', 'Partner tables', 'Partner documents', 'Predicted NL last ms',
                                'Predicted NL first ms'],
                               [[row['seeker'], row['inner_runtime_s'] * 1000]
                                + [row[key] for key in ('values', 'columns', 'source_table', 'candidate_rows')]
                                + [len(row['inner_tables']), row['predicted']['NL last']['allowed']]
                                + [row['predicted'][order]['seconds'] * 1000 for order in ORDERS]
                                for row in risky[:LISTED_INPUTS]])
    return text


def _best(seconds: dict) -> str:
    return min(ORDERS, key=lambda order: seconds[order]['seconds'])


def main(argv=None) -> int:
    return run_lakes('conditional', run_lake, argv, inputs=10, k=10, inner_k=INNER_K, nl_inner_k=50, d4='',
                     memory_limit_gb=50.0, time_limit_min=10.0)


if __name__ == '__main__':
    sys.exit(main())
