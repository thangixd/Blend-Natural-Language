import math
import multiprocessing
import random
import statistics
import sys
import time
from itertools import product
from pathlib import Path

from benchmarks.nl_integration.common import (NLMeter, Records, markdown_table, nl_config, questions, run_lakes,
                                              write_summary)
from benchmarks.nl_integration.metrics import distinct, measured_basis, rule_order_breaks
from scripts.train_ml_optimizer import TableSampler, build_correlation, build_keyword, build_mc, build_sc
from src.DBHandler import DBHandler
from src.Operators.OperatorBase import Operator
from src.Operators.Seekers import NaturalLanguage

from typing import List, Union

PLAN_K = 10
INNER_K = 100
NL_SEEKER = 'NL judge on'
RULE_COST = {'KW': 3, 'SC': 4, 'C': 6, 'MC': 10, NL_SEEKER: 11}
TOKEN_SEEKERS = sorted(['KW', 'SC', 'C', 'MC'], key=RULE_COST.get)


def timed(run):
    start = time.perf_counter()
    result = run()
    return time.perf_counter() - start, result


def p90(runtimes: list) -> float:
    return sorted(runtimes)[math.ceil(0.9 * len(runtimes)) - 1]


class Sampler(TableSampler):
    source = None
    anchor = None

    def table(self, table_id: int):
        self.source = table_id
        return super().table(table_id)

    def random_table(self):
        return super().random_table() if self.anchor is None else self.table(self.anchor)


def _serve(config_path: str, ks: List[int], connection) -> None:
    Operator.DB.load_config(config_path)
    for seeker in iter(connection.recv, None):
        for k in ks:
            seeker.k = k
            connection.send(timed(seeker.run))


def _rss_bytes(pid: int) -> int:
    try:
        with open(f'/proc/{pid}/status') as f:
            return next(int(line.split()[1]) * 1024 for line in f if line.startswith('VmRSS:'))
    except (FileNotFoundError, StopIteration):
        return 0


class Worker:
    def __init__(self, config_path: Path, ks: List[int], limit_bytes: int, limit_s: float) -> None:
        self.args, self.ks = (str(config_path), ks), ks
        self.limit_bytes, self.limit_s = limit_bytes, limit_s
        self._start()

    def _start(self) -> None:
        context = multiprocessing.get_context('spawn')
        self.connection, child = context.Pipe()
        self.process = context.Process(target=_serve, args=(*self.args, child), daemon=True)
        self.process.start()
        child.close()

    def time(self, seeker) -> Union[list, str]:
        self.connection.send(seeker)
        runs = []
        for _ in self.ks:
            deadline = time.monotonic() + self.limit_s
            while True:
                limit = ('memory' if _rss_bytes(self.process.pid) > self.limit_bytes
                         else 'time' if time.monotonic() > deadline else None)
                if limit:
                    self.process.kill()
                    self.process.join()
                    self._start()
                    return limit
                if self.connection.poll(1):
                    break
            runs.append(self.connection.recv())
        return runs

    def close(self) -> None:
        self.connection.send(None)
        self.process.join()


def input_shape(seeker, db: DBHandler, source) -> dict:
    values, candidate_rows, columns = seeker._features(db)
    return {'values': values, 'columns': columns, 'source_table': source, 'candidate_rows': candidate_rows}


def run_lake(args, db: DBHandler) -> None:
    records = Records(args, 'runtime')
    rng = random.Random(args.seed)

    sampler = Sampler(db, rng)
    numeric = db.execute_and_fetchall('SELECT DISTINCT TableId, ColumnId FROM AllTables WHERE Quadrant IS NOT NULL')
    builders = {'KW': lambda: build_keyword(db, rng), 'SC': lambda: build_sc(sampler),
                'MC': lambda: build_mc(sampler),
                'C': lambda: build_correlation(sampler, numeric) if numeric else None}
    worker = Worker(db.config_path, [PLAN_K] * (args.repeats + 1) + [INNER_K] * args.repeats,
                    int(args.memory_limit_gb * 2 ** 30), args.time_limit_min * 60)
    for name, build in builders.items():
        measured, attempts, unfinished = 0, 0, 0
        while measured < args.inputs and attempts < 3 * args.inputs:
            attempts += 1
            sampler.source = None
            seeker = build()
            if seeker is None:
                continue
            timings = worker.time(seeker)
            shape = input_shape(seeker, db, sampler.source)
            measured += 1
            if isinstance(timings, str):
                unfinished += 1
                records.write({'seeker': name, 'finished': False, 'limit': timings,
                               'memory_limit_bytes': worker.limit_bytes, 'time_limit_s': worker.limit_s, **shape})
                continue
            runs, inner_runs = timings[1:args.repeats + 1], timings[args.repeats + 1:]
            seconds = [s for s, _ in runs]
            inner_seconds = [s for s, _ in inner_runs]
            records.write({'seeker': name, 'finished': True, 'runs_s': seconds,
                           'runtime_s': statistics.median(seconds), 'tables': len(distinct(runs[-1][1])),
                           'inner_k': INNER_K, 'inner_runs_s': inner_seconds,
                           'inner_runtime_s': statistics.median(inner_seconds),
                           'inner_tables': distinct(inner_runs[-1][1]), **shape})
        print(f'{name}: {measured} inputs, {unfinished} past a limit')
    worker.close()

    meter = NLMeter(db, args.out / f'nl_runs_runtime_{args.lake}.jsonl')

    def nl_run(question, k, n, rerank, arm):
        seconds, _ = timed(NaturalLanguage(question['question'], k=k, n=n, rerank=rerank).run)
        (part,) = meter.write(arm=arm, qid=question['qid'], run_s=seconds)
        return part

    def record(arm, question, k, n, parts):
        runs = [part['run_s'] for part in parts]
        records.write({'seeker': arm, 'qid': question['qid'], 'k': k, 'n': n,
                       'runs_s': runs, 'runtime_s': statistics.median(runs), 'tables': len(parts[0]['tables']),
                       'parts': parts})

    warm_up, *sample = random.Random(args.seed).sample(questions(db, args.lake), args.questions + 1)
    cells = list(product(args.k, args.n))
    nl_run(warm_up, 1, 1, True, 'warm-up')
    for question, (k, n) in product(sample, cells):
        record(NL_SEEKER, question, k, n, [nl_run(question, k, n, True, NL_SEEKER)])
        record('NL judge off', question, k, n, [nl_run(question, k, n, False, 'NL judge off')
                                               for _ in range(args.repeats)])
    print(f'NL: {len(sample)} questions')

    write_summary(args, 'runtime', summarize(records.read(), args, nl_config(db).n))


def summarize(rows: list, args, config_n: int) -> str:
    past_limit = {(name, limit): sum(row['seeker'] == name and row.get('limit') == limit for row in rows)
                  for name in RULE_COST for limit in ('memory', 'time')}
    rows = [row for row in rows if row.get('finished', True)]
    basis_rows = {name: [row['runtime_s'] for row in rows if row['seeker'] == name
                         and (name in TOKEN_SEEKERS or (row['k'], row['n']) == (PLAN_K, config_n))]
                  for name in RULE_COST}
    basis_rows = {name: runtimes for name, runtimes in basis_rows.items() if runtimes}
    medians = {name: statistics.median(runtimes) for name, runtimes in basis_rows.items()}
    basis = measured_basis(medians) if medians else {}

    inner = {name: statistics.median(row['inner_runtime_s'] for row in rows if row['seeker'] == name)
             for name in TOKEN_SEEKERS if name in basis_rows}
    table = [[name, len(runtimes), past_limit[name, 'memory'], past_limit[name, 'time'], medians[name] * 1000,
              p90(runtimes) * 1000, inner[name] * 1000 if name in inner else None, RULE_COST[name], basis[name]]
             for name, runtimes in basis_rows.items()]
    text = (f'# Seeker runtimes on {args.lake}\n\nPrefix cache off. Token seekers at k {PLAN_K} and {INNER_K}, '
            f'{args.repeats} repeats after a warm-up; NL judge on at k {PLAN_K}, n {config_n}. Every seeker is '
            f'timed around seeker.run(); medians are plain medians over the inputs that finished. '
            f'A token-seeker input runs in a child process killed past {args.memory_limit_gb:g} GB RSS or when one '
            f'run takes longer than {args.time_limit_min:g} min; such an input is counted, not timed. '
            f'Seed {args.seed}.\n\n')
    text += markdown_table(['Seeker', 'Inputs', 'Past memory limit', 'Past time limit', 'Median ms', 'p90 ms',
                            f'Median ms at k {INNER_K}', 'Rule cost', 'Measured basis'], table)

    order = TOKEN_SEEKERS + [NL_SEEKER]
    breaks = rule_order_breaks(medians, order)
    verdict = 'holds' if not breaks else 'breaks: ' + ', '.join(f'{a} above {b}' for a, b in breaks)
    missing = [name for name in order if name not in medians]
    text += (f'\nRule order {" < ".join(name for name in order if name in medians)}: {verdict}.'
             + (f' Not measured: {", ".join(missing)}.' if missing else '') + '\n')
    for name in TOKEN_SEEKERS:
        if name not in medians:
            text += (f'\nD4 measured no {name} inputs on {args.lake}: '
                     + (f'all {past_limit[name, "memory"] + past_limit[name, "time"]} passed a limit.\n'
                        if past_limit[name, 'memory'] + past_limit[name, 'time'] else 'its sampler found none.\n'))

    stages = []
    for arm, k, n in sorted({(row['seeker'], row['k'], row['n']) for row in rows if 'parts' in row}):
        parts = [part for row in rows if (row['seeker'], row.get('k'), row.get('n')) == (arm, k, n)
                 for part in row['parts']]
        calls = sum(part['judge_calls'] for part in parts)
        characters = sum(part['pooled_characters'] for part in parts)
        stages.append([arm, k, n, len(parts)]
                      + [statistics.median(part[key] for part in parts) * 1000
                         for key in ('run_s', 'total', 'embedding', 'retrieval', 'judge')]
                      + [p90([part['run_s'] for part in parts]) * 1000]
                      + [statistics.median(part['run_s'] - part['total'] for part in parts) * 1000,
                         statistics.mean(part['pooled_documents'] for part in parts),
                         statistics.mean(part['pooled_characters'] for part in parts),
                         sum(part['judge'] for part in parts) / calls * 1000 if calls else None,
                         sum(part['judge'] for part in parts) / characters * 1000 if calls and characters else None])
    text += ('\n## NL stages\n\nMedians per run, and the p90 of seeker.run(). Retrieve is the meter\'s window, '
             'split into embedding, retrieval (the SQL plus Python around it) and judge; outside is seeker.run() '
             'minus retrieve: the ini read and the ranked-id SQL.\n\n')
    text += markdown_table(['Arm', 'k', 'n', 'Runs', 'seeker.run ms', 'Retrieve ms', 'Embedding ms', 'Retrieval ms',
                            'Judge ms', 'p90 seeker.run ms', 'Outside ms', 'Mean pooled docs', 'Mean pooled characters',
                            'ms per judge call', 'ms per judged character'],
                           stages)
    return text


def main(argv=None) -> int:
    return run_lakes('runtime', run_lake, argv, inputs=200, questions=100, repeats=3, memory_limit_gb=50.0,
                     time_limit_min=10.0, k=[5, 10, 20], n=[3, 5, 10])


if __name__ == '__main__':
    sys.exit(main())
