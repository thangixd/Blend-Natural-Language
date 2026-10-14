import csv
import json
import statistics
import sys
from collections import Counter, defaultdict
from itertools import zip_longest
from pathlib import Path

import yaml

from evaluation.nlseeker.metrics import latency_stats
from evaluation.nlseeker.prepare import sha256_file
from evaluation.nlseeker.runner import SHARED_DOCUMENTS, TEXTS

from typing import Callable, List, Optional, Tuple

USAGE = '''python -m evaluation.nlseeker.report <blend run> <pneuma run>
python -m evaluation.nlseeker.report parity <out dir> <blend run> <pneuma run> [<blend run> <pneuma run> ...]
python -m evaluation.nlseeker.report efficiency <out dir> <run> [<run> ...]'''


GROUP = ('dataset', 'family', 'texts', 'k', 'n', 'alpha', 'rerank')

SUMMARY_COLUMNS = ['dataset', 'family', 'texts', 'k', 'n', 'alpha', 'rerank', 'n_questions', 'hit_rate',
                   'recall', 'mrr', 'latency_mean_ms', 'latency_p50_ms', 'latency_p95_ms']


def collect_records(run_path: Path, datasets: List[str]) -> List[dict]:
    records = []
    for dataset in datasets:
        per_query = run_path / dataset / 'per_query.jsonl'
        if not per_query.is_file():
            continue
        with open(per_query) as f:
            for line in f:
                record = json.loads(line)
                record['dataset'] = dataset
                record.setdefault('texts', 'own')
                records.append(record)
    return records


def aggregate(records: List[dict]) -> List[dict]:
    groups = defaultdict(list)
    for record in records:
        groups[tuple(record[key] for key in GROUP)].append(record)

    rows = []
    for (dataset, family, texts, k, n, alpha, rerank), group in sorted(groups.items()):
        count = len(group)
        mean_ms, p50_ms, p95_ms = latency_stats([r['total'] * 1000.0 for r in group])
        rows.append({
            'dataset': dataset, 'family': family, 'texts': texts, 'k': k, 'n': n, 'alpha': alpha,
            'rerank': rerank, 'n_questions': count,
            'hit_rate': sum(r['hit'] for r in group) / count,
            'recall': sum(r['recall'] for r in group) / count,
            'mrr': sum(r['rr'] for r in group) / count,
            'latency_mean_ms': mean_ms, 'latency_p50_ms': p50_ms, 'latency_p95_ms': p95_ms,
        })
    return rows


def pair(left: List[dict], right: List[dict]) -> List[dict]:
    sides = []
    for records in (left, right):
        groups = defaultdict(dict)
        for record in records:
            groups[tuple(record[key] for key in GROUP)][record['qid']] = record
        sides.append(groups)
    first, second = sides
    for group in sorted(first.keys() | second.keys()):
        if first[group].keys() != second[group].keys():
            raise ValueError(f'{dict(zip(GROUP, group))}: {len(first[group])} vs {len(second[group])} questions, '
                             f'or different ones')

    paired = []
    for group in sorted(first):
        for qid, one in first[group].items():
            other = second[group][qid]
            paired.append({**dict(zip(GROUP, group)), 'qid': qid, 'answers': one['answers'],
                           one['backend']: one, other['backend']: other,
                           'pool_size_differs': one['pool_size'] != other['pool_size']})
    return paired


TIMINGS = ('total', 'embedding', 'retrieval', 'judge', 'embedding_calls', 'judge_calls')


def _filled(record: dict) -> bool:
    return all(isinstance(record.get(column), (int, float)) for column in TIMINGS) and record['embedding_calls'] > 0


def _count_differences(dataset: str, counts: List[dict]) -> List[list]:
    differences = []
    for file in sorted(counts[0].keys() | counts[1].keys()):
        for kind in sorted(counts[0].get(file, {}).keys() | counts[1].get(file, {}).keys()):
            one, other = (side.get(file, {}).get(kind, 0) for side in counts)
            if one != other:
                differences.append([dataset, file, kind, one, other])
    return differences


def pilot_check(left: Path, right: Path, datasets: List[str]) -> dict:
    metas = [json.loads((run / 'run_meta.json').read_text()) for run in (left, right)]
    shared_differences, own_differences, other_source = [], [], []
    own_texts = defaultdict(lambda: [0, 0])
    for dataset in datasets:
        directories = [run / dataset for run in (left, right)]
        importing = [(directory / SHARED_DOCUMENTS).is_file() for directory in directories]
        if sum(importing) != 1:
            raise ValueError(f'{dataset}: exactly one of the two runs must hold the index of the shared texts')
        importer, source = (1, 0) if importing[1] else (0, 1)
        if metas[importer]['datasets'][dataset]['shared_texts']['sha256'] != sha256_file(directories[source] / TEXTS):
            other_source.append(dataset)
        shared_differences += _count_differences(dataset, [
            json.loads((directory / (SHARED_DOCUMENTS if side == importer else 'documents.json')).read_text())
            for side, directory in enumerate(directories)])
        own_differences += _count_differences(
            dataset, [json.loads((directory / 'documents.json').read_text()) for directory in directories])
        texts = [json.loads((directory / TEXTS).read_text()) for directory in directories]
        for file in texts[0].keys() | texts[1].keys():
            sides = [side.get(file, {}) for side in texts]
            for kind in sides[0].keys() | sides[1].keys():
                for one, other in zip_longest(*(side.get(kind, []) for side in sides)):
                    own_texts[kind][0] += one != other
                    own_texts[kind][1] += 1

    records = [collect_records(run, datasets) for run in (left, right)]
    names = [side[0]['backend'] for side in records]
    paired = pair(*records)

    def shared_arm(alpha: float) -> List[dict]:
        return [p for p in paired if p['texts'] == 'shared' and p['alpha'] == alpha and p['rerank'] == 'off']

    def differing(arm: List[dict]) -> List[dict]:
        return [p for p in arm if p[names[0]]['retrieved'] != p[names[1]]['retrieved']]

    bm25, vectors = shared_arm(1.0), shared_arm(0.0)
    bm25_differing, vectors_differing = differing(bm25), differing(vectors)
    unfilled = {name: {texts: sum(not _filled(r) for r in side if r['texts'] == texts) for texts in ('own', 'shared')}
                for name, side in zip(names, records)}
    arms = [{r['texts'] for r in side} for side in records]
    unfinished = [run.name for run, meta in zip((left, right), metas) if run.name.endswith('.partial')
                  or any(meta['datasets'].get(dataset, {}).get('status') != 'complete' for dataset in datasets)]
    return {'unfinished_runs': unfinished,
            'shared_texts_from_another_run': other_source,
            'shared_document_differences': shared_differences,
            'bm25_identical': [len(bm25) - len(bm25_differing), len(bm25)],
            'bm25_differing': sorted({f"{p['dataset']}/{p['qid']}" for p in bm25_differing}),
            'vectors_identical': [len(vectors) - len(vectors_differing), len(vectors)],
            'unfilled_timings': unfilled,
            'own_document_differences': own_differences,
            'own_text_differences': dict(sorted(own_texts.items())),
            'met': {'both_ran': not unfinished,
                    'shared_documents_equal': not shared_differences and not other_source,
                    'bm25_identical': bool(bm25) and not bm25_differing,
                    'vectors_ran': bool(vectors),
                    'timings_filled': all(side == {'own', 'shared'} for side in arms)
                    and not any(count for side in unfilled.values() for count in side.values())}}


TOLERANCE = Path(__file__).with_name('parity_tolerance.yaml')
COMPONENTS = {1.0: 'bm25', 0.0: 'vector'}


def _cause(paired: dict, lookup: Callable[..., Optional[dict]]) -> Optional[str]:
    if not paired['differs']:
        return None
    if paired['texts'] == 'shared':
        return COMPONENTS.get(paired['alpha'], 'unattributed')
    if paired['rerank'] == 'on':
        judge_off = lookup(texts='own', alpha=paired['alpha'], rerank='off')
        if judge_off and not judge_off['differs']:
            return 'judge'
    bm25, vectors = (lookup(texts='shared', alpha=alpha, rerank='off') for alpha in (1.0, 0.0))
    if bm25 is None or vectors is None:
        return 'unattributed'
    if bm25['differs']:
        return 'bm25'
    return 'vector' if vectors['differs'] else 'own texts'


def _verdict(group: dict, tolerance: dict) -> Optional[bool]:
    if group['n_questions'] < tolerance['min_questions']:
        return None
    component = COMPONENTS.get(group['alpha']) if group['texts'] == 'shared' else None
    if component == 'bm25':
        return group['differing'] <= tolerance['bm25']['max_differing_lists']
    if component == 'vector':
        return group['hit_rate_difference'] >= -tolerance['vector']['max_hit_rate_drop']
    judged = group['causes'].get('judge', 0) / group['n_questions'] <= tolerance['judge']['max_share']
    return judged and group['hit_rate_difference'] >= -tolerance['main']['max_hit_rate_drop']


def parity_report(pairs: List[Tuple[Path, Path]], tolerance_path: Path = TOLERANCE) -> Tuple[dict, List[dict]]:
    tolerance_sha = sha256_file(tolerance_path)
    runs = [run for left_right in pairs for run in left_right]
    metas = [json.loads((run / 'run_meta.json').read_text()) for run in runs]
    for run, meta in zip(runs, metas):
        if meta.get('tolerance_sha256') != tolerance_sha:
            raise ValueError(f'{run} started under another tolerance than {tolerance_path}, or none: the tolerance '
                             f'must be written before the run')
    datasets = {run: sorted(meta['datasets']) for run, meta in zip(runs, metas)}
    checks = [{'runs': [left.name, right.name], **pilot_check(left, right, datasets[left])} for left, right in pairs]
    sides = [sum((collect_records(run, datasets[run]) for run in side), []) for side in zip(*pairs)]
    names = [side[0]['backend'] for side in sides]

    paired = pair(*sides)
    for p in paired:
        p['differs'] = p[names[0]]['retrieved'] != p[names[1]]['retrieved']
    index = {tuple(p[key] for key in GROUP) + (p['qid'],): p for p in paired}
    questions = []
    for p in paired:
        def lookup(**arm) -> Optional[dict]:
            return index.get(tuple(arm.get(key, p[key]) for key in GROUP) + (p['qid'],))

        hits = [p[name]['hit'] for name in names]
        questions.append({**{key: p[key] for key in GROUP}, 'qid': p['qid'], 'answers': p['answers'],
                          'differs': p['differs'], 'cause': _cause(p, lookup),
                          'hit_flip': names[hits.index(True)] if hits[0] != hits[1] else None,
                          'pool_size_differs': p['pool_size_differs'],
                          **{name: {column: p[name][column] for column in
                                    ('retrieved', 'hit', 'pool_size', 'judge_calls', 'total', 'embedding',
                                     'retrieval', 'judge')} for name in names}})

    tolerance = yaml.safe_load(Path(tolerance_path).read_text())
    groups = defaultdict(list)
    for question in questions:
        groups[tuple(question[key] for key in GROUP)].append(question)
    rows = []
    for key, group in sorted(groups.items()):
        count = len(group)
        hit_rate = {name: sum(q[name]['hit'] for q in group) / count for name in names}
        row = {**dict(zip(GROUP, key)), 'n_questions': count, 'hit_rate': hit_rate,
               'hit_rate_difference': hit_rate[names[0]] - hit_rate[names[1]],
               'differing': sum(q['differs'] for q in group),
               'hit_flips': {name: sum(q['hit_flip'] == name for q in group) for name in names},
               'causes': dict(Counter(q['cause'] for q in group if q['cause'])),
               'flip_causes': {name: dict(Counter(q['cause'] for q in group if q['hit_flip'] == name))
                               for name in names},
               'pool_size_differs': sum(q['pool_size_differs'] for q in group),
               'latency_p50_ms': {name: latency_stats([q[name]['total'] * 1000.0 for q in group])[1]
                                  for name in names}}
        row['within_tolerance'] = _verdict(row, tolerance)
        rows.append(row)

    condition = {run.name: meta.get('condition') for run, meta in zip(runs, metas)}
    return {'backends': names, 'tolerance': {'path': str(tolerance_path), 'sha256': tolerance_sha, **tolerance},
            'condition': condition, 'checks': checks, 'groups': rows}, questions


def _counted(causes: dict) -> str:
    return ', '.join(f'{cause} {count}' for cause, count in sorted(causes.items()))


def write_parity(summary: dict, questions: List[dict], out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / 'parity.json').write_text(json.dumps(summary, indent=1))
    with open(out / 'parity_questions.jsonl', 'w') as f:
        f.writelines(json.dumps(question) + '\n' for question in questions)

    left, right = summary['backends']
    verdicts = [g['within_tolerance'] for g in summary['groups']]
    tolerance = summary['tolerance']
    lines = ['# Parity: NL inside Blend against vanilla Pneuma', '',
             f"Tolerance `{tolerance['path']}` ({tolerance['status']}, sha256 {tolerance['sha256'][:12]}), recorded by "
             f"every run when it started. Groups under {tolerance['min_questions']} questions are not judged.", '',
             f"Timings as measured; {left} and {right} ran "
             + ('at the same time, each on its own GPUs and CPU cores.'
                if all(c and c['parallel'] for c in summary['condition'].values()) else 'one at a time.')
             + f" Condition per run: `{json.dumps(summary['condition'])}`.", '',
             f'**Within tolerance: {verdicts.count(True)} of {len(verdicts)} groups; '
             f'outside: {verdicts.count(False)}; not judged: {verdicts.count(None)}.**', '',
             'Causes of differing top-k lists: `judge` (the judge-off lists agree), `bm25` and `vector` (the '
             'attribution arm on the shared texts differs; BM25 is named first where both differ), `own texts` (both '
             'components agree on the shared texts: the own builds index other row samples or narration wording, or '
             'the alpha merge differs). Flips are the questions only one side hits, with their causes.', '',
             f'| dataset | family | texts | k | alpha | judge | questions | hit {left} | hit {right} | difference '
             f'| differing | flips {left}/{right} | causes | p50 ms {left}/{right} | within |',
             '| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |']
    for g in summary['groups']:
        flips = ' / '.join(str(g['hit_flips'][name]) + (f" ({_counted(g['flip_causes'][name])})"
                                                         if g['flip_causes'][name] else '') for name in (left, right))
        within = {True: 'yes', False: '**no**', None: 'n/a'}[g['within_tolerance']]
        lines.append(f"| {g['dataset']} | {g['family']} | {g['texts']} | {g['k']} | {g['alpha']} | {g['rerank']} "
                     f"| {g['n_questions']} | {g['hit_rate'][left]:.3f} | {g['hit_rate'][right]:.3f} "
                     f"| {g['hit_rate_difference']:+.3f} | {g['differing']} "
                     f"| {flips} | {_counted(g['causes']) or '-'} "
                     f"| {g['latency_p50_ms'][left]:.0f}/{g['latency_p50_ms'][right]:.0f} | {within} |")
    lines += ['', '## Run checks (shared texts, timings, own-build differences)', '']
    for check in summary['checks']:
        texts = check['own_text_differences']
        named = '; '.join(f'{label} in {texts[kind][0]} of {texts[kind][1]} blocks' for kind, label in
                          (('row', 'row samples'), ('schema', 'narration wording'), ('contexts', 'contexts'))
                          if kind in texts)
        lines += [f"- {' vs '.join(check['runs'])}: met `{json.dumps(check['met'])}`; the own builds differ in "
                  f"{named or 'nothing'}; per-table document count differences: "
                  f"{len(check['own_document_differences'])}"]
    print('\n'.join(lines))


def _throughput(run: Path, meta: dict) -> List[dict]:
    subsets = [name for name, dataset in meta['datasets'].items() if 'tables' in dataset]
    passes = defaultdict(lambda: defaultdict(list))
    for record in collect_records(run, subsets):
        passes[tuple(record[key] for key in ('dataset', 'k', 'n', 'alpha', 'rerank'))][record['rep']].append(record)
    rows = []
    for (dataset, k, n, alpha, rerank), by_pass in sorted(passes.items()):
        totals = {stage: [sum(r[stage] for r in records) for records in by_pass.values()]
                  for stage in ('total', 'embedding', 'retrieval', 'judge')}
        mean = statistics.mean(totals['total'])
        counts = {len(records) for records in by_pass.values()}
        if len(counts) != 1:
            raise ValueError(f'{run.name}/{dataset}: passes over {sorted(counts)} questions; a pass is unfinished')
        questions = counts.pop()
        rows.append({'run': run.name, 'dataset': dataset, 'tables': meta['datasets'][dataset]['tables'],
                     'backend': meta['backend'], 'k': k, 'n': n, 'alpha': alpha, 'rerank': rerank,
                     'passes': len(by_pass), 'questions': questions, 'mean_s': mean,
                     'stdev_s': statistics.stdev(totals['total']) if len(by_pass) > 1 else 0.0,
                     'min_s': min(totals['total']), 'max_s': max(totals['total']),
                     'queries_per_s': questions / mean if mean else None,
                     **{f'{stage}_s': statistics.mean(totals[stage]) for stage in ('embedding', 'retrieval', 'judge')},
                     'hit_rate': statistics.mean(r['hit'] for records in by_pass.values() for r in records)})
    return rows


def _offline(run: Path, meta: dict, dataset: str, build: str) -> dict:
    stages = meta['datasets'][dataset][build]['stages']
    documents = run / dataset / (SHARED_DOCUMENTS if build == 'shared_build' else 'documents.json')
    tables = len(json.loads(documents.read_text()))
    texts = 'own' if 'narrate' in stages else 'imported'
    narrate = stages['narrate' if texts == 'own' else 'import']
    index = stages['index']
    return {'run': run.name, 'dataset': dataset, 'backend': meta['backend'], 'texts': texts, 'tables': tables,
            'register_s': stages['register']['seconds'], 'narrate_s': narrate['seconds'],
            'narration_calls': narrate['judge_calls'],
            'narration_calls_per_s': narrate['judge_calls'] / narrate['seconds'] if narrate['seconds'] else None,
            'tables_per_s': tables / narrate['seconds'] if texts == 'own' and narrate['seconds'] else None,
            'embed_s': index['embedding'], 'embedding_calls': index['embedding_calls'],
            'index_s': index['seconds'] - index['embedding'],
            'total_s': sum(stage['seconds'] for stage in stages.values())}


def efficiency_report(runs: List[Path]) -> dict:
    summary = {'condition': {}, 'throughput': [], 'overlap': {}, 'offline': [], 'storage': []}
    subsets = defaultdict(list)
    for run in runs:
        meta = json.loads((run / 'run_meta.json').read_text())
        summary['condition'][run.name] = meta.get('condition')
        summary['throughput'] += _throughput(run, meta)
        for name, entry in meta['datasets'].items():
            if 'tables' in entry:
                subsets[name].append(entry)
        for dataset, entry in sorted(meta['datasets'].items()):
            for build in ('build', 'shared_build'):
                if build in entry:
                    offline = _offline(run, meta, dataset, build)
                    summary['offline'].append(offline)
                    summary['storage'].append({key: offline[key] for key in ('run', 'dataset', 'backend', 'texts')}
                                              | entry[build]['storage'])
    for name, sides in sorted(subsets.items()):
        if len({(side['tables_sha256'], side['shared_texts']['sha256']) for side in sides}) > 1:
            raise ValueError(f'{name}: the runs timed other tables or other texts')
        if len(sides) == 2:
            summary['overlap'][name] = overlap(*(side['questions_window'] for side in sides))
    return summary


def overlap(window: List[float], other: List[float]) -> float:
    (start, end), (other_start, other_end) = window, other
    shorter = min(end - start, other_end - other_start)
    return max(0.0, min(end, other_end) - max(start, other_start)) / shorter if shorter else 0.0


def write_efficiency(summary: dict, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / 'efficiency.json').write_text(json.dumps(summary, indent=1))
    conditions = summary['condition'].values()

    def number(value, digits=2) -> str:
        return '-' if value is None else f'{value:,.{digits}f}'

    lines = ['# Efficiency: NL inside Blend against vanilla Pneuma', '',
             'Timings as measured; the backends ran '
             + ('at the same time, each on its own GPUs and CPU cores.' if all(c and c['parallel'] for c in conditions)
                else 'one at a time in at least one run.')
             + f" Condition per run: `{json.dumps(summary['condition'])}`.", '',
             '## Query throughput', '',
             'A pass is the questions once, in the same order on both sides; its time is the sum of the questions\' '
             'totals. Judge on and off alternate per question inside a pass (Pneuma\'s script times one setting\'s '
             'loop back to back); each setting\'s pass time sums only its own questions. Mean and spread over the '
             'passes; stage seconds are those of a mean pass.', '',
             'Share of the shorter question loop that ran beside the other backend\'s, per subset: '
             + (', '.join(f'{name} {share:.0%}' for name, share in summary['overlap'].items()) or 'no pair') + '.', '',
             '| subset | tables | backend | judge | k | passes | questions | mean s | stdev s | min s | max s '
             '| queries/s | embedding s | retrieval s | judge s | hit rate |',
             '| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |']
    for row in sorted(summary['throughput'], key=lambda r: (r['tables'], r['rerank'], r['backend'])):
        lines.append(f"| {row['dataset']} | {row['tables']} | {row['backend']} | {row['rerank']} | {row['k']} "
                     f"| {row['passes']} | {row['questions']} | {number(row['mean_s'])} | {number(row['stdev_s'])} "
                     f"| {number(row['min_s'])} | {number(row['max_s'])} | {number(row['queries_per_s'])} "
                     f"| {number(row['embedding_s'])} | {number(row['retrieval_s'])} | {number(row['judge_s'])} "
                     f"| {row['hit_rate']:.3f} |")
    lines += ['', '## Offline stages', '',
              'Every build through the shared clients at the same narration concurrency. `imported` builds index '
              'another build\'s texts, so their narrate column is the import. Build indexes is the index stage '
              'without its embedding calls.', '',
              '| run | dataset | backend | texts | tables | register s | narrate s | narration calls | calls/s '
              '| tables/s | embed s | embedding calls | build indexes s | total s |',
              '| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |']
    for row in summary['offline']:
        lines.append(f"| {row['run']} | {row['dataset']} | {row['backend']} | {row['texts']} | {row['tables']} "
                     f"| {number(row['register_s'])} | {number(row['narrate_s'])} | {row['narration_calls']} "
                     f"| {number(row['narration_calls_per_s'])} | {number(row['tables_per_s'])} "
                     f"| {number(row['embed_s'])} | {row['embedding_calls']} | {number(row['index_s'])} "
                     f"| {number(row['total_s'])} |")
    categories = list(summary['storage'][0]['bytes']) if summary['storage'] else []
    lines += ['', '## Storage (MiB)', '',
              'The same categories on both sides. Blend\'s inverted index is apart and never counted against NL; '
              'its time is in the build record (`inverted_index_s`), outside every stage. Blend\'s vectors include the '
              '`tableid` ART of its documents, which the restricted exact scan uses and Pneuma has no counterpart for; '
              'its tokens include the token ART BM25 uses. These Blend rows are D3 on Pneuma\'s lakes, beside '
              'standalone Pneuma.', '',
              '| run | dataset | backend | texts | ' + ' | '.join(categories) + ' | NL total | inverted index |',
              '| --- ' * (len(categories) + 6) + '|']
    for row in summary['storage']:
        mib = [number(size / 2 ** 20) for size in row['bytes'].values()]
        lines.append(f"| {row['run']} | {row['dataset']} | {row['backend']} | {row['texts']} | " + ' | '.join(mib)
                     + f" | {number((row['total'] - row['inverted index']) / 2 ** 20)} "
                     f"| {number(row['inverted index'] / 2 ** 20)} |")
    print('\n'.join(lines))


def write_summary(rows: List[dict], run_path: Path) -> None:
    with open(run_path / 'summary.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    if sys.argv[1] == 'parity':
        out, runs = Path(sys.argv[2]), [Path(argument) for argument in sys.argv[3:]]
        if not runs or len(runs) % 2:
            sys.exit(USAGE)
        write_parity(*parity_report(list(zip(runs[::2], runs[1::2]))), out)
        return
    if sys.argv[1] == 'efficiency':
        out, runs = Path(sys.argv[2]), [Path(argument) for argument in sys.argv[3:]]
        if not runs:
            sys.exit(USAGE)
        write_efficiency(efficiency_report(runs), out)
        return
    left, right = (Path(argument) for argument in sys.argv[1:3])
    datasets = sorted(json.loads((left / 'run_meta.json').read_text())['datasets'])
    print(json.dumps(pilot_check(left, right, datasets), indent=1))


if __name__ == '__main__':
    main()
