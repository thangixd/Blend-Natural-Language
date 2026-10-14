import configparser
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import duckdb

from evaluation.nlseeker.datasets import pneuma_id
from src.NLSeeker.Retriever import bm25_sql
from src.NLSeeker.Tokenizer import tokenize

from typing import Callable, Dict, Iterator, List, Optional

USAGE = '''python -m evaluation.nlseeker.diagnosis <out dir> <blend run> <pneuma run> [<blend run> <pneuma run> ...]'''

TIE = 1e-4
FIELDS = ('family', 'k', 'qid', 'texts', 'alpha', 'rerank', 'documents', 'retrieved', 'answers')


def pneuma_key(document: str) -> str:
    table = re.split('_SEP_(?:contents_SEP_|contexts-)', document)[0]
    return f'{Path(table).name}|{document[len(table) + len("_SEP_"):]}'


def blend_key(document: str, files: Dict[int, str]) -> str:
    table, rest = document.split('_SEP_', 1)
    return f'{Path(files[int(table)].replace(chr(92), "/")).name.replace(chr(34), chr(39))}|{rest}'


def table(key: str) -> str:
    return pneuma_id(key.split('|', 1)[0])


def hits(keys: List[str], k: int, answers: set) -> bool:
    return bool(set(list(dict.fromkeys(table(key) for key in keys))[:k]) & answers)


def same_scores(ours: List[float], theirs: List[float]) -> bool:
    return len(ours) == len(theirs) and all(abs(a - b) <= TIE * max(1.0, abs(a), abs(b))
                                            for a, b in zip(ours, theirs))


def vector_question(exact: List[str], pneuma: List[str], candidates: List[str], k: int, answers: set) -> dict:
    candidates, found = set(candidates), set(pneuma)
    beyond = [key for key in exact if key not in candidates]
    without = [key for key in pneuma if key not in candidates or key in exact]
    nearest = exact[0]
    return {
        'pool_recall': len(found & set(exact)) / len(exact),
        'recall_outside_candidates': len(found & set(beyond)) / len(beyond) if beyond else None,
        'seeker_hit': hits(exact, k, answers), 'pneuma_hit': hits(pneuma, k, answers),
        'pneuma_hit_without_candidates': hits(without, k, answers), 'empty_without_candidates': not without,
        'nearest': 'answer' if table(nearest) in answers else 'other',
        'missed': nearest not in found,
        'nearest_is_candidate': nearest in candidates,
        'first_is_answer': table(pneuma[0]) in answers,
    }


def _share(rows: List[dict], field: str) -> Optional[float]:
    values = [row[field] for row in rows if row[field] is not None]
    return sum(values) / len(values) if values else None


def _vector_summary(rows: List[dict]) -> dict:
    summary = {'questions': len(rows)}
    for field in ('pool_recall', 'recall_outside_candidates', 'seeker_hit', 'pneuma_hit',
                  'pneuma_hit_without_candidates'):
        summary[field] = _share(rows, field)
    summary['empty_without_candidates'] = sum(row['empty_without_candidates'] for row in rows)
    for kind in ('answer', 'other'):
        group = [row for row in rows if row['nearest'] == kind]
        missed = [row for row in group if row['missed']]
        summary[f'nearest_{kind}'] = len(group)
        summary[f'missed_{kind}'] = len(missed)
        summary[f'first_is_answer_after_missed_{kind}'] = _share(missed, 'first_is_answer')
        outside = [row for row in group if not row['nearest_is_candidate']]
        summary[f'nearest_{kind}_outside_candidates'] = len(outside)
        summary[f'missed_{kind}_outside_candidates'] = sum(row['missed'] for row in outside)
    return summary


def diagnose(blend: Dict[tuple, dict], pneuma: Dict[tuple, dict], seeker_id: Dict[str, str],
             score: Callable[[str, List[str]], Dict[str, float]], questions: Dict[tuple, str]) -> dict:
    ties, vector, judge = (defaultdict(lambda: defaultdict(int)), defaultdict(list),
                           defaultdict(lambda: defaultdict(int)))
    for name, ours in sorted(blend.items()):
        texts, alpha, rerank, family, k, qid = name
        theirs, group = pneuma[name], f'{family} k={k}'
        if texts == 'shared' and alpha == 1.0 and ours['retrieved'] != theirs['retrieved']:
            ids = [[seeker_id[key] for key in record['keys']] for record in (ours, theirs)]
            scores = score(questions[(qid, family)], sorted(set(ids[0]) | set(ids[1])))
            equal = same_scores(*([scores.get(d, 0.0) for d in side] for side in ids))
            ties[group]['tie' if equal else 'score mismatch'] += 1
        elif texts == 'shared' and alpha == 0.0:
            candidates = pneuma[('shared', 1.0, rerank, family, k, qid)]['keys']
            vector[group].append(vector_question(ours['keys'], theirs['keys'], candidates, k, set(ours['answers'])))
        elif texts == 'own' and rerank == 'on' and ours['retrieved'] != theirs['retrieved']:
            off = ('own', alpha, 'off', family, k, qid)
            if blend[off]['retrieved'] == pneuma[off]['retrieved']:
                judge[group]['different lists'] += 1
                judge[group]['different documents'] += set(ours['keys']) != set(theirs['keys'])
    return {'ties': {g: dict(v) for g, v in sorted(ties.items())},
            'vector': {g: _vector_summary(rows) for g, rows in sorted(vector.items())},
            'judge': {g: dict(v) for g, v in sorted(judge.items())}}


def read(run: Path, dataset: str, key: Callable[[str], str]) -> Dict[tuple, dict]:
    records = {}
    for record in _lines(run / dataset / 'per_query.jsonl'):
        record = {field: record.get(field, 'own') if field == 'texts' else record[field] for field in FIELDS}
        record['keys'] = [key(d) for d in record['documents']]
        records[tuple(record[f] for f in ('texts', 'alpha', 'rerank', 'family', 'k', 'qid'))] = record
    return records


def _lines(path: Path) -> Iterator[dict]:
    with open(path) as f:
        for line in f:
            yield json.loads(line)


def write(result: dict, out: Path, label: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / f'diagnosis_{label}.json').write_text(json.dumps(result, indent=1))
    lines = [f'# Diagnosis of the parity runs: {label}', '']
    for section, title in (('ties', 'Shared texts, alpha=1: differing lists, scored position by position'),
                           ('vector', 'Shared texts, alpha=0: Pneuma against the exact nearest documents'),
                           ('judge', 'Own texts, judge on: lists that differ only with the judge')):
        rows = result[section]
        columns = list(dict.fromkeys(c for row in rows.values() for c in row))
        lines += [f'## {title}', '', '| group | ' + ' | '.join(columns) + ' |', '|' + ' --- |' * (len(columns) + 1)]
        lines += [f'| {group} | ' + ' | '.join(_cell(row.get(c)) for c in columns) + ' |' for group, row in rows.items()]
        lines.append('')
    print('\n'.join(lines))


def _cell(value) -> str:
    return '-' if value is None else f'{value:.3f}' if isinstance(value, float) else str(value)


def main() -> None:
    out, runs = Path(sys.argv[1]), [Path(argument) for argument in sys.argv[2:]]
    if not runs or len(runs) % 2:
        sys.exit(USAGE)
    for blend_run, pneuma_run in zip(runs[::2], runs[1::2]):
        for dataset in sorted(json.loads((blend_run / 'run_meta.json').read_text())['datasets']):
            ini = configparser.ConfigParser()
            ini.read(blend_run / dataset / 'artifacts' / 'nlseeker.ini')
            con = duckdb.connect(ini['Database']['path'], read_only=True)
            index, nl_index = ini['Database']['index_table'], ini['NLSeeker']['index_name']
            files = dict(con.execute(f'SELECT tableid, filename FROM "{index}_tables"').fetchall())
            documents, tokens = f'{nl_index}_nl_documents', f'{nl_index}_nl_tokens'
            seeker_id = {blend_key(row[0], files): row[0]
                         for row in con.execute(f'SELECT docid FROM "{documents}"').fetchall()}
            questions, duckdb_hnsw, short = {}, defaultdict(list), defaultdict(int)
            for run in _lines(blend_run / dataset / 'nl_runs.jsonl'):
                if 'qid' not in run:
                    continue
                questions[(run['qid'], run['family'])] = run['question']
                group = f'{run["family"]} k={run["k"]}'
                if run['alpha'] == 0.0 and not run['judge_on'] and run.get('hnsw_recall') is not None:
                    duckdb_hnsw[group].append(run['hnsw_recall'])
                if run['alpha'] == 1.0 and not run['judge_on'] and run['bm25_matched'] < run['k'] * run['n']:
                    short[group] += 1

            def score(question: str, docids: List[str]) -> Dict[str, float]:
                ids = ', '.join(f"'{d}'" for d in docids)
                return dict(con.execute(bm25_sql(documents, tokens, tokenize(question),
                                                 f' AND t.docid IN ({ids}) ')).fetchall())

            result = diagnose(read(blend_run, dataset, lambda d: blend_key(d, files)),
                              read(pneuma_run, dataset, pneuma_key), seeker_id, score, questions)
            for group, row in result['vector'].items():
                recalls = duckdb_hnsw.get(group)
                row['duckdb_hnsw_recall'] = sum(recalls) / len(recalls) if recalls else None
                row['short_keyword_lists'] = short.get(group, 0)
            write(result, out, dataset)


if __name__ == '__main__':
    main()
