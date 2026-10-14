import json
import random
from dataclasses import dataclass
from pathlib import Path

from typing import List, Optional, Tuple


FAMILIES = ('BC1', 'BC2', 'BX1', 'BX2')

FAMILY_FIELDS = {
    'BC1': ('bc', 'question_from_sql_1'),
    'BC2': ('bc', 'question'),
    'BX1': ('bx', 'question_bx1'),
    'BX2': ('bx', 'question_bx2'),
}


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    tar: str
    questions: str
    zip: str
    zip_name: str
    lake: str


DATASETS = {spec.name: spec for spec in (
    DatasetSpec('adventure_works', 'pneuma_adventure_works.tar',
                'pneuma_adventure_works_questions_annotated.jsonl',
                'adventure-20260604T041942Z-3-001.zip', 'adventure', 'adventure_works'),
    DatasetSpec('chembl_10K', 'pneuma_chembl_10K.tar',
                'pneuma_chembl_10K_questions_annotated.jsonl',
                'chembl-20260604T041943Z-3-001.zip', 'chembl', 'chembl'),
    DatasetSpec('chicago_10K', 'pneuma_chicago_10K.tar',
                'pneuma_chicago_10K_questions_annotated.jsonl',
                'chicago-20260604T041945Z-3-001.zip', 'chicago', 'chicago_open'),
    DatasetSpec('fetaqa', 'pneuma_fetaqa.tar',
                'pneuma_fetaqa_questions_annotated.jsonl',
                'fetaqa-20260604T041946Z-3-001.zip', 'fetaqa', 'fetaqa'),
    DatasetSpec('public_bi', 'pneuma_public_bi.tar',
                'pneuma_public_bi_questions_annotated.jsonl',
                'public-20260604T041947Z-3-001.zip', 'public', 'public_bi'),
)}


@dataclass(frozen=True)
class Question:
    qid: str
    text: str
    answer_tables: Tuple[str, ...]


def lake_source(lake: str) -> Tuple[str, Optional[int]]:
    for spec in DATASETS.values():
        if lake == spec.lake:
            return spec.name, None
        prefix, _, size = lake.rpartition('_')
        if prefix == spec.lake and size.isdigit():
            return spec.name, int(size)
    raise ValueError(f'unknown lake {lake!r}; expected one of {sorted(spec.lake for spec in DATASETS.values())} '
                     f'or <lake>_<tables>')


def pneuma_id(filename: str) -> str:
    if '_SEP_' not in filename:
        raise ValueError(f'not a pneuma lake filename: {filename!r}')
    return filename.rsplit('_SEP_', 1)[1].removesuffix('.csv')


def subset_order(files: List[str], questions: List[Question]) -> List[str]:
    by_id = {pneuma_id(file): file for file in files}
    answers = [by_id[table] for question in questions for table in question.answer_tables if table in by_id]
    first = list(dict.fromkeys(answers))
    rest = sorted(set(files) - set(first))
    random.Random(0).shuffle(rest)
    return first + rest


def load_questions(jsonl_path: Path, family: str, limit: Optional[int]) -> list:
    if family not in FAMILY_FIELDS:
        raise ValueError(f'unknown family {family!r}; expected one of {FAMILIES}')
    source, field = FAMILY_FIELDS[family]
    id_field = 'id' if source == 'bc' else 'context_id'

    questions = []
    with open(jsonl_path) as f:
        for line in f:
            row = json.loads(line)
            answers = tuple(dict.fromkeys(row['answer_tables']))
            questions.append(Question(qid=row[id_field], text=row[field], answer_tables=answers))
            if limit is not None and len(questions) >= limit:
                break
    return questions
