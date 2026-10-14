import math
import random
import statistics
import sys

from benchmarks.nl_integration.common import (NLMeter, Records, markdown_table, nl_config, nl_table_ids, questions,
                                              run_lakes, value_tokens, write_summary)
from benchmarks.nl_integration.metrics import answer_found, filtered_prefix, normalization_costs_answers, retention
from benchmarks.nl_integration.order import join_partner
from benchmarks.nl_integration.runtime import INNER_K
from src.DBHandler import DBHandler
from src.Operators.Seekers import Keyword, NaturalLanguage, SingleColumnOverlap

from typing import Tuple

ARMS = ('Restricted', 'Post-filter', 'Deep post-filter', 'Filtered full ranking')
RESTRICTIONS = ('SC', 'Keyword')


def restrictions(args, db: DBHandler, question: dict, rng: random.Random) -> Tuple[dict, dict]:
    found, skipped = {}, {}
    partner = join_partner(args, db, question, rng)
    if partner:
        fields, values = partner
        exclude = f' AND TableId <> {fields["partner"]} '
        found['SC'] = (SingleColumnOverlap(values, args.inner_k).run(exclude), exclude, fields)
    else:
        skipped['SC'] = 'no partner'
    values = value_tokens(question['ref_values'])
    if values:
        found['Keyword'] = (Keyword(values, args.inner_k).run(), '', {'values': values})
    else:
        skipped['Keyword'] = 'no values'
    for name in [name for name, (first_leg, *_) in found.items() if not first_leg]:
        del found[name]
        skipped[name] = 'empty first leg'
    return found, skipped


def run_lake(args, db: DBHandler) -> None:
    records = Records(args, 'pushdown')
    rng = random.Random(args.seed)
    n = nl_config(db).n
    tables = len(nl_table_ids(db))
    documents = db.execute_and_fetchall(f'SELECT count(*) FROM "{nl_config(db).documents_table}"')[0][0]
    whole_lake = math.ceil(documents / tables) + 1

    meter = NLMeter(db, args.out / f'nl_runs_pushdown_{args.lake}.jsonl')
    NaturalLanguage('warm-up', k=1, n=1, rerank=False).run()
    meter.write(arm='warm-up')
    measured = 0
    pool = questions(db, args.lake)
    rng.shuffle(pool)
    for question in pool:
        if measured == args.questions:
            break
        qid, text, answer = question['qid'], question['question'], question['answer']
        found, skipped = restrictions(args, db, question, rng)
        if not found:
            records.write({'qid': qid, 'answer': answer, 'skipped': 'no restriction'})
            continue
        for name, reason in skipped.items():
            records.write({'qid': qid, 'answer': answer, 'restriction': name, 'skipped': reason})
        measured += 1
        full = NaturalLanguage(text, k=tables, n=whole_lake, rerank=False).run()
        (full_run,) = meter.write(arm='full ranking', qid=qid)
        top_k = NaturalLanguage(text, k=args.k, rerank=False).run()
        (post_run,) = meter.write(arm='unrestricted', qid=qid)
        deep = NaturalLanguage(text, k=args.k * n, n=1, rerank=False).run()
        (deep_run,) = meter.write(arm='deep', qid=qid)

        for name, (first_leg, exclude, fields) in found.items():
            additionals = f'{exclude} AND TableId IN ({db.create_sql_list_numeric(first_leg)}) '
            restricted = NaturalLanguage(text, k=args.k, rerank=False).run(additionals)
            (run,) = meter.write(arm='restricted', qid=qid, restriction=name)
            repeat = NaturalLanguage(text, k=args.k, rerank=False).run(additionals)
            meter.write(arm='restricted, repeat', qid=qid, restriction=name)
            allowed = set(first_leg)
            records.write({
                'qid': qid, 'answer': answer, 'answers': question['answers'], 'restriction': name, **fields,
                'allowed': sorted(allowed),
                'restriction_size': run['restriction_size'],
                'arms': {'Restricted': restricted, 'Post-filter': filtered_prefix(top_k, allowed, args.k),
                         'Deep post-filter': filtered_prefix(deep, allowed, args.k),
                         'Filtered full ranking': filtered_prefix(full, allowed, args.k)},
                'restricted_repeat': repeat,
                'pooled': {'Restricted': run['pooled_documents'], 'Post-filter': post_run['pooled_documents'],
                           'Deep post-filter': deep_run['pooled_documents'],
                           'Filtered full ranking': full_run['pooled_documents']}})

    write_summary(args, 'pushdown', summarize(records.read(), args, tables, documents))


def arm_row(rows: list, arm: str) -> list:
    def listed(row):
        return row['arms'][arm]

    found = sum(answer_found(row, listed(row)) for row in rows) / len(rows)
    pooled = statistics.median(row['pooled'][arm] for row in rows)
    if arm == 'Restricted':
        return [found, None, None, None, pooled]
    return [found, statistics.mean(retention(listed(row), row['arms']['Restricted']) for row in rows),
            sum(len(listed(row)) < len(row['arms']['Restricted']) for row in rows) / len(rows),
            sum(listed(row) == row['arms']['Restricted'] for row in rows) / len(rows), pooled]


def normalization_row(name: str, rows: list) -> list:
    def found(arm):
        return [answer_found(row, row['arms'][arm]) for row in rows]

    def shared_order(row):
        restricted, filtered = row['arms']['Restricted'], row['arms']['Filtered full ranking']
        return ([table for table in restricted if table in filtered],
                [table for table in filtered if table in restricted])

    restricted, filtered = sum(found('Restricted')), sum(found('Filtered full ranking'))
    floor = sum(answer_found(row, row['restricted_repeat']) != hit
                for row, hit in zip(rows, found('Restricted'))) / len(rows)
    return [name, len(rows),
            sum(set(row['arms']['Filtered full ranking']) != set(row['arms']['Restricted']) for row in rows),
            sum(a != b for a, b in map(shared_order, rows)), restricted, filtered,
            (filtered - restricted) / len(rows), floor,
            'fires' if normalization_costs_answers(restricted, filtered, len(rows), floor) else 'does not fire']


def answerable(rows: list) -> list:
    return [row for row in rows if answer_found(row, row['allowed'])]


def summarize(records: list, args, tables: int, documents: int) -> str:
    skipped = [row for row in records if row.get('skipped') == 'no restriction']
    measured = {row['qid'] for row in records} - {row['qid'] for row in skipped}
    groups = {name: [row for row in records if row.get('restriction') == name and 'arms' in row]
              for name in RESTRICTIONS}

    def skips(name, reasons):
        return sum(row.get('restriction') == name and row.get('skipped') in reasons for row in records)

    text = (f'# D2: restriction inside retrieval vs post-filter on {args.lake}\n\n'
            f'{tables} tables with NL documents ({documents} documents), k {args.k}, judge off, seed {args.seed}. '
            'Each restriction is the first leg Blend\'s Intersection pushes into NL as its last input: SC on the '
            f'answer table\'s join partner (U1; the partner\'s table excluded) or Keyword on the values the question '
            f'names (U2), each at first leg k {args.inner_k}. The restricted run is what Blend executes. Post-filter '
            'filters NL\'s unrestricted top k; deep post-filter its whole unrestricted pool of k·n documents; the '
            'filtered full ranking filters NL\'s ranking of every document, so it differs from the restricted run '
            'only through the pool and its min-max normalization. An arm finds the answer when it returns any table '
            'Pneuma labels for the question, as parity scores a hit.\n\n'
            f'{len(measured)} questions measured; {len(skipped)} skipped without a restriction.\n\n')
    text += markdown_table(
        ['Restriction', 'Questions', 'No partner or values', 'Empty first leg', 'Median allowed tables',
         'Median allowed with NL documents', 'With an answer table allowed', 'Their median allowed tables'],
        [[name, len(rows), skips(name, ('no partner', 'no values')), skips(name, ('empty first leg',)),
          statistics.median(len(row['allowed']) for row in rows) if rows else None,
          statistics.median(row['restriction_size'] for row in rows) if rows else None,
          len(answerable(rows)), statistics.median(len(row['allowed']) for row in answerable(rows))
          if answerable(rows) else None]
         for name, rows in groups.items()])
    groups = {name: answerable(rows) for name, rows in groups.items()}
    groups = {name: rows for name, rows in groups.items() if rows}
    if not groups:
        return text + '\nNo restricted runs with an answer table in the restriction.\n'
    text += ('\nThe sections below count only the questions with a labelled answer table among the allowed tables: '
             + ', '.join(f'{name} {len(rows)}' for name, rows in groups.items()) + '.\n')

    text += ('\n## Arms\n\nRetention is the share of the restricted answer an arm returns; short is the share of '
             'questions where it returns fewer tables; identical is the share where it returns the same list in the '
             'same order. Pooled documents are the judge calls the run would make with the judge on.\n\n')
    text += markdown_table(
        ['Restriction', 'Arm', 'Questions', 'Answer found', 'Retention', 'Short', 'Identical',
         'Median pooled documents'],
        [[name, arm, len(rows)] + arm_row(rows, arm) for name, rows in groups.items() for arm in ARMS])

    verdicts = [normalization_row(name, rows) for name, rows in groups.items()]
    fires = [row[0] for row in verdicts if row[-1] == 'fires']
    text += ('\n## Normalization\n\nFixed before the data: fix normalization only if, on either lake, the filtered '
             'full ranking finds a labelled answer table in more questions than the restricted run, by more than '
             'the noise floor and by at least 3 percentage points. Checked per restriction, so a question counts '
             'once in each. The judge is off, so the arms differ only through retrieval, and the noise floor is the '
             'restricted run repeated: the share of questions where the repeat flips whether a labelled answer table '
             'is found. Other sets counts questions where the filtered full ranking returns other tables; re-ordered '
             'counts those where the tables both lists return come in another order.\n\n')
    text += markdown_table(
        ['Restriction', 'Questions', 'Other sets', 'Re-ordered', 'Restricted finds', 'Filtered full ranking finds',
         'Difference', 'Noise floor', 'Rule'], verdicts)
    text += (f'\nNormalization rule: fires on {args.lake} ({", ".join(fires)}): normalizing against global ranges '
             'would find more labelled answers; fix it as follow-up work.\n' if fires else
             f'\nNormalization rule: does not fire on {args.lake}: the re-ordering costs no labelled answers by the '
             'rule; document it.\n')
    return text


def main(argv=None) -> int:
    return run_lakes('pushdown', run_lake, argv, questions=100, k=10, inner_k=INNER_K, containment=0.9,
                     min_values=10)


if __name__ == '__main__':
    sys.exit(main())
