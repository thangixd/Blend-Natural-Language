import json
import random
import statistics
import sys

from benchmarks.nl_integration.common import (NLMeter, Records, answer_holds_values, markdown_table, nl_table_ids,
                                              questions, run_lakes, value_tokens, write_summary)
from benchmarks.nl_integration.order import arm_names, arms_table, order_sections, plot, run_arms
from benchmarks.nl_integration.runtime import INNER_K
from src.DBHandler import DBHandler
from src.Operators.Seekers import Keyword, NaturalLanguage

ARMS = arm_names('Keyword')


def run_lake(args, db: DBHandler) -> None:
    records = Records(args, 'regime')
    meter = NLMeter(db, args.out / f'nl_runs_regime_{args.lake}.jsonl')
    rng = random.Random(args.seed)
    nl_tables = set(nl_table_ids(db))

    NaturalLanguage('warm-up', k=1, n=1, rerank=bool(args.rerank)).run()
    meter.write(arm='warm-up')
    measured = 0
    pool = questions(db, args.lake)
    rng.shuffle(pool)
    for question in pool:
        if measured == args.questions:
            break
        values = value_tokens(question['ref_values'])
        if not values:
            records.write({'qid': question['qid'], 'answer': question['answer'], 'skipped': 'no values'})
            continue
        measured += 1
        records.write({'qid': question['qid'], 'answer': question['answer'], 'answers': question['answers'],
                       'values': values,
                       'held': answer_holds_values(db, question),
                       **run_arms(args, meter, question, lambda k: Keyword(values, k), 'Keyword', nl_tables)})

    rows = records.read()
    write_summary(args, 'regime', summarize(rows, args))
    measured_rows = [row for row in rows if 'skipped' not in row]
    if measured_rows:
        plot(measured_rows, args.out / f'regime_{args.lake}.png', ARMS,
             f'U2 on {args.lake}, {len(measured_rows)} questions')


def summarize(records: list, args) -> str:
    rows = [row for row in records if 'skipped' not in row]
    text = (f'# U2: Keyword with NL on {args.lake}\n\nk {args.k}, first leg k {args.inner_k} (Keyword) or '
            f'{args.nl_inner_k} (NL), judge {"on" if args.rerank else "off"}, seed {args.seed}. Keyword queries the '
            'values the question names, taken from Pneuma\'s SQL (a clean extraction) and tokenized as the index '
            'stores them. No table is excluded. The true intersection is Keyword\'s untruncated candidates that '
            'have NL documents. An arm returns the answer when it returns any table Pneuma labels for the question, '
            'as parity scores a hit.\n\n'
            f'{len(rows)} questions measured; {len(records) - len(rows)} skipped without a named value.\n')
    if not rows:
        return text
    text += ('\n## NL runs\n\nPooled documents equal the judge calls a run makes: NL first pools from the whole '
             'lake, NL last only from Keyword\'s first leg.\n\n')
    measured = {row['qid'] for row in rows}
    by_arm = {}
    with open(args.out / f'nl_runs_regime_{args.lake}.jsonl') as f:
        for run in map(json.loads, f):
            if run.get('qid') in measured:
                by_arm.setdefault(run['arm'], []).append(run)
    text += markdown_table(['NL run', 'Runs', 'Median pooled documents', 'Median judge calls', 'Median NL ms'],
                           [[arm, len(runs), statistics.median(r['pooled_documents'] for r in runs),
                             statistics.median(r['judge_calls'] for r in runs),
                             statistics.median(r['total'] for r in runs) * 1000] for arm, runs in by_arm.items()])
    text += ('\nThe questions are split by whether the answer table holds a named value: only there can the value '
             'anchor help; where it holds none, no Keyword arm and no composed arm can return it. Value share is '
             'the share of an arm\'s tables that hold a named value, 1 for every composed arm by construction. '
             'Answer retention is the share of the questions where NL alone returns a labelled answer table in '
             'which the arm returns one too. NL alone at 2k returns up to twice k tables.\n')
    for held, title in ((True, 'Answer table holds a value'), (False, 'Answer table holds none')):
        group = [row for row in rows if row['held'] is held]
        text += f'\n# {title} ({len(group)} questions)\n\n'
        text += (arms_table(group, 'Keyword', 'Value share') + order_sections(group, args, 'Keyword')
                 if group else 'No questions.\n')
    return text


def main(argv=None) -> int:
    return run_lakes('regime', run_lake, argv, questions=100, k=10, inner_k=INNER_K, nl_inner_k=50, rerank=1)


if __name__ == '__main__':
    sys.exit(main())
