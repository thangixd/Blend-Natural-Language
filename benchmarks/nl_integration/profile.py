import sys

from benchmarks.nl_integration.common import (Records, answer_holds_values, markdown_table, questions, run_lakes,
                                              shape, write_summary)
from src.DBHandler import DBHandler


def run_lake(args, db: DBHandler) -> None:
    holds = [answer_holds_values(db, question) for question in questions(db, args.lake)]
    record = {**shape(db), 'questions': len(holds),
              'questions_naming_values': sum(held is not None for held in holds),
              'answer_holds_values': sum(held is True for held in holds)}
    Records(args, 'profile').write(record)
    write_summary(args, 'profile', summarize(record, args))


def summarize(record: dict, args) -> str:
    named = record['questions_naming_values']
    rows = [[key.replace('_', ' '), record[key]] for key in
            ('tables', 'columns', 'cells', 'cells_per_column', 'median_cells_per_column', 'documents',
             'documents_per_column', 'questions', 'questions_naming_values', 'answer_holds_values')]
    text = (f'# Profile of {args.lake}\n\nCells are inverted index rows. Of the {named} questions naming values '
            f'(normalized), the answer table holds at least one in '
            f'{record["answer_holds_values"] / named if named else 0:.0%}; that is U2\'s split.\n\n')
    return text + markdown_table(['Measure', 'Value'], rows)


def main(argv=None) -> int:
    return run_lakes('profile', run_lake, argv)


if __name__ == '__main__':
    sys.exit(main())
