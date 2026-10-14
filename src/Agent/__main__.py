import argparse
import os
import traceback
from pathlib import Path

os.environ.setdefault('PYDANTIC_AI_NO_BANNER', '1')
os.environ.setdefault('TQDM_DISABLE', '1')
os.environ.setdefault('HF_HUB_VERBOSITY', 'error')

import pandas as pd

from src.Agent.Config import AgentConfig
from src.Agent.Dialogue import (Click, DialogueManager, LoadTable, Message, NoPreference, NotIt, Other, PlanCard,
                                Question, SetK, SetWording)
from src.Agent.Actions import ACTIONS, candidates, perform, plan_columns
from src.Agent.Plans import WORDING
from src.Agent.Text import Opaque

from typing import Optional, Tuple, Union
from src.Agent.Dialogue import Reply, Response

PREVIEW_ROWS = 5
PREVIEW_WIDTH = 40
FAILED = 'That turn failed with an internal error, which the record keeps. Please try again, or rephrase your message.'


def show(response: Response) -> str:
    lines = [_shown(response.shown)] + ([_content(response.chat)] if response.chat else [])
    older = [question for question in response.decisions if question is not response.shown]
    if older and not isinstance(response.shown, Question):
        lines.append(f'Open: {_shown(older.pop(0))}')
    lines += [f'Also open: {question.id} {_content(question.text)} '
              + ' | '.join(f'{option.id} {_content(option.label)}' for option in question.options)
              for question in older]
    return '\n'.join(lines)


def _shown(shown: Union[PlanCard, Question, Reply]) -> str:
    if isinstance(shown, PlanCard):
        lines = [f'Plan v{shown.version}: {shown.words}']
        lines += [f'Recognised as: {shown.task}{" (assumed)" if "structure" in shown.assumed else ""}'] \
            if shown.task else []
        lines += [f'Assumed: {", ".join(shown.assumed)}'] if shown.assumed else []
        lines += [f'Not used: {", ".join(repr(_content(words)) for words in shown.not_used)}'] \
            if shown.not_used else []
        lines += [f'Changed: {_content(change)}' for change in shown.changed]
        lines += [f'Notice: {_content(notice)}' for notice in shown.notices]
        lines += [f'Discrepancy: {_content(line)}' for line in shown.discrepancies]
        lines += [f'Preselected: {name} = {value} (change with :{name} <value>)'
                  for name, value in shown.preselected.items()]
        lines += ['Blend plan:']
        lines += [f'  {node.name}: {node.operator}(k={node.k}) {node.description}'
                  + (f' <- {", ".join(node.inputs)}' if node.inputs else '') for node in shown.nodes]
        lines += [] if shown.not_it else ['Not it is off: this request used its revisions, so start a new request.']
        lines += ['Tables:'] + [f'  {rank}. table id {table_id}: {name}'
                                for rank, (table_id, name) in enumerate(shown.tables, 1)]
        return '\n'.join(lines if shown.tables else lines + ['  none'])
    if isinstance(shown, Question):
        return '\n'.join([_content(shown.text)] + [f'  {option.id}  {_content(option.label)}'
                                                    for option in shown.options])
    return _content(shown.text)


def _content(text: Union[str, Opaque]) -> str:
    return text.for_prompt() if isinstance(text, Opaque) else text


def pick(argument: str, version: Optional[int]) -> Union[Click, NoPreference]:
    decision = argument.split('.')[0]
    if argument.strip() == f'{decision}.none':
        return NoPreference(decision, version=version)
    options = tuple(f'{decision}.{option.strip().split(".")[-1]}' for option in argument.split(','))
    return Click(decision, options if len(options) > 1 else options[0], version=version)


def main() -> None:
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--table', help='The query table, a CSV file.')
    source.add_argument('--find', help='A description of the query table: the natural-language seeker retrieves '
                                       'candidates from the lake and the one picked is loaded.')
    parser.add_argument('--datalake',
                        help='The directory of the indexed lake files, needed with --find and the result actions.')
    parser.add_argument('--no-header', action='store_true',
                        help='The first line is a row; the columns are col_0, col_1, ...')
    parser.add_argument('--config', default='config/config.ini')
    args = parser.parse_args()
    if args.find and not args.datalake:
        parser.error('--find needs --datalake')

    config = AgentConfig.load(args.config)
    table = args.table or find(args.find, args.datalake, args.config, config.k_default)
    if table is None:
        return
    manager = DialogueManager(config, args.config)
    print(show(manager.handle(LoadTable(table, header=not args.no_header))))
    print(f'Record: {manager.record.path}')
    while True:
        try:
            line = input('> ').strip()
        except EOFError:
            break
        if line.split(' ')[0][1:] in ACTIONS and line.startswith(':'):
            try:
                manager, text = action_turn(manager, line, args.datalake)
            except Exception:
                manager.record.write(event='error', line=line, traceback=traceback.format_exc())
                text = FAILED
            print(text)
        elif line:
            print(turn(manager, line))


def find(description: str, datalake: str, lake_config: str, k: int) -> Optional[Path]:
    found = candidates(description, lake_config, k, datalake)
    if not found:
        print('The natural-language seeker found no table for that description.')
        return None
    for rank, (table_id, path) in enumerate(found, 1):
        columns = pd.read_csv(path, nrows=0).columns
        print(f'  {rank}. table id {table_id}: {path.name} ({", ".join(map(str, columns))})')
    numbers = [str(rank) for rank in range(1, len(found) + 1)]
    while True:
        try:
            choice = input('Query table number, or show <number> to see its first rows: ').strip()
        except EOFError:
            return None
        if choice in numbers:
            return found[int(choice) - 1][1]
        if choice.startswith('show ') and choice[5:].strip() in numbers:
            print(pd.read_csv(found[int(choice[5:]) - 1][1], nrows=PREVIEW_ROWS, low_memory=False)
                  .to_string(max_colwidth=PREVIEW_WIDTH))
            continue
        print(f'Pick 1 to {len(found)}, or show <number>.')


def action_turn(manager: DialogueManager, line: str, datalake: Optional[str]) -> Tuple[DialogueManager, str]:
    action, number, rest = (line.split(maxsplit=2) + ['', ''])[:3]
    if not datalake:
        return manager, 'The result actions read the lake files: start the agent with --datalake <dir>.'
    if manager.card is None:
        return manager, 'There are no found tables yet.'
    if not number.isdigit() or not 1 <= int(number) <= len(manager.card.tables):
        return manager, f'Pick a table from 1 to {len(manager.card.tables)}.'
    column, columns = '', ()
    if action in (':join', ':fill'):
        column, _, named = rest.strip().partition(' by ') if action == ':fill' else ('', '', rest.strip())
        columns = tuple(given.strip() for given in named.split(',')) if named else None
        if not columns:
            keys = [key for key in (tuple(other for other in key if other != column)
                                    for key in plan_columns(manager.plan)) if key]
            if len(keys) != 1:
                usage = f':join {number} <column>[,<column>...]' if action == ':join' \
                    else f':fill {number} {column or "<column>"} by <key column>[,<key column>...]'
                return manager, (f'The plan matches on {"several sets of columns" if keys else "no column"} of '
                                 f'yours; name it, as {usage}.')
            columns = keys[0]
        unknown = [given for given in (column, *columns) if given and given not in manager.table.columns]
        if unknown:
            return manager, f'Your table has no column {unknown[0]}.'
        if action == ':fill' and not column:
            return manager, f'Name the column to fill, as :fill {number} <column>.'
    done = perform(manager, action[1:], int(number), datalake, columns, column)
    if isinstance(done, str):
        return manager, done
    fresh, text, loaded = done
    return fresh, f'{text}\n{show(loaded)}\nRecord: {fresh.record.path}'


def turn(manager: DialogueManager, line: str) -> str:
    try:
        return _turn(manager, line)
    except Exception:
        manager.record.write(event='error', line=line, traceback=traceback.format_exc())
        return FAILED


def _turn(manager: DialogueManager, line: str) -> str:
    version = manager.plan.version if manager.plan else None
    if line.startswith(':pick '):
        return show(manager.handle(pick(line.split(maxsplit=1)[1], version)))
    if line.split(' ')[0] == ':other' and len(line.split(maxsplit=2)) == 3:
        _, decision, text = line.split(maxsplit=2)
        return show(manager.handle(Other(decision, text, version=version)))
    if line.split(' ')[0] == ':notit':
        return show(manager.handle(NotIt(version=version, text=line[len(':notit'):].strip() or None)))
    if line.startswith(':k ') and line[3:].strip().isdigit() and int(line[3:]) > 0:
        return show(manager.handle(SetK(int(line[3:]), version=version)))
    if line.startswith(':') and line[1:].split(' ')[0].endswith(tuple(f'.{name}' for name in WORDING)):
        target, _, value = line[1:].partition(' ')
        part, _, slot = target.partition('.')
        value = tuple(word.strip() for word in value.split(',')) if slot == 'words' else value.strip()
        return show(manager.handle(SetWording(part, value, version=version)))
    if line.startswith(':'):
        return ('The controls are :pick <option id>[,<option id>...], :pick <question id>.none, '
                ':join <table number> [<column>,<column>...], :union <table number>, :feature <table number>, '
                ':fill <table number> <column> [by <key column>,<key column>...], :use <table number>, '
                ':other <question id> <your answer>, :notit [<what is wrong>], '
                ':k <number of tables>, :<part>.words <word>, <word> and :<part>.description <text>.')
    return show(manager.handle(Message(line)))


if __name__ == '__main__':
    main()
