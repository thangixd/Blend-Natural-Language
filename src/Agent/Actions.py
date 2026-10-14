from dataclasses import dataclass
from itertools import product
from pathlib import Path

import pandas as pd

from src.Agent.Compiler import filenames
from src.Agent.Dialogue import DialogueManager, LoadTable
from src.Agent.Plans import CorrelationFill, MultiColumnFill, SingleColumnFill
from src.DBHandler import DBHandler
from src.Operators.Seekers import NaturalLanguage

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
from src.Agent.Dialogue import Response
from src.Agent.Plans import SearchPlan
from src.Agent.Profile import QueryTable

CANDIDATES = 5
MIN_KEYS = 3
ACTIONS = {'join': 'joined with', 'union': 'with the rows of', 'feature': 'with a feature from',
           'fill': 'filled from', 'use': ''}
NO_CORRELATION = 'Adding a feature needs a plan with one correlation, which gives the key and the target column.'


@dataclass
class Done:
    df: pd.DataFrame
    text: str
    record: Dict[str, Any]


def plan_columns(plan: SearchPlan) -> List[Tuple[str, ...]]:
    keys = []
    for part in plan.parts():
        fill = part.fill
        key = ((fill.column.value,) if isinstance(fill, SingleColumnFill)
               else tuple(fill.columns.value) if isinstance(fill, MultiColumnFill)
               else (fill.join.value,) if isinstance(fill, CorrelationFill) else None)
        if key and key not in keys:
            keys.append(key)
    return keys


def plan_correlations(plan: SearchPlan) -> List[Tuple[str, str]]:
    return list(dict.fromkeys((part.fill.join.value, part.fill.target.value) for part in plan.parts()
                              if isinstance(part.fill, CorrelationFill)))


def join(table: QueryTable, lake: pd.DataFrame, name: str, columns: Sequence[str]) -> Union[Done, str]:
    tokens = lake.map(DBHandler.normalize_token)
    found = _key(table, tokens, name, columns)
    if isinstance(found, str):
        return found
    shared, assignment = found
    keys = [f'key {i}' for i in range(len(columns))]
    left = pd.concat([table.df, table.normalized(columns).set_axis(keys, axis=1)], axis=1)
    right = lake.drop(columns=list(assignment))
    right = right.rename(columns={column: f'{column} ({name})' for column in right.columns if column in table.df})
    right[keys] = tokens[list(assignment)].to_numpy()
    right = right[(right[keys] != '').all(axis=1)]
    repeated = int(right[right.duplicated(keys, keep=False)].drop_duplicates(keys).shape[0])
    merged = left.merge(right.drop_duplicates(keys), on=keys, how='left', indicator=True)
    matched = int((merged['_merge'] == 'both').sum())
    pairs = list(zip(columns, assignment))
    unit = 'values' if len(pairs) == 1 else 'value combinations'
    text = (f'Joined {name} on {_pairs(pairs)} ({shared} shared {unit}): {matched} of your {len(merged)} rows found '
            f'a row' + (f'; {repeated} keys have several rows in {name}, each took its first.' if repeated else '.'))
    return Done(merged.drop(columns=keys + ['_merge']), text,
                {'pairs': pairs, 'shared': shared, 'matched': matched, 'repeated': repeated})


def append(table: QueryTable, lake: pd.DataFrame, name: str) -> Union[Done, str]:
    tokens = lake.map(DBHandler.normalize_token)
    ours = {column: set(table.tokens(column)) for column in table.columns}
    overlaps = sorted(((len(values & set(tokens[lake_column])), column, lake_column)
                       for column, values in ours.items() for lake_column in lake.columns),
                      key=lambda overlap: -overlap[0])
    pairs = {}
    for shared, column, lake_column in overlaps:
        if shared and column not in pairs and lake_column not in pairs.values():
            pairs[column] = lake_column
    if not pairs:
        return f'No column of {name} shares a value with a column of yours, so no rows were appended.'
    rows = lake[list(pairs.values())].set_axis(list(pairs), axis=1)
    df = pd.concat([table.df, rows], ignore_index=True)
    empty = [column for column in table.columns if column not in pairs]
    text = (f'Appended the {len(lake)} rows of {name} under your {len(table.df)}, matched {_pairs(pairs.items())}'
            + (f'; {", ".join(empty)} stay empty in them.' if empty else '.'))
    return Done(df, text, {'pairs': list(pairs.items()), 'rows': len(lake)})


def add_feature(table: QueryTable, lake: pd.DataFrame, name: str, key: str, target: str) -> Union[Done, str]:
    tokens = lake.map(DBHandler.normalize_token)
    found = _key(table, tokens, name, (key,))
    if isinstance(found, str):
        return found
    (lake_key,) = found[1]
    ours_keys = table.normalized([key])[key]
    ours = table.df[target].groupby(ours_keys).mean().drop('', errors='ignore')
    scores = {}
    for column in lake.columns:
        numbers = pd.to_numeric(lake[column], errors='coerce')
        if (column == lake_key or tokens[column].equals(tokens[lake_key])
                or not numbers.notna().any() or numbers.notna().sum() != (tokens[column] != '').sum()):
            continue
        theirs = numbers.groupby(tokens[lake_key]).mean()
        both = pd.concat([ours.rename('ours'), theirs.rename('theirs')], axis=1, join='inner').dropna()
        r = both['ours'].corr(both['theirs']) if len(both) >= MIN_KEYS else float('nan')
        if pd.notna(r):
            scores[column] = (r, len(both), theirs)
    if not scores:
        return (f'No number column of {name} correlates with your {target} over {MIN_KEYS} or more {key} values, so '
                f'no feature was added.')
    best = max(scores, key=lambda column: abs(scores[column][0]))
    r, keys, theirs = scores[best]
    added = f'{best} ({name})' if best in table.df else best
    df = table.df.copy()
    df[added] = ours_keys.map(theirs)
    text = (f'Added {added} from {name}, averaged per {key}: it {"rises" if r > 0 else "falls"} with your {target}, '
            f'r = {r:.2f} over {keys} {key} values.')
    return Done(df, text, {'column': best, 'key': [key, lake_key], 'r': r, 'keys': keys})


def fill(table: QueryTable, lake: pd.DataFrame, name: str, column: str, columns: Sequence[str]) -> Union[Done, str]:
    blank = table.normalized([column])[column] == ''
    if not blank.any():
        return f'Your column {column} has no empty cell.'
    tokens = lake.map(DBHandler.normalize_token)
    found = _key(table, tokens, name, columns)
    if isinstance(found, str):
        return found
    assignment = found[1]
    keys = [f'key {i}' for i in range(len(columns))]
    first = tokens[list(assignment)].set_axis(keys, axis=1)
    first = first[(first != '').all(axis=1)].drop_duplicates(keys).reset_index(names='row')
    row = table.normalized(columns).set_axis(keys, axis=1).merge(first, on=keys, how='left')['row']
    examples = row.notna() & ~blank
    ours = table.normalized([column])[column][examples].to_numpy()
    agree = {lake_column: int((tokens[lake_column].loc[row[examples].astype(int)].to_numpy() == ours).sum())
             for lake_column in lake.columns if lake_column not in assignment}
    best = max(agree, key=agree.get, default=None)
    if not best or not agree[best]:
        return (f'No column of {name} holds the values your {column} already has for the same '
                f'{", ".join(columns)}, so nothing was filled.')
    taken = row[row.notna() & blank].astype(int)
    taken = taken[(tokens[best].loc[taken] != '').to_numpy()]
    if taken.empty:
        return f'{name} has no {best} value for the empty {column} cells, so nothing was filled.'
    df = table.df.copy()
    df[column] = df[column].astype(object)
    df.loc[taken.index, column] = lake[best].loc[taken].to_numpy()
    marker = f'filled from {name}'
    df[marker] = ''
    df.loc[taken.index, marker] = 'yes'
    text = (f'Filled {len(taken)} of the {int(blank.sum())} empty {column} cells from {name}.{best}, matched on '
            f'{_pairs(zip(columns, assignment))}; {agree[best]} of your {int(examples.sum())} {column} values with a '
            f'match agree with it. The column "{marker}" marks the filled cells.')
    return Done(df, text, {'column': best, 'pairs': list(zip(columns, assignment)), 'filled': len(taken),
                           'agree': agree[best], 'examples': int(examples.sum())})


def candidates(description: str, lake_config: Union[str, Path], k: int,
               datalake: Union[str, Path]) -> List[Tuple[int, Path]]:
    NaturalLanguage.DB.load_config(lake_config)
    ids = NaturalLanguage(description, k).run()
    return list(zip(ids, (Path(datalake) / name for name in filenames(NaturalLanguage.DB, ids))))


def perform(manager: DialogueManager, action: str, rank: int, datalake: Union[str, Path], columns: Sequence[str] = (),
            column: Optional[str] = None) -> Union[Tuple[DialogueManager, str, Response], str]:
    table_id, name = manager.card.tables[rank - 1]
    if action == 'use':
        return _session(manager, Path(datalake) / name, name, f'{name} is the query table now; a new session '
                                                               f'continues on it.')
    lake = pd.read_csv(Path(datalake) / name, low_memory=False)
    if action == 'union':
        done = append(manager.table, lake, name)
    elif action == 'feature':
        correlations = plan_correlations(manager.plan)
        done = add_feature(manager.table, lake, name, *correlations[0]) if len(correlations) == 1 else NO_CORRELATION
    elif action == 'join':
        done = join(manager.table, lake, name, columns)
    else:
        done = fill(manager.table, lake, name, column, columns)
    if isinstance(done, str):
        return done
    path = manager.record.path.with_name(f'{manager.record.path.stem}-{action}-{table_id}.csv')
    done.df.to_csv(path, index=False)
    manager.record.write(event=action, table_id=table_id, name=name, path=str(path), **done.record)
    return _session(manager, path, f'{manager.table.name} {ACTIONS[action]} {name}',
                    f'{done.text} Saved to {path}; a new session continues on it.')


def _session(manager: DialogueManager, path: Path, name: str, text: str) -> Tuple[DialogueManager, str, Response]:
    fresh = DialogueManager(manager.config, manager.lake_config, manager.models)
    return fresh, text, fresh.handle(LoadTable(path, name=name))


def _key(table: QueryTable, tokens: pd.DataFrame, name: str,
         columns: Sequence[str]) -> Union[Tuple[int, Tuple[str, ...]], str]:
    candidates = []
    for column in columns:
        ours = set(table.tokens(column))
        overlap = {lake_column: len(ours & set(tokens[lake_column])) for lake_column in tokens.columns}
        candidates.append([lake_column for lake_column in sorted(overlap, key=overlap.get, reverse=True)[:CANDIDATES]
                           if overlap[lake_column]])
        if not candidates[-1]:
            return f'No column of {name} shares a value with your column {column}, so nothing was done.'
    ours = {key for key in table.normalized(columns).itertuples(index=False, name=None) if all(key)}
    scored = [(len(ours & set(tokens[list(assignment)].itertuples(index=False, name=None))), assignment)
              for assignment in product(*candidates) if len(set(assignment)) == len(assignment)]
    shared, assignment = max(scored, key=lambda entry: entry[0], default=(0, ()))
    if not shared:
        return f'No row of {name} holds values of your columns {", ".join(columns)} together, so nothing was done.'
    return shared, assignment


def _pairs(pairs) -> str:
    return ', '.join(f'{ours} = {theirs}' for ours, theirs in pairs)
