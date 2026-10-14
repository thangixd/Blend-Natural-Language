import hashlib
import heapq
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from pandas.api.types import is_numeric_dtype

from src.Agent.Config import DEFAULTS
from src.DBHandler import DBHandler

from pydantic import BaseModel
from typing import Dict, List, Optional, Sequence, Tuple, Union


@dataclass
class ColumnProfile:
    name: str
    type: str
    rows: int
    distinct: int
    empty: int
    examples: List[str]
    cut: List[str]
    minimum: Optional[float] = None
    maximum: Optional[float] = None
    mean: Optional[float] = None

    def stats(self) -> str:
        line = f'{self.distinct} distinct, {self.empty} empty'
        if self.minimum is not None:
            line += f', min {self.minimum:g}, max {self.maximum:g}, mean {self.mean:g}'
        return line


class QueryTable:
    def __init__(self, df: pd.DataFrame, name: str, example_length: int = int(DEFAULTS['example_length'])) -> None:
        self.df = df
        self.name = name
        self.profile: Dict[str, ColumnProfile] = {str(column): _profile(df[column], example_length)
                                                  for column in df.columns}
        self.semantics: Dict[str, BaseModel] = {}

    @staticmethod
    def load(path: Union[str, Path], example_length: int = int(DEFAULTS['example_length']), header: bool = True,
             name: Optional[str] = None) -> 'QueryTable':
        df = pd.read_csv(path, low_memory=False) if header \
            else pd.read_csv(path, header=None, low_memory=False).add_prefix('col_')
        return QueryTable(df, name or Path(path).name, example_length)

    @property
    def columns(self) -> List[str]:
        return list(self.profile)

    @property
    def numeric_columns(self) -> List[str]:
        return [name for name, column in self.profile.items() if column.type == 'numeric']

    def option(self, columns: Union[str, Tuple[str, ...]]) -> str:
        return ', '.join(f'{column} ({", ".join(map(repr, self.profile[column].examples[:3]))})'
                         for column in ((columns,) if isinstance(columns, str) else columns))

    def tokens(self, column: str) -> List[str]:
        return [token for token in dict.fromkeys(map(DBHandler.normalize_token, self.df[column])) if token]

    def normalized(self, columns: Sequence[str]) -> pd.DataFrame:
        return self.df[list(columns)].map(DBHandler.normalize_token)

    def describe(self) -> str:
        lines = [f'Query table {self.name}: {len(self.df)} rows, {len(self.profile)} columns.']
        for column in self.profile.values():
            lines.append(f'- {column.name} ({column.type}): {column.stats()}; examples {column.examples!r}')
        return '\n'.join(lines)

    def short_context(self) -> str:
        return ', '.join(f'{column.name} ({column.type})' for column in self.profile.values())


def _profile(series: pd.Series, example_length: int) -> ColumnProfile:
    tokens = [DBHandler.normalize_token(value) for value in series]
    spellings: Dict[str, str] = {}
    for value, token in zip(series, tokens):
        if token:
            spellings.setdefault(token, str(value))
    chosen = [spellings[token] for token in
              heapq.nsmallest(5, spellings, key=lambda token: hashlib.sha1(token.encode()).hexdigest())]
    column = ColumnProfile(name=str(series.name), type='numeric' if is_numeric_dtype(series) else 'text',
                           rows=len(series), distinct=len(spellings), empty=tokens.count(''),
                           examples=[value[:example_length] + '…' if len(value) > example_length else value
                                     for value in chosen],
                           cut=[value for value in chosen if len(value) > example_length])
    if column.type == 'numeric' and series.notna().any():
        column.minimum, column.maximum, column.mean = float(series.min()), float(series.max()), float(series.mean())
    return column
