from dataclasses import dataclass
from typing import Annotated, Generic, Literal, Optional, Tuple, TypeVar, Union

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, PositiveInt, field_validator, model_validator
from pydantic.json_schema import SkipJsonSchema

from src.Agent.Text import ModelText
from src.DBHandler import DBHandler

from typing import Any, Dict, Hashable, List, Set
from pydantic import ValidationInfo
from src.Agent.Profile import QueryTable

T = TypeVar('T')

Status = Literal['certain', 'ambiguous', 'not_stated']


class Frozen(BaseModel):
    model_config = ConfigDict(frozen=True)


class Slot(Frozen, Generic[T]):
    """A chosen value with its typed status; an uncertain one is assumed on the card, or preselected if low-stakes."""
    value: T
    status: Status
    alternatives: Tuple[T, ...] = Field(default=(), description='Other likely values, most likely first.')


def _alternatives_first(schema: Dict[str, Any]) -> None:
    properties = schema['properties']
    schema['properties'] = {name: properties[name] for name in ('value', 'alternatives', 'status')}
    schema['required'] = list(schema['properties'])


class ColumnSlot(Slot[T], Generic[T]):
    """A column slot as the Fillers write it: the other fitting columns, then the status, certain only without them."""
    model_config = ConfigDict(json_schema_extra=_alternatives_first)

    @model_validator(mode='before')
    @classmethod
    def _listed_is_ambiguous(cls, data: Any) -> Any:
        if isinstance(data, dict) and data.get('status') == 'certain' and data.get('alternatives'):
            return {**data, 'status': 'ambiguous'}
        return data


class KeywordFill(Frozen):
    seeker: Literal['keyword']
    words: Slot[Annotated[Tuple[str, ...], Field(min_length=1)]]

    def inputs(self, table: QueryTable) -> Tuple[List[str]]:
        return (self._tokens(),)

    def describe(self) -> str:
        return f'words {self._tokens()!r}'

    def checks(self, table: QueryTable) -> List[Tuple[str, str, List[str]]]:
        return [('words', word, [DBHandler.normalize_token(word)]) for word in self.words.value]

    def _tokens(self) -> List[str]:
        return sorted({DBHandler.normalize_token(word) for word in self.words.value} - {''})


class RowsWhere(Frozen):
    """The query table's rows a fill searches: those whose column is empty, or those where it has a value."""
    column: str
    cell: Literal['empty', 'value']

    def selects(self, table: QueryTable) -> pd.Series:
        empty = table.df[self.column].map(DBHandler.normalize_token) == ''
        return empty if self.cell == 'empty' else ~empty

    def describe(self) -> str:
        return f'the rows whose {self.column} {"is empty" if self.cell == "empty" else "has a value"}'


ROWS_DESCRIPTION = ("Rows the request gives, to search instead of the query table's values: each row a value per "
                    "column, in the columns' order. Empty when the search uses the query table's values.")


class Rows:
    def input(self, table: QueryTable) -> pd.DataFrame:
        if self.rows:
            return pd.DataFrame(typed_rows(self), columns=list(self.row_columns())).map(DBHandler.normalize_token)
        if self.rows_where is None:
            return table.normalized(self.row_columns())
        return table.df.loc[self.rows_where.selects(table), list(self.row_columns())].map(DBHandler.normalize_token)

    def tokens(self, column: str, table: QueryTable) -> List[str]:
        return [token for token in dict.fromkeys(self.input(table)[column]) if token]

    def _checks(self, slot: str, table: QueryTable) -> List[Tuple[str, str, List[str]]]:
        return [('rows' if self.rows else slot, column, self.tokens(column, table)) for column in self.row_columns()]

    def _typed(self) -> str:
        if self.rows_where is not None:
            return f', {self.rows_where.describe()}'
        return f', typed rows {typed(self)}' if self.rows else ''


class SingleColumnFill(Rows, Frozen):
    seeker: Literal['single_column']
    column: Slot[str]
    rows: Tuple[Tuple[str, ...], ...] = Field((), description=ROWS_DESCRIPTION)
    rows_where: SkipJsonSchema[Optional[RowsWhere]] = None

    @field_validator('rows')
    @classmethod
    def _rows_aligned(cls, rows: Tuple[Tuple[str, ...], ...], info: ValidationInfo) -> Tuple[Tuple[str, ...], ...]:
        return _aligned(rows, info)

    @field_validator('rows_where')
    @classmethod
    def _one_source(cls, rows_where: Optional[RowsWhere], info: ValidationInfo) -> Optional[RowsWhere]:
        return _one_source(rows_where, info)

    def row_columns(self) -> Tuple[str, ...]:
        return (self.column.value,)

    def inputs(self, table: QueryTable) -> Tuple[List[str]]:
        return (self.tokens(self.column.value, table),)

    def describe(self) -> str:
        return f'column {self.column.value}{self._typed()}'

    def checks(self, table: QueryTable) -> List[Tuple[str, str, List[str]]]:
        return self._checks('column', table)


class MultiColumnFill(Rows, Frozen):
    seeker: Literal['multi_column']
    columns: Slot[Annotated[Tuple[str, ...], Field(min_length=2)]]
    rows: Tuple[Tuple[str, ...], ...] = Field((), description=ROWS_DESCRIPTION)
    rows_where: SkipJsonSchema[Optional[RowsWhere]] = None

    @field_validator('rows')
    @classmethod
    def _rows_aligned(cls, rows: Tuple[Tuple[str, ...], ...], info: ValidationInfo) -> Tuple[Tuple[str, ...], ...]:
        return _aligned(rows, info)

    @field_validator('rows_where')
    @classmethod
    def _one_source(cls, rows_where: Optional[RowsWhere], info: ValidationInfo) -> Optional[RowsWhere]:
        return _one_source(rows_where, info)

    def row_columns(self) -> Tuple[str, ...]:
        return self.columns.value

    def inputs(self, table: QueryTable) -> Tuple[pd.DataFrame]:
        return (self.input(table),)

    def describe(self) -> str:
        return f'columns {list(self.columns.value)!r}{self._typed()}'

    def checks(self, table: QueryTable) -> List[Tuple[str, str, List[str]]]:
        return self._checks('columns', table)


def _aligned(rows: Tuple[Tuple[str, ...], ...], info: ValidationInfo) -> Tuple[Tuple[str, ...], ...]:
    slot = info.data.get('columns') or info.data.get('column')
    columns = slot.value if 'columns' in info.data else (slot.value,) if slot else ()
    if columns and any(len(row) != len(columns) for row in rows):
        raise ValueError(f'each typed row needs one value per column of {columns}')
    return rows


def _one_source(rows_where: Optional[RowsWhere], info: ValidationInfo) -> Optional[RowsWhere]:
    if rows_where is not None and info.data.get('rows'):
        raise ValueError("a fill searches typed rows or a selection of the query table's rows, not both")
    return rows_where


def typed(fill: Any) -> str:
    return ', '.join(repr(row[0]) if len(row) == 1 else repr(row) for row in getattr(fill, 'rows', ()))


def typed_rows(fill: Any) -> List[Dict[str, str]]:
    return [dict(zip(fill.row_columns(), row)) for row in getattr(fill, 'rows', ())]


class CorrelationFill(Frozen):
    seeker: Literal['correlation']
    join: Slot[str]
    target: Slot[str]

    def inputs(self, table: QueryTable) -> Tuple[List[str], List[Any]]:
        return table.normalized([self.join.value])[self.join.value].tolist(), table.df[self.target.value].tolist()

    def describe(self) -> str:
        return f'join {self.join.value}, target {self.target.value}'

    def checks(self, table: QueryTable) -> List[Tuple[str, str, List[str]]]:
        return [('join', self.join.value, table.tokens(self.join.value))]


class NaturalLanguageFill(Frozen):
    seeker: Literal['natural_language']
    description: Slot[str]

    def inputs(self, table: QueryTable) -> Tuple[str]:
        return (self.description.value,)

    def describe(self) -> str:
        return f'description {self.description.value!r}'

    def checks(self, table: QueryTable) -> List[Tuple[str, str, List[str]]]:
        return []


Seeker = Literal['keyword', 'single_column', 'multi_column', 'correlation', 'natural_language']

Fill = Annotated[Union[KeywordFill, SingleColumnFill, MultiColumnFill, CorrelationFill, NaturalLanguageFill],
                 Field(discriminator='seeker')]

WORDING = ('words', 'description')

REQUESTED = ('words', 'rows')

COLUMN_SLOTS = ('column', 'columns', 'join', 'target')


class Part(Frozen):
    """One seeker of the plan: the request's words it covers and the Filler's typed fill, values unnormalized."""
    id: str
    words: ModelText
    fill: Fill
    seeker_status: Status = 'certain'
    k: Optional[PositiveInt] = Field(None, description="The k rule's k; none when the plan's k cuts it.")


class Combiner(Frozen):
    id: str
    kind: Literal['intersection', 'union', 'counter', 'difference']
    inputs: Tuple[Union[Part, 'Combiner'], ...] = Field(min_length=2)
    removal: Optional[Literal['exclusion', 'comparison']] = None
    k: Optional[PositiveInt] = Field(None, description="The k rule's k; none when the plan's k cuts it.")


def walk(node: Union[Part, Combiner]) -> List[Union[Part, Combiner]]:
    if isinstance(node, Part):
        return [node]
    return [inner for child in node.inputs for inner in walk(child)] + [node]


def ranked_by(node: Combiner, ranking: Dict[str, str]) -> Union[Part, Combiner]:
    return next(child for child in node.inputs if child.id == ranking[node.id]) if node.kind == 'intersection' \
        else node.inputs[0]


def repeating_seekers(node: Union[Part, Combiner], ranking: Dict[str, str]) -> Set[str]:
    if isinstance(node, Part):
        return {node.fill.seeker} & {'single_column', 'correlation'}
    if node.kind in ('intersection', 'difference'):
        return repeating_seekers(ranked_by(node, ranking), ranking)
    return set()


def unranked(node: Union[Part, Combiner], ranking: Dict[str, str]) -> bool:
    if isinstance(node, Part) or node.kind == 'counter':
        return False
    return node.kind == 'union' or unranked(ranked_by(node, ranking), ranking)


TemplateId = Literal['single_column_join', 'multi_column_join', 'keyword', 'correlation', 'nl_search', 'hybrid_nl',
                     'union_search', 'imputation', 'dependent_data', 'feature_for_ml', 'collinearity',
                     'negative_examples']


GOALS: Dict[TemplateId, str] = {
    'single_column_join': 'tables to join with on one column',
    'multi_column_join': 'tables to join with on several columns together',
    'keyword': 'tables that contain given words',
    'correlation': 'tables with a column whose numbers go along with one of yours',
    'nl_search': 'tables about a described subject',
    'hybrid_nl': 'tables about a described subject that also join on one of your columns',
    'union_search': 'more rows like yours',
    'imputation': 'fill in the missing values of a column, or add a column by example',
    'dependent_data': 'tables that join on two columns separately, or on two others',
    'feature_for_ml': 'features to predict a numeric column',
    'collinearity': 'features to predict a numeric column, from tables that also hold several of your columns together',
    'negative_examples': 'tables with some of your rows but not the rows given as negative examples',
}

Place = Tuple[str, str, Optional[int]]
SHARED: Dict[str, Dict[str, Tuple[Place, ...]]] = {
    'imputation': {'key': (('p2', 'column', None), ('p1', 'columns', 0)),
                   'fill': (('p1', 'columns', 1), ('p1', 'rows_where', None), ('p2', 'rows_where', None))},
    'feature_for_ml': {'join': (('p1', 'join', None), ('p2', 'join', None))},
    'collinearity': {'join': (('p2', 'join', None), ('p3', 'join', None))},
}


class SearchPlan(Frozen):
    version: int
    k: Slot[PositiveInt]
    root: Union[Part, Combiner]
    structure: Status = 'certain'
    template: Optional[TemplateId] = None
    template_parts: Tuple[str, ...] = ()

    def parts(self) -> List[Part]:
        return [node for node in walk(self.root) if isinstance(node, Part)]

    def assumed(self) -> List[str]:
        covered = self.covered()
        slots = [f'{part.id}.{name}' for part in self.parts()
                 for name, status in [('seeker', part.seeker_status)] + [(name, value.status) for name, value in part.fill
                                                                        if isinstance(value, Slot)]
                 if status != 'certain' and name not in WORDING and f'{part.id}.{name}' not in covered]
        shared = [f't.{name}' for name in SHARED.get(self.template, {})
                  if self.shared(name).status != 'certain']
        return (['structure'] if self.structure != 'certain' else []) + slots + shared

    def preselected(self) -> Dict[str, Any]:
        wording = {f'{part.id}.{name}': value.value for part in self.parts() for name, value in part.fill
                   if name in WORDING and value.status != 'certain'}
        return ({'k': self.k.value} if self.k.status != 'certain' else {}) | wording

    def edit_targets(self) -> List[str]:
        combiner, covered = [node.id for node in walk(self.root) if isinstance(node, Combiner)], self.covered()
        return ['k'] + combiner + [target for part in self.parts()
                                   for target in [part.id] + [f'{part.id}.{name}' for name, slot in part.fill
                                                              if isinstance(slot, Slot)
                                                              and f'{part.id}.{name}' not in covered]] \
            + [f't.{name}' for name in SHARED.get(self.template, {})]

    @property
    def task(self) -> Optional[TemplateId]:
        return None if self.template_parts else self.template

    def task_ids(self) -> Dict[str, str]:
        if self.template is None:
            return {}
        ids = self.template_parts or [part.id for part in self.parts()]
        return {f'p{number}': part_id for number, part_id in enumerate(ids, 1)}

    def places(self, name: str) -> Tuple[Place, ...]:
        ids = self.task_ids()
        return tuple((ids[part_id], field, position)
                     for part_id, field, position in SHARED.get(self.template, {}).get(name, ()))

    def shared(self, name: str) -> Optional[Slot]:
        places = self.places(name)
        if not places:
            return None
        part_id, field, position = places[0]
        slot = getattr(next(part for part in self.parts() if part.id == part_id).fill, field)
        return slot if position is None else Slot(value=_held(self.root, places[0]), status=slot.status)

    def with_shared(self, name: str, slot: Slot) -> 'SearchPlan':
        root = self.root
        for index, (part_id, field, position) in enumerate(self.places(name)):
            part = next(node for node in walk(root) if isinstance(node, Part) and node.id == part_id)
            current = getattr(part.fill, field)
            if isinstance(current, RowsWhere):
                new = current.model_copy(update={'column': slot.value})
            elif position is None:
                new = slot
            else:
                value = current.value[:position] + (slot.value,) + current.value[position + 1:]
                new = current.model_copy(update={'value': value} | ({} if index else {'status': slot.status}))
            fill = type(part.fill).model_validate(part.fill.model_dump() | {field: new.model_dump()})
            root = _set(root, part.model_copy(update={'fill': fill}))
        return self.model_copy(update={'root': root})

    def covered(self) -> Set[str]:
        return {f'{part_id}.{field}' for name in SHARED.get(self.template, {})
                for part_id, field, _ in self.places(name)}


def still_task(before: SearchPlan, after: SearchPlan) -> SearchPlan:
    agree = not any(isinstance(held(after.root, after.places(name)), frozenset)
                    for name in SHARED.get(after.template, {}))
    task = _task_node(after)
    if after.template is None or agree and task is not None and _shape(task) == _shape(_task_node(before)):
        return after
    return after.model_copy(update={'template': None, 'template_parts': ()})


def _task_node(plan: SearchPlan) -> Union[None, Part, Combiner]:
    if not plan.template_parts:
        return plan.root
    return next((node for node in walk(plan.root)
                 if {inner.id for inner in walk(node) if isinstance(inner, Part)} == set(plan.template_parts)), None)


def held(root: Union[Part, Combiner], places: Tuple[Place, ...]) -> Any:
    if not places:
        return None
    values = {_held(root, place) for place in places}
    return next(iter(values)) if len(values) == 1 else frozenset(values)


def _shape(node: Union[Part, Combiner]) -> Hashable:
    if isinstance(node, Part):
        return node.id, node.fill.seeker
    return node.id, node.kind, node.removal, tuple(_shape(child) for child in node.inputs)


def _held(root: Union[Part, Combiner], place: Place) -> Any:
    part_id, field, position = place
    part = next((node for node in walk(root) if isinstance(node, Part) and node.id == part_id), None)
    value = getattr(part.fill, field, None) if part else None
    if value is None:
        return None
    if isinstance(value, RowsWhere):
        return value.column
    return value.value if position is None else value.value[position]


def _set(root: Union[Part, Combiner], part: Part) -> Union[Part, Combiner]:
    if root.id == part.id:
        return part
    if isinstance(root, Part):
        return root
    return root.model_copy(update={'inputs': tuple(_set(child, part) for child in root.inputs)})


@dataclass
class BlendNode:
    name: str
    operator: str
    inputs: List[str]
    k: int
    description: str
