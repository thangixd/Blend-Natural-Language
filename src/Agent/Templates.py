from collections import Counter
from dataclasses import dataclass
from itertools import count, takewhile
from typing import Literal, Optional, Tuple

from openai import OpenAI, OpenAIError
from pydantic import BaseModel, Field, create_model
from pydantic_ai import NativeOutput

from src.Agent import Planner
from src.Agent.Plans import ROWS_DESCRIPTION, ColumnSlot, Combiner, Frozen, Part, RowsWhere, Slot
from src.Agent.Sampling import canonical
from src.Agent.Text import ModelText
from src.DBHandler import DBHandler
from src.NLSeeker.Config import NLSeekerConfig
from src.Tasks.AugmentationByExample import AugmentationByExample
from src.Tasks.CorrelationSearch import CorrelationSearch
from src.Tasks.DataImputation import DataImputation
from src.Tasks.DependentDataSearch import DependentDataSearch
from src.Tasks.FeatureForMLSearch import FeatureForMLSearch
from src.Tasks.HybridNLSearch import HybridNLSearch
from src.Tasks.KeywordSearch import KeywordSearch
from src.Tasks.MultiColumnCollinearitySearch import MultiColumnCollinearitySearch
from src.Tasks.MultiColumnJoinSearch import MultiColumnJoinSearch
from src.Tasks.NegativeExampleSearch import NegativeExampleSearch
from src.Tasks.NLSearch import NLSearch
from src.Tasks.SingleColumnJoinSearch import SingleColumnJoinSearch
from src.Tasks.UnionSearch import UnionSearch

from typing import Any, Callable, Dict, Hashable, List, Sequence, Union
from src.Agent.Plans import SearchPlan, TemplateId
from src.Agent.Profile import QueryTable
from src.Plan import Plan

NO_GAPS = '{} has no empty cells, so there is nothing to fill.'
NO_KEYS = '{} has no value in the rows whose {} is empty, so there is nothing to look up.'
KEY_IS_FILL = '{} cannot be both the key and the column to fill.'
NOT_NUMERIC = '{} is not numeric, so nothing can correlate with it.'
TWICE = 'The pair names {} twice.'
TARGET_IS_FEATURE = '{} cannot be both the target and the existing feature: the difference would remove what it finds.'
SAME_PAIRS = 'The two pairs name the same columns, {}, so the union would search one pair twice.'
NO_NATURAL_LANGUAGE = 'The natural-language search is not available: {}.'
SAME_SIDES = ('The negative examples have the columns of the rows to keep ({}) and no typed rows of their own, so they '
              'would remove the tables found.')


@dataclass(frozen=True)
class Template:
    tasks: Tuple[Callable[..., Plan], ...]
    slots: Dict[str, str]
    expand: Callable[[Any, ModelText], Union[Part, Combiner]]
    read: Callable[[Callable[[str, str], Any]], Dict[str, Any]]


SLOTS = {'column': 'the column of the query table to join on', 'columns': 'the columns of the query table whose values '
         'the tables hold', 'words': 'the words to find', 'join': 'the column of the query table to join on',
         'target': 'the numeric column to correlate with', 'description': 'the subject of the tables, in the '
         "request's words", 'key': 'the column that identifies a row, to look the missing values up by', 'fill': 'the '
         'column whose empty cells are to be filled', 'first': 'the first two columns to join on separately',
         'second': 'the other two columns to join on separately', 'feature': 'the existing feature: a numeric column '
         'whose best correlations are left out', 'exclusive': 'the columns of the rows the tables should hold',
         'inclusive': 'the columns of the negative examples', 'exclusive_rows': 'rows the request gives for the '
         'exclusive columns, if any', 'inclusive_rows': 'the negative examples the request gives as rows, if any'}


COPIED = ('words', 'rows')


class DoesNotFit(BaseModel):
    """The answer that the request's searches cannot be put into this task's slots."""
    rationale: ModelText


def _slot_types(template: TemplateId, table: QueryTable) -> Dict[str, Any]:
    column = Literal[tuple(table.columns)]
    types = {'pair': (ColumnSlot[Tuple[column, column]], ...), 'target': (ColumnSlot[column], ...),
             'rows': (Tuple[Tuple[str, ...], ...], Field((), description=ROWS_DESCRIPTION))}
    return {name: types.get(kind) or (Planner.column_slot(table, kind), ...)
            for name, kind in TEMPLATES[template].slots.items()}


def slots_type(template: TemplateId, table: QueryTable) -> type:
    return create_model(template, __base__=Frozen, **_slot_types(template, table))


def filling_type(template: TemplateId, table: QueryTable) -> NativeOutput:
    return NativeOutput([create_model('TemplateFilling', __base__=Frozen, rationale=(ModelText, ...),
                                      **_slot_types(template, table)), DoesNotFit])


def filling_key(template: TemplateId) -> Callable[[BaseModel], Hashable]:
    def key(output: BaseModel) -> Hashable:
        if isinstance(output, DoesNotFit):
            return 'does_not_fit'
        return tuple(sorted((name, canonical(getattr(output, name).value)) for name, kind
                            in TEMPLATES[template].slots.items() if kind not in (*COPIED, 'description')))
    return key


def first_meeting(template: TemplateId, group: List[BaseModel], table: QueryTable, db: DBHandler
                  ) -> Tuple[List[BaseModel], List[Tuple[str, str]]]:
    found = checks(template, group[0], table, db)
    meeting = next((output for output in group[1:] if not checks(template, output, table, db)), None) if found else None
    if meeting is None:
        return group, found
    return [meeting, *[output for output in group if output is not meeting]], []


def copies_differ(template: TemplateId, group: List[BaseModel]) -> List[str]:
    return [name for name, kind in TEMPLATES[template].slots.items()
            if kind in COPIED and len({_key(kind, getattr(output, name)) for output in group}) > 1]


def _key(kind: str, slot: Any) -> Hashable:
    if kind == 'rows':
        return tuple(sorted({tuple(DBHandler.normalize_token(value) for value in row) for row in slot}))
    return tuple(sorted({DBHandler.normalize_token(word) for word in slot.value}))


def readings(template: TemplateId, groups: List[List[BaseModel]], table: QueryTable, db: DBHandler
             ) -> Tuple[BaseModel, List[Tuple[int, BaseModel, bool]]]:
    outputs, proposal = [output for group in groups for output in group], groups[0][0]
    others, disputed = {}, {}
    for name, kind in TEMPLATES[template].slots.items():
        if kind in ('words', 'description', 'rows'):
            continue
        slot = getattr(proposal, name)
        if kind != 'pair' and Planner.one_value(table, kind):
            disputed[name] = slot.model_copy(update={'status': 'certain'})
            continue
        others[name] = Planner.tally([[getattr(output, name).value, *getattr(output, name).alternatives]
                                      for output in outputs], canonical, slot.value)
        if slot.status == 'certain' and (others[name] or any(getattr(output, name).status != 'certain'
                                                             for output in outputs)):
            disputed[name] = slot.model_copy(update={'status': 'ambiguous'})
    proposal = proposal.model_copy(update=disputed)
    found = [_with(proposal, {name: value}) for name, values in others.items() for _, value, _ in values]
    for output in outputs:
        changed = {name: getattr(output, name).value for name in others
                   if canonical(getattr(output, name).value) != canonical(getattr(proposal, name).value)}
        found += [_with(proposal, changed)] if len(changed) > 1 else []
    key, distinct = filling_key(template), {}
    chosen = Counter(key(output) for output in outputs)
    for reading in found:
        changed = [name for name in others if getattr(reading, name).value != getattr(proposal, name).value]
        if not facts(template, values(template, reading), table, db, changed):
            distinct.setdefault(key(reading), reading)
    return proposal, [(chosen[reading_key], reading, not chosen[reading_key])
                      for reading_key, reading in distinct.items()]


def _with(slots: BaseModel, changed: Dict[str, Any]) -> BaseModel:
    return slots.model_copy(update={name: getattr(slots, name).model_copy(update={'value': value, 'alternatives': ()})
                                    for name, value in changed.items()})


def values(template: TemplateId, slots: BaseModel) -> Dict[str, Any]:
    return {name: getattr(slots, name) if kind == 'rows' else getattr(slots, name).value
            for name, kind in TEMPLATES[template].slots.items()}


def checks(template: TemplateId, slots: BaseModel, table: QueryTable, db: DBHandler) -> List[Tuple[str, str]]:
    return facts(template, values(template, slots), table, db)


def facts(template: TemplateId, filled: Dict[str, Any], table: QueryTable, db: DBHandler,
          only: Optional[Sequence[str]] = None) -> List[Tuple[str, str]]:
    kinds, found = TEMPLATES[template].slots, []
    for name, kind in kinds.items():
        if kind == 'target' and filled[name] not in table.numeric_columns:
            found.append((name, NOT_NUMERIC.format(filled[name])))
        elif kind == 'pair' and filled[name][0] == filled[name][1]:
            found.append((name, TWICE.format(filled[name][0])))
    if template == 'imputation':
        found += _imputation_checks(filled['key'], filled['fill'], table)
    if template == 'negative_examples':
        found += _negative_checks(filled)
    if template in ('feature_for_ml', 'collinearity') and filled['target'] == filled['feature']:
        found.append(('feature', TARGET_IS_FEATURE.format(filled['feature'])))
    if template == 'dependent_data' and set(filled['first']) == set(filled['second']):
        found.append(('second', SAME_PAIRS.format(', '.join(filled['second']))))
    if 'description' in kinds.values() and (only is None or 'description' in only):
        why = _natural_language(db)
        found += [('description', NO_NATURAL_LANGUAGE.format(why))] if why else []
    return [(name, fact) for name, fact in found
            if only is None or name in only or template in ('imputation', 'negative_examples')]


def read(plan: SearchPlan) -> Dict[str, Any]:
    parts, ids = {part.id: part for part in plan.parts()}, plan.task_ids()

    def at(part_id: str, field: str) -> Any:
        part = parts.get(ids.get(part_id))
        value = getattr(part.fill, field) if part else None
        return value.value if isinstance(value, Slot) else value

    return TEMPLATES[plan.template].read(at)


def _imputation_checks(key: str, fill: str, table: QueryTable) -> List[Tuple[str, str]]:
    if key == fill:
        return [('fill', KEY_IS_FILL.format(fill))]
    gaps = RowsWhere(column=fill, cell='empty').selects(table)
    if not gaps.any():
        return [('fill', NO_GAPS.format(fill))]
    if not table.normalized([key])[key][gaps].any():
        return [('key', NO_KEYS.format(key, fill))]
    return []


def _negative_checks(filled: Dict[str, Any]) -> List[Tuple[str, str]]:
    removed, kept = filled['inclusive_rows'], filled['exclusive_rows']
    if set(filled['inclusive']) == set(filled['exclusive']) \
            and (not removed or _key('rows', removed) == _key('rows', kept)):
        return [('inclusive_rows', SAME_SIDES.format(', '.join(filled['inclusive'])))]
    return []


def _natural_language(db: DBHandler) -> Optional[str]:
    try:
        config = NLSeekerConfig.load(db.config_path)
    except KeyError:
        return 'the lake config has no natural-language settings'
    if not db.execute_and_fetchall('SELECT 1 FROM information_schema.tables WHERE table_name = ?',
                                   [config.documents_table]):
        return 'the lake has no natural-language index'
    try:
        for url, key in ((config.llm_base_url, config.llm_api_key),
                         (config.embedding_base_url, config.embedding_api_key)):
            OpenAI(base_url=url, api_key=key, timeout=5, max_retries=0).models.list()
    except OpenAIError:
        return 'its endpoints do not answer'
    return None


def expand(template: TemplateId, slots: BaseModel, words: ModelText) -> Union[Part, Combiner]:
    return TEMPLATES[template].expand(slots, words)


def _part(part_id: str, words: ModelText, seeker: str, /, **fill: Any) -> Part:
    return Part(id=part_id, words=words, fill={'seeker': seeker, **{
        name: value.model_dump() if isinstance(value, BaseModel) else value for name, value in fill.items()}})


def _taken(value: Any, *slots: Slot) -> Dict[str, Any]:
    return {'value': value, 'status': next((slot.status for slot in slots if slot.status != 'certain'), 'certain')}


def _joins(columns: Tuple[str, ...], slot: Slot, first: int, words: ModelText) -> Tuple[Part, ...]:
    return tuple(_part(f'p{number}', words, 'single_column', column=_taken(column, slot))
                 for number, column in enumerate(columns, first))


def _union(at: Callable[[str, str], Any]) -> Dict[str, Any]:
    return {'columns': tuple(takewhile(lambda column: column is not None,
                                       (at(f'p{number}', 'column') for number in count(1))))}


def _imputation(slots: BaseModel, words: ModelText) -> Combiner:
    fill = slots.fill.value
    examples = _part('p1', words, 'multi_column',
                     columns={'value': (slots.key.value, fill), 'status': slots.fill.status},
                     rows_where={'column': fill, 'cell': 'value'})
    queries = _part('p2', words, 'single_column', column=slots.key, rows_where={'column': fill, 'cell': 'empty'})
    return Combiner(id='c1', kind='intersection', inputs=(examples, queries))


def _dependent(slots: BaseModel, words: ModelText) -> Combiner:
    return Combiner(id='c1', kind='union', inputs=(
        Combiner(id='c2', kind='intersection', inputs=_joins(slots.first.value, slots.first, 1, words)),
        Combiner(id='c3', kind='intersection', inputs=_joins(slots.second.value, slots.second, 3, words))))


def _feature(combiner_id: str, first: int, slots: BaseModel, words: ModelText) -> Combiner:
    return Combiner(id=combiner_id, kind='difference', removal='comparison', inputs=(
        _part(f'p{first}', words, 'correlation', join=slots.join, target=slots.target),
        _part(f'p{first + 1}', words, 'correlation', join=slots.join, target=slots.feature)))


TEMPLATES: Dict[TemplateId, Template] = {
    'single_column_join': Template((SingleColumnJoinSearch,), {'column': 'column'}, lambda slots, words: _part(
        'p1', words, 'single_column', column=slots.column), lambda at: {'column': at('p1', 'column')}),
    'multi_column_join': Template((MultiColumnJoinSearch,), {'columns': 'columns'}, lambda slots, words: _part(
        'p1', words, 'multi_column', columns=slots.columns), lambda at: {'columns': at('p1', 'columns')}),
    'keyword': Template((KeywordSearch,), {'words': 'words'}, lambda slots, words: _part(
        'p1', words, 'keyword', words=slots.words), lambda at: {'words': at('p1', 'words')}),
    'correlation': Template((CorrelationSearch,), {'join': 'column', 'target': 'target'}, lambda slots, words: _part(
        'p1', words, 'correlation', join=slots.join, target=slots.target),
        lambda at: {'join': at('p1', 'join'), 'target': at('p1', 'target')}),
    'nl_search': Template((NLSearch,), {'description': 'description'}, lambda slots, words: _part(
        'p1', words, 'natural_language', description=slots.description),
        lambda at: {'description': at('p1', 'description')}),
    'hybrid_nl': Template((HybridNLSearch,), {'column': 'column', 'description': 'description'},
                          lambda slots, words: Combiner(id='c1', kind='intersection', inputs=(
                              _part('p1', words, 'single_column', column=slots.column),
                              _part('p2', words, 'natural_language', description=slots.description))),
                          lambda at: {'column': at('p1', 'column'), 'description': at('p2', 'description')}),
    'union_search': Template((UnionSearch,), {'columns': 'columns'}, lambda slots, words: Combiner(
        id='c1', kind='union', inputs=_joins(slots.columns.value, slots.columns, 1, words)), _union),
    'imputation': Template((DataImputation, AugmentationByExample), {'key': 'column', 'fill': 'column'},
                           _imputation, lambda at: {'key': at('p2', 'column'), 'fill': at('p1', 'columns')[1]}),
    'dependent_data': Template((DependentDataSearch,), {'first': 'pair', 'second': 'pair'}, _dependent,
                               lambda at: {'first': (at('p1', 'column'), at('p2', 'column')),
                                           'second': (at('p3', 'column'), at('p4', 'column'))}),
    'feature_for_ml': Template((FeatureForMLSearch,), {'join': 'column', 'target': 'target', 'feature': 'target'},
                               lambda slots, words: _feature('c1', 1, slots, words),
                               lambda at: {'join': at('p1', 'join'), 'target': at('p1', 'target'),
                                           'feature': at('p2', 'target')}),
    'collinearity': Template((MultiColumnCollinearitySearch,), {'columns': 'columns', 'join': 'column',
                                                                 'target': 'target', 'feature': 'target'},
                             lambda slots, words: Combiner(id='c1', kind='intersection', inputs=(
                                 _part('p1', words, 'multi_column', columns=slots.columns),
                                 _feature('c2', 2, slots, words))),
                             lambda at: {'columns': at('p1', 'columns'), 'join': at('p2', 'join'),
                                         'target': at('p2', 'target'), 'feature': at('p3', 'target')}),
    'negative_examples': Template((NegativeExampleSearch,), {'exclusive': 'columns', 'inclusive': 'columns',
                                                             'exclusive_rows': 'rows', 'inclusive_rows': 'rows'},
                                  lambda slots, words: Combiner(
                                      id='c1', kind='difference', removal='exclusion', inputs=(
                                          _part('p1', words, 'multi_column', columns=slots.exclusive,
                                                rows=slots.exclusive_rows),
                                          _part('p2', words, 'multi_column', columns=slots.inclusive,
                                                rows=slots.inclusive_rows))),
                                  lambda at: {'exclusive': at('p1', 'columns'), 'inclusive': at('p2', 'columns'),
                                              'exclusive_rows': at('p1', 'rows'), 'inclusive_rows': at('p2', 'rows')}),
}
