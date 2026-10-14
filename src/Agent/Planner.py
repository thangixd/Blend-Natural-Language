import re
from collections import Counter
from typing import Annotated, Any, List, Literal, Optional, Tuple, Union

from pydantic import AfterValidator, BaseModel, Field, PositiveInt, create_model, model_validator
from pydantic_ai import Agent, NativeOutput

from src.Agent.Plans import (COLUMN_SLOTS, GOALS, ColumnSlot, Combiner, CorrelationFill, Frozen, KeywordFill,
                             MultiColumnFill, NaturalLanguageFill, Part, Seeker, SingleColumnFill, Slot, Status,
                             TemplateId, WORDING, typed_rows, walk)
from src.Agent.Sampling import canonical
from src.Agent.Text import Line, ModelText, joined
from src.DBHandler import DBHandler

from typing import Callable, Dict, Hashable, Sequence
from src.Agent.Plans import SearchPlan
from src.Agent.Profile import QueryTable

AMBIGUOUS = """Mark a field ambiguous only if a typical user could reasonably mean either reading and nothing in the
request or the query table decides it. Then return the most likely reading as the proposal and each other one as an
alternative. If one reading is the usual meaning of the words, choose it and mark the field certain."""

COLUMN_ALTERNATIVES = """For a column slot, first list as its alternatives every other column of the query table that
the request's words could refer to, most likely first; then give its status, certain only when that list is empty, even
when one column is the usual meaning of the words."""

TREE = """Return the tree: one part with the words of the request it covers, verbatim, or a combiner over parts:
intersection (tables that fit all of them), union (any of them), counter (most of them), or difference (the tables
of the kept part without those of the removed part). A combiner's inputs may themselves be combiners over parts, so
a tree has at most two levels of combiners. A difference is an exclusion when the removed part names what
the tables must not have ("but not tables that mention Paris"), and a comparison when the removed part is a search
of its own whose best matches are taken away ("tables joining on my cities that are not among the best matches for
my postal codes").
What each shape finds, in the user's terms:
- one part over several columns of the query table: tables whose rows hold those values together, in the same rows;
- an intersection of parts, one per column: tables that have each kind of values, not necessarily in the same rows;
- a union: tables that have any one of the parts' values;
- several values for one keyword search ("containing A or B"): one part with all the words, not a union;
- a counter: tables that have most of them, those with the most first;
- a difference: the tables the kept part finds that the removed part does not; the kept part is what the user wants,
  the removed part what they want left out.
A join on several columns ("join on A and B", "match on both A and B") means one multi-column search: the values must
appear together in the same rows.
A correlation always joins on a column of the query table. When a request asks for a correlation and also names a
column the tables join, match or share values on, that column is the correlation's join column, wherever its words
stand: one correlation part, not an intersection of a join and a correlation. The join is a part of its own only
when the correlation names another join column or the request says the join is separate. So when you split a list
of conditions, check each item that only names a column to join, match or key on: if another item is a correlation
that names no join column, those two items are one correlation part, and only the remaining items are parts of their
own ("a join via A, some value that grows and shrinks as my B does, nothing about X" forms a difference: one
correlation on B with A as its join, after dropping each table naming X). Listing conditions with "and" does not make
the join separate.
"""

TASKS = """First decide whether the request is one of Blend's tasks. Then answer the task, {"template": "<id>"}, in
place of the tree; its parts are filled later. The tasks, by what the user wants:
- single_column_join: tables to join with on one column of the query table.
- multi_column_join: tables to join with on several columns together, their values in the same rows.
- keyword: tables that contain given words or values anywhere.
- correlation: tables with a column whose numbers go along with a numeric column of the query table, joined on a
  column of it.
- nl_search: tables about a subject the request describes in words.
- hybrid_nl: tables about a described subject that also join with one column of the query table.
- union_search: more rows like the query table's: tables that could extend it with further rows of the same kind.
- imputation: fill in the missing values of a column of the query table, or add a column to it by example (the user
  typed some values, the rest are empty): tables that hold the complete rows' key with its value and also the keys
  of the rows to fill.
- dependent_data: tables that join on two columns separately, or on two other columns separately: a union of two
  such pairs, only when the request asks for exactly that.
- feature_for_ml: find features to predict a numeric column: tables whose numbers correlate with that target, without
  the best correlations with a feature the user already has.
- collinearity: such features, from tables that also hold several columns of the query table together.
- negative_examples: tables that hold some rows of the query table's values but not the rows the request gives as
  negative examples.
Choose a task when it holds the whole search of the request. k and words with no search meaning are no reason for a
tree: "the top 50 tables I can join with on my cities" is single_column_join with k 50. Write a tree instead when the
request adds a search constraint the task has no slot for ("tables I can join with on my cities that mention
Seattle"), names values to find in a column of its own ("tables with a column holding the cities Paris or Berlin"),
combines searches in a way no task does, or is worded only by its structure, without asking to fill in or add values
("tables that hold my address lines with their cities and that join on my postal codes").
A task built from several searches (imputation, union_search, hybrid_nl, feature_for_ml, negative_examples) that comes
with such a constraint stays whole, as one part of the tree: write {"template": "<id>", "words": "<the words of the
request the task covers>"} in the place of a part and combine it with the constraint's parts. A tree holds at most one
such task part, always as an input of its root combiner; a task of one search is written as a part like any other.
Joins: one join is single_column_join; columns joined together ("join on A and B", "match on both A and B") are
multi_column_join. Two joins the request keeps separate ("join on A and, separately, on B"; "join on A or on B") are no
task: write the intersection or union as a tree.
When a task and a tree, or two tasks, are both likely readings, answer the most likely and list the other in readings.
"""

DECOMPOSER_INSTRUCTIONS = """You split a request for tables from a data lake into the parts Blend searches for.
""" + TASKS + TREE + """Only when the tree is ambiguous, list its other readings in readings, most likely first; otherwise leave readings
empty.
""" + AMBIGUOUS + """
Give each part an id, p1, p2, ... in the order of the tree. A reading names a part of the tree by its id; a part
that is in no other tree gets an id of its own. When the view lists the parts of the plan shown or parts the user
dropped, a part you mean the same way keeps its id, and a new part takes a number none of them has.
Set fit to fits when the tree holds the whole request. Set it to too_deep when the request needs more than two
levels of combiners, or to no_combiner when it relates its parts in a way none of the four combiners expresses; then
the tree is the largest part of the request that fits, and sub_plans lists the other parts that fit, largest first.
List the word groups of the request that no part covers, verbatim, in unplaced: a constraint on the tables that
no part expresses, or words with no search meaning ("please", "I need").
Give k only if the request states how many tables it wants, with its status: certain if stated plainly, ambiguous
if vague ("a few", "about 10"), not_stated if the request does not say."""

SEEKERS = """- keyword: tables containing the given words, as the request states them.
- natural_language: tables matching a description of their content, in the request's words.
- single_column: tables with a column whose values overlap one column of the query table (joinable tables).
- multi_column: tables whose rows match several columns of the query table together.
  single_column and multi_column search the query table's values, or rows the request gives ("tables that also
  have Ken Sánchez or Terri Duffy"): then give those rows, each with a value per column, and name the query-table
  columns they belong to.
- correlation: tables with a column that correlates with a numeric target column of the query table, joined on
  a key column of the query table.
"""

FILLER_INSTRUCTIONS = """You fill one part of a table-search plan with one of Blend's seekers and its slots.
Columns come only from the query table. Give the seeker and each slot a status: certain if the request names or
clearly means that value, not_stated if the request does not say, ambiguous only as below; when it is not certain,
list the other likely values (for the seeker, the other seekers), most likely first.
""" + AMBIGUOUS + """
""" + COLUMN_ALTERNATIVES + """
""" + SEEKERS + """- none: none of these seekers fits the part."""

TEMPLATE_FILLER_INSTRUCTIONS = """You fill the slots of one of Blend's tasks, which a request for tables was recognised
as. Columns come only from the query table. Give each slot a status: certain if the request names or clearly means
that value, not_stated if the request does not say, ambiguous only as below; when it is not certain, list the other
likely values, most likely first.
""" + AMBIGUOUS + """
""" + COLUMN_ALTERNATIVES + """
Answer DoesNotFit when the request's searches cannot be put into the task's slots: for example negative examples with
nothing to remove, or two keys for one column to fill."""

EDIT_INSTRUCTIONS = """You change a table-search plan the way the user's newest message asks.
The view lists the plan's fields by id: k (how many tables are shown), each combiner (intersection, union, counter,
or difference, whose first input is kept and second removed), each part (one Blend search) and each of its slots.
Return a change for each id under "Change only" that the message changes, and for no other id:
- k: the new number of tables.
- a combiner: its new kind; a difference also takes exclusion or comparison.
- a part: its new seeker with all of that seeker's slots.
- a slot: its new value.
Columns come only from the query table. Give each value a status: certain if the message names or clearly means it,
ambiguous if another value fits too, not_stated if the message does not say; when it is not certain, list the other
likely values, most likely first. The seekers:
""" + SEEKERS


SEMANTIC_PROFILE = """The view's semantic profile is inferred by a model from the data profile, per column: temporal and
spatial granularity, entity type, domain and usage. It describes the query table's columns and adds no column to the
set you may choose from."""

Role = Literal['alone', 'intersection', 'union', 'counter', 'kept', 'exclusion', 'comparison']

PartId = Annotated[str, Field(pattern=r'^p[1-9][0-9]*$')]


class DecomposedPart(BaseModel):
    id: PartId
    words: ModelText = Field(description='The words of the request this part covers, verbatim.')


class DecomposedCombiner(BaseModel):
    combiner: Literal['intersection', 'union', 'counter']
    parts: List[DecomposedPart] = Field(min_length=2)


class DecomposedDifference(BaseModel):
    combiner: Literal['difference']
    kept: DecomposedPart
    removed: DecomposedPart
    removal: Literal['exclusion', 'comparison']


Group = Union[DecomposedPart, DecomposedCombiner, DecomposedDifference]


class RootCombiner(DecomposedCombiner):
    parts: List[Group] = Field(min_length=2)


class RootDifference(DecomposedDifference):
    kept: Group
    removed: Group


Tree = Union[DecomposedPart, RootCombiner, RootDifference]


class TemplateChoice(BaseModel):
    """One of Blend's tasks from the catalogue, whose parts the Template Filler fills."""
    template: TemplateId


PartTemplate = Literal['imputation', 'union_search', 'hybrid_nl', 'feature_for_ml', 'negative_examples']


class TemplatePart(BaseModel):
    """One of Blend's tasks as a part of a tree, with the words it covers; the Template Filler fills it."""
    template: PartTemplate
    words: ModelText = Field(description='The words of the request this task covers, verbatim.')


TaskGroup = Union[TemplatePart, DecomposedPart, DecomposedCombiner, DecomposedDifference]


class TaskCombiner(RootCombiner):
    parts: List[TaskGroup] = Field(min_length=2)


class TaskDifference(RootDifference):
    kept: TaskGroup
    removed: TaskGroup


Structure = Union[TemplateChoice, DecomposedPart, TaskCombiner, TaskDifference]


class Unplaced(BaseModel):
    words: ModelText = Field(description='The words of the request, verbatim.')
    kind: Literal['constraint', 'no_search_meaning']


class Decomposition(BaseModel):
    rationale: ModelText
    tree: Structure
    readings: List[Structure] = Field(default=[], description='Other structures the request can be read as, most '
                                                          'likely first; empty if there is only one.')
    fit: Literal['fits', 'too_deep', 'no_combiner']
    sub_plans: List[Tree] = Field(default=[], description='When the request does not fit: the other parts of it '
                                                          'that fit, largest first.')
    unplaced: List[Unplaced] = Field(default=[], description='Word groups of the request that no part covers.')
    k: Optional[int] = Field(ge=1, description='How many tables the request asks for, only if it says.')
    k_status: Status

    @model_validator(mode='after')
    def _ids_unique(self) -> 'Decomposition':
        for tree in [self.tree, *self.readings, *self.sub_plans]:
            _unique_ids(tree)
        return self


def _unique_ids(tree: Structure) -> None:
    ids = [part.id for part in parts(tree)]
    if len(ids) != len(set(ids)):
        raise ValueError(f'two parts of one tree share an id: {ids}')
    if not isinstance(tree, (DecomposedPart, TemplateChoice)) \
            and sum(isinstance(child, TemplatePart) for child in children(tree)) > 1:
        raise ValueError('a tree holds at most one task part')


class Reflection(BaseModel):
    reflection: ModelText = Field(description='In a sentence or two: which fields you now read differently from the '
                                              'plans the user rejected.')


class Revision(Decomposition, Reflection):
    """The Decomposer's re-plan of the request after a rejection with text."""


REVISION = NativeOutput(Revision)


class NoDifferentPlan(BaseModel):
    """The answer that the request, with the user's change, has no plan besides the plans the user rejected."""
    rationale: ModelText


REVISION_REPAIR = NativeOutput([Revision, NoDifferentPlan])


class Reading(BaseModel):
    """One further reading of the request, other than the plans the user rejected."""
    rationale: ModelText
    tree: Structure
    fit: Literal['fits', 'too_deep', 'no_combiner']

    @model_validator(mode='after')
    def _ids_unique(self) -> 'Reading':
        _unique_ids(self.tree)
        return self


class NoOtherReading(BaseModel):
    """The answer that the request has no reading besides the plans the user rejected."""
    rationale: ModelText


FURTHER = NativeOutput([Reading, NoOtherReading])


def decomposition(tree: Structure) -> Decomposition:
    return Decomposition(rationale='', tree=tree, fit='fits', k=None, k_status='not_stated')


class NoFill(BaseModel):
    """The Filler's answer that no Blend seeker fits the part."""
    seeker: Literal['none']


decomposer = Agent(name='decomposer', instructions=DECOMPOSER_INSTRUCTIONS,
                   output_type=NativeOutput(Decomposition), retries=0)
filler = Agent(name='filler', instructions=FILLER_INSTRUCTIONS, retries=0)
template_filler = Agent(name='template_filler', instructions=TEMPLATE_FILLER_INSTRUCTIONS, retries=0)


def _distinct(columns: Tuple[str, ...]) -> Tuple[str, ...]:
    if len(set(columns)) < len(columns):
        raise ValueError(f'a column is named twice: {columns}')
    return columns


def _fills(table: QueryTable) -> Dict[str, type]:
    column = Literal[tuple(table.columns)]
    fills = (KeywordFill, NaturalLanguageFill,
             create_model('SingleColumnFill', __base__=SingleColumnFill, column=(ColumnSlot[column], ...)),
             create_model('MultiColumnFill', __base__=MultiColumnFill,
                          columns=(ColumnSlot[Annotated[Tuple[column, ...], Field(min_length=2),
                                                        AfterValidator(_distinct)]], ...)))
    if table.numeric_columns:
        fills += (create_model('CorrelationFill', __base__=CorrelationFill, join=(ColumnSlot[column], ...),
                               target=(ColumnSlot[Literal[tuple(table.numeric_columns)]], ...)),)
    return {fill.model_fields['seeker'].annotation.__args__[0]: fill for fill in fills}


def column_slot(table: QueryTable, slot: str) -> type:
    return next(fill for fill in _fills(table).values() if slot in fill.model_fields).model_fields[slot].annotation


def one_value(table: QueryTable, slot: str) -> bool:
    return len(table.numeric_columns if slot == 'target' else table.columns) == (2 if slot == 'columns' else 1)


def filling_type(table: QueryTable, locked: Sequence[str] = (), seeker: Optional[Seeker] = None,
                 mapped: bool = False, way_out: bool = False) -> Optional[NativeOutput]:
    fills = tuple(fill for name, fill in _fills(table).items() if set(locked) <= set(fill.model_fields)
                  and seeker in (None, name))
    if not fills:
        return None
    seekers = Literal[tuple(fill.model_fields['seeker'].annotation.__args__[0] for fill in fills)]
    part = {'id': (PartId, Field(description='The id of the part of the plan shown, or of a part the user dropped, '
                                             'that this part means, or a number none of them has if it is a new '
                                             'part.'))} if mapped else {}
    filling = create_model('Filling', rationale=(ModelText, ...), **part,
                           fill=(Union[fills + (NoFill,)], Field(discriminator='seeker')),
                           seeker_status=(Status, ...),
                           seeker_alternatives=(List[seekers], Field(default=[], description=(
                               'Other seekers that fit the part, most likely first; empty if the seeker is certain.'))))
    return NativeOutput([filling, NoDifferentPlan] if way_out else filling)


SINGLE_CALL_INSTRUCTIONS = """You turn a request for tables from a data lake into a whole Blend plan: the tree of the
parts Blend searches for, and for each part of the tree its fill, one of Blend's seekers with its slots.
""" + TREE + """Give each part an id, p1, p2, ... in the order of the tree, and one fill per part, named by its id.
Columns come only from the query table. The seekers:
""" + SEEKERS + """Give k only if the request states how many tables it wants."""

single_call = Agent(name='single_call', instructions=SINGLE_CALL_INSTRUCTIONS, retries=0)


class Value(Frozen):
    """A slot's value as the single-call baseline gives it: no status and no alternatives."""
    value: Any


class SingleCallPlan(BaseModel):
    """The single-call baseline's whole plan: the Decomposer's tree, each part's fill, and k only if stated."""
    rationale: ModelText
    tree: Tree
    fills: List[Any]
    k: Optional[int] = Field(ge=1, description='How many tables the request asks for, only if it says.')

    @model_validator(mode='after')
    def _one_fill_per_part(self) -> 'SingleCallPlan':
        _unique_ids(self.tree)
        filled, ids = [fill.part for fill in self.fills], [part.id for part in parts(self.tree)]
        if sorted(filled) != sorted(ids):
            raise ValueError(f'each part of the tree needs exactly one fill: the tree has {ids}, the fills {filled}')
        return self


def single_call_type(table: QueryTable) -> NativeOutput:
    values = {}

    def value(name: str, slot: type) -> type:
        kind = slot.__pydantic_generic_metadata__['args'][0]
        return values.setdefault(repr(kind), create_model(f'{name.capitalize()}Value', __base__=Value,
                                                          value=(kind, ...)))

    fills = tuple(create_model(fill.__name__, __base__=fill, **{
        name: (value(name, field.annotation), ...) for name, field in fill.model_fields.items()
        if isinstance(field.annotation, type) and issubclass(field.annotation, Slot)}) for fill in _fills(table).values())
    part_fill = create_model('PartFill', part=(PartId, ...), fill=(Union[fills], Field(discriminator='seeker')))
    return NativeOutput(create_model('SingleCallPlan', __base__=SingleCallPlan,
                                     fills=(List[part_fill], Field(min_length=1))))


def single_call_key(plan: SingleCallPlan) -> Hashable:
    fills = {part.id: _fill_key(part.fill, wording=False) for part in planned_parts(plan)}

    def node(tree: Union[Tree, Group]) -> Hashable:
        if isinstance(tree, DecomposedPart):
            return fills[tree.id]
        if isinstance(tree, DecomposedDifference):
            return 'difference', tree.removal, node(tree.kept), node(tree.removed)
        return tree.combiner, tuple(sorted((node(child) for child in tree.parts), key=repr))

    return node(plan.tree), plan.k


def planned_parts(plan: SingleCallPlan) -> List[Part]:
    fills = {fill.part: fill.fill for fill in plan.fills}
    planned = []
    for part in parts(plan.tree):
        fill = fills[part.id].model_dump()
        fill.update({name: {'value': value['value'], 'status': 'certain'} for name, value in fill.items()
                     if isinstance(getattr(fills[part.id], name), Value)})
        planned.append(Part(id=part.id, words=part.words, fill=fill))
    return planned


class KChange(BaseModel):
    target: Literal['k']
    value: Slot[PositiveInt]


class CombinerChange(BaseModel):
    target: str
    kind: Literal['intersection', 'union', 'counter', 'difference']
    removal: Optional[Literal['exclusion', 'comparison']] = Field(None, description='Only for a difference.')

    @model_validator(mode='after')
    def _removal_of_a_difference(self) -> 'CombinerChange':
        if (self.kind == 'difference') != (self.removal is not None):
            raise ValueError('a difference, and only a difference, takes exclusion or comparison')
        return self


class PartChange(BaseModel):
    target: str
    fill: Any
    seeker_status: Status


class SlotChange(BaseModel):
    target: str
    value: Any


class Edit(BaseModel):
    rationale: ModelText
    changes: List[BaseModel]


editor = Agent(name='edit', instructions=EDIT_INSTRUCTIONS, retries=0)


def edit_type(plan: SearchPlan, targets: Sequence[str], table: QueryTable) -> NativeOutput:
    fills = _fills(table)
    nodes = {node.id: node for node in walk(plan.root)}
    changes = []
    for target in targets:
        node_id, _, name = target.partition('.')
        model = f'Change_{node_id}_{name}'
        if target == 'k':
            changes.append(KChange)
        elif node_id == 't':
            changes.append(create_model(model, __base__=SlotChange, target=(Literal[target], ...),
                                        value=(column_slot(table, 'column'), ...)))
        elif isinstance(nodes[node_id], Combiner):
            kinds = ('intersection', 'union', 'counter') + (('difference',) if len(nodes[node_id].inputs) == 2 else ())
            changes.append(create_model(model, __base__=CombinerChange, target=(Literal[target], ...),
                                        kind=(Literal[kinds], ...)))
        elif name:
            slot = fills[nodes[node_id].fill.seeker].model_fields[name].annotation
            changes.append(create_model(model, __base__=SlotChange, target=(Literal[target], ...), value=(slot, ...)))
        else:
            changes.append(create_model(model, __base__=PartChange, target=(Literal[target], ...),
                                        fill=(Union[tuple(fills.values())], Field(discriminator='seeker'))))
    return NativeOutput(create_model('Edit', __base__=Edit, changes=(List[Union[tuple(changes)]], Field(min_length=1))))


def edit_key(edit: Edit) -> Hashable:
    changes = []
    for change in edit.changes:
        dumped, name = change.model_dump(exclude={'fill'}), change.target.partition('.')[2]
        if name == 'description':
            del dumped['value']
        elif name:
            dumped['value']['value'] = _normalized(name, dumped['value']['value'])
        fill = _fill_key(change.fill, wording=False) if isinstance(change, PartChange) else None
        changes.append((canonical(dumped, ('status', 'alternatives', 'seeker_status')), fill))
    return tuple(sorted(changes, key=repr))


def children(tree: Tree) -> List[Group]:
    return [tree.kept, tree.removed] if isinstance(tree, DecomposedDifference) else list(tree.parts)


def parts(tree: Union[Structure, TaskGroup]) -> List[DecomposedPart]:
    if isinstance(tree, DecomposedPart):
        return [tree]
    if isinstance(tree, (TemplateChoice, TemplatePart)):
        return []
    return [part for child in children(tree) for part in parts(child)]


def task(tree: Union[Structure, TaskGroup]) -> Union[None, TemplateChoice, TemplatePart]:
    if isinstance(tree, (TemplateChoice, TemplatePart)):
        return tree
    if isinstance(tree, DecomposedPart):
        return None
    return next((found for child in children(tree) for found in [task(child)] if found), None)


def outline(node: Union[Structure, TaskGroup, Part, Combiner]) -> Line:
    if isinstance(node, (DecomposedPart, Part)):
        return Line('{!r}', node.words)
    if isinstance(node, TemplatePart):
        return Line("{!r} (Blend's {} task)", node.words, node.template.replace('_', ' '))
    if isinstance(node, TemplateChoice):
        return Line("{} (Blend's {} task)", GOALS[node.template], node.template.replace('_', ' '))
    kind = node.kind if isinstance(node, Combiner) else node.combiner
    inputs = [outline(child) if isinstance(child, (DecomposedPart, TemplatePart, Part))
              else Line('({})', outline(child))
              for child in (node.inputs if isinstance(node, Combiner) else children(node))]
    if kind == 'difference':
        return Line('{} but not the best matches for {}' if node.removal == 'comparison' else '{} but not {}', *inputs)
    if kind == 'counter':
        return Line('most of {}', joined(', ', inputs))
    return joined(' and ' if kind == 'intersection' else ' or ', inputs)


def roles(node: Union[Tree, Part, Combiner], role: Role = 'alone') -> Dict[str, Role]:
    if isinstance(node, (DecomposedPart, Part)):
        return {node.id: role}
    if isinstance(node, (TemplateChoice, TemplatePart)):
        return {}
    kind = node.kind if isinstance(node, Combiner) else node.combiner
    inputs = node.inputs if isinstance(node, Combiner) else children(node)
    kinds = ('kept', node.removal) if kind == 'difference' else (kind,) * len(inputs)
    return {part_id: part_role for child, child_role in zip(inputs, kinds)
            for part_id, part_role in roles(child, child_role).items()}


def structure(node: Union[Structure, Part, Combiner], ids: Optional[Sequence[str]] = None) -> Optional[str]:
    if isinstance(node, (DecomposedPart, Part)):
        return node.id if ids is None or node.id in ids else None
    if isinstance(node, TemplateChoice):
        return node.template
    if isinstance(node, TemplatePart):
        return node.template if ids is None else None
    kind = node.kind if isinstance(node, Combiner) else node.combiner
    inputs = [structure(child, ids) for child in (node.inputs if isinstance(node, Combiner) else children(node))]
    if kind == 'difference' and None not in inputs:
        return f'difference(kept {inputs[0]}, removed {inputs[1]}, {node.removal})'
    inputs = [shape for shape in inputs if shape is not None]
    if len(inputs) < 2:
        return next(iter(inputs), None)
    return f'{kind}({", ".join(sorted(inputs))})'


def siblings(tree: Union[Structure, TaskGroup]) -> Dict[str, List[str]]:
    if isinstance(tree, (DecomposedPart, TemplateChoice, TemplatePart)):
        return {}
    ids = [child.id for child in children(tree) if isinstance(child, DecomposedPart)]
    found = {part_id: [other for other in ids if other != part_id] for part_id in ids}
    for child in children(tree):
        found |= siblings(child)
    return found


def structure_parts(shape: str) -> List[str]:
    return re.findall(r'\bp[1-9][0-9]*\b', shape)


def decomposition_key(decomposition: Decomposition) -> Hashable:
    return canonical(decomposition, ('rationale', 'reflection', 'id', 'words', 'unplaced', 'readings', 'sub_plans',
                                     'k_status'))


def structure_key(tree: Union[None, Structure, Group], sides: Optional[Dict[str, Hashable]] = None) -> Hashable:
    if tree is None:
        return None
    if isinstance(tree, DecomposedPart):
        return 'part'
    if isinstance(tree, (TemplateChoice, TemplatePart)):
        return 'template', tree.template
    if _single_part_difference(tree):
        return 'difference', tree.removal, *(('part', (sides or {}).get(side.id)) for side in (tree.kept, tree.removed))
    if isinstance(tree, DecomposedDifference):
        return 'difference', tree.removal, structure_key(tree.kept, sides), structure_key(tree.removed, sides)
    return tree.combiner, tuple(sorted((structure_key(child, sides) for child in tree.parts), key=repr))


def single_part_sides(tree: Union[Structure, Group]) -> List[str]:
    if _single_part_difference(tree):
        return [tree.kept.id, tree.removed.id]
    return [] if isinstance(tree, (DecomposedPart, TemplateChoice, TemplatePart)) \
        else [part for child in children(tree) for part in single_part_sides(child)]


def _single_part_difference(tree: Union[Tree, Group]) -> bool:
    return isinstance(tree, DecomposedDifference) and all(isinstance(side, DecomposedPart)
                                                          for side in (tree.kept, tree.removed))


def plan_key(plan: SearchPlan) -> Hashable:
    return plan.k.value, _node_key(plan.root)


def _node_key(node: Union[Part, Combiner], with_k: bool = True) -> Hashable:
    k = node.k if with_k else None
    if isinstance(node, Part):
        return k, _fill_key(node.fill)
    inputs = tuple(_node_key(child, with_k) for child in node.inputs)
    return node.kind, node.removal, k, inputs if node.kind == 'difference' else tuple(sorted(inputs, key=repr))


def same_search(node: Union[Part, Combiner], other: Union[Part, Combiner]) -> bool:
    return _node_key(node, with_k=False) == _node_key(other, with_k=False)


def repeated(root: Union[Part, Combiner]) -> Optional[Tuple[Union[Part, Combiner], Union[Part, Combiner]]]:
    for combiner in (node for node in walk(root) if isinstance(node, Combiner)):
        inputs = [[child] if combiner.kind == 'union' else [node for node in walk(child) if isinstance(node, Part)]
                  for child in combiner.inputs]
        for later, nodes in enumerate(inputs):
            for node in nodes:
                earlier = next((other for others in inputs[:later] for other in others if same_search(other, node)),
                               None)
                if earlier:
                    return earlier, node
    return None


def _normalized(name: str, value: Any) -> Any:
    if name == 'description':
        return value.strip()
    if name in COLUMN_SLOTS:
        return tuple(sorted(set(value))) if isinstance(value, tuple) else value
    if isinstance(value, tuple):
        return tuple(sorted({DBHandler.normalize_token(item) for item in value}))
    return DBHandler.normalize_token(value) if isinstance(value, str) else value


def _fill_key(fill: Any, wording: bool = True) -> Hashable:
    return fill.seeker, tuple(sorted((name, _normalized(name, slot.value)) for name, slot in fill
                                     if isinstance(slot, Slot) and (wording or name != 'description'))), _rows(fill), \
        getattr(fill, 'rows_where', None)


def _rows(fill: Any) -> Tuple:
    return tuple(sorted({tuple(sorted((column, DBHandler.normalize_token(value)) for column, value in row.items()))
                         for row in typed_rows(fill)}))


def filling_key(filling: BaseModel) -> Hashable:
    if isinstance(filling, NoDifferentPlan):
        return ()
    return _fill_key(filling.fill, wording=False)


class Unfilled(BaseModel):
    """A seeker only among a part's ranked alternatives, filled with the seeker fixed when a rejection reaches it."""
    part: str
    seeker: Seeker


class StoredReading(BaseModel):
    """A reading that did not run and how many samples chose it; a structural one stays the Decomposer's tree."""
    samples: int
    root: Union[Part, Combiner, Structure]
    unfilled: Optional[Unfilled] = None
    alternative: bool = False
    template: Optional[TemplateId] = None
    template_parts: Tuple[str, ...] = ()


def tally(produced: List[List[Any]], key: Callable[[Any], Hashable], proposal: Any) -> List[Tuple[int, Any, bool]]:
    counted = {}
    for readings in produced:
        for position, (reading_key, reading) in enumerate({key(reading): reading for reading in readings}.items()):
            if reading_key != key(proposal):
                count, first = counted.get(reading_key, (0, reading))
                counted[reading_key] = (count + (position == 0), first)
    return [(count, reading, count == 0) for count, reading in counted.values()]


def seeker_readings(part: Part, fillings: List[BaseModel]
                    ) -> Tuple[Part, List[Tuple[int, Union[Part, Unfilled], bool]]]:
    chosen = {}
    for filling in fillings:
        chosen.setdefault(filling.fill.seeker, filling.fill)
    others = tally([[filling.fill.seeker, *getattr(filling, 'seeker_alternatives', ())] for filling in fillings],
                   lambda seeker: seeker, part.fill.seeker)
    if others or any(filling.seeker_status != 'certain' for filling in fillings):
        part = part.model_copy(update={'seeker_status': 'ambiguous' if part.seeker_status == 'certain'
                                       else part.seeker_status})
    return part, [(count, Part(id=part.id, words=part.words, fill=chosen[seeker].model_dump(),
                               seeker_status='ambiguous') if seeker in chosen
                   else Unfilled(part=part.id, seeker=seeker), alternative)
                  for count, seeker, alternative in others if seeker != 'none']


def column_readings(part: Part, fills: List[BaseModel], table: QueryTable
                    ) -> Tuple[Part, List[Tuple[int, Part, bool]]]:
    same_seeker = [fill for fill in fills if fill.seeker == part.fill.seeker]
    other_values, disputed = {}, {}
    for name, slot in part.fill:
        if isinstance(slot, Slot) and name in COLUMN_SLOTS and one_value(table, name):
            disputed[name] = slot.model_copy(update={'status': 'certain'})
        elif isinstance(slot, Slot) and name not in WORDING:
            samples = [getattr(fill, name) for fill in same_seeker]
            other_values[name] = [] if getattr(part.fill, 'rows', ()) else tally(
                [[sample.value, *sample.alternatives] for sample in samples], canonical, slot.value)
            if slot.status == 'certain' and (other_values[name]
                                             or any(sample.status != 'certain' for sample in samples)):
                disputed[name] = slot.model_copy(update={'status': 'ambiguous'})
    part = part.model_copy(update={'fill': part.fill.model_copy(update=disputed)})
    readings = [_replaced(part, name, value) for name, values in other_values.items() for _, value, _ in values]
    for fill in same_seeker:
        names = [name for name in other_values
                 if canonical(getattr(fill, name).value) != canonical(getattr(part.fill, name).value)]
        whole = part
        for name in names:
            whole = _replaced(whole, name, getattr(fill, name).value)
        if len(names) > 1:
            readings.append(whole)
    chosen, distinct = Counter(_fill_key(fill, wording=False) for fill in same_seeker), {}
    for reading in readings:
        distinct.setdefault(_fill_key(reading.fill, wording=False), reading)
    return part, [(chosen[key], reading, not chosen[key]) for key, reading in distinct.items()]


def _replaced(part: Part, name: str, value: Any) -> Part:
    slot = getattr(part.fill, name).model_copy(update={'value': value, 'alternatives': ()})
    return part.model_copy(update={'fill': part.fill.model_copy(update={name: slot})})
