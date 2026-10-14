from dataclasses import dataclass, field

from src.Agent.Planner import outline
from src.Agent.Plans import SHARED, Combiner, Part, Slot, ranked_by, repeating_seekers, typed, unranked, walk
from src.Agent.Text import Line, joined

from typing import Any, Collection, Dict, FrozenSet, List, Optional, Tuple, Union
from src.Agent.Plans import BlendNode, SearchPlan
from src.Agent.Text import ModelText

SHAPES = {'intersection': 'tables found by all searches', 'union': 'tables found by any search',
          'counter': 'tables found by most searches', 'difference': 'tables of one search without another'}


@dataclass
class PlanCard:
    version: int
    tables: List[Tuple[int, str]]
    words: str
    nodes: List[BlendNode]
    assumed: List[str] = field(default_factory=list)
    preselected: Dict[str, Any] = field(default_factory=dict)
    not_used: List[ModelText] = field(default_factory=list)
    notices: List[Union[str, Line]] = field(default_factory=list)
    changed: List[Union[str, Line]] = field(default_factory=list)
    not_it: bool = True
    ranked: bool = True
    discrepancies: List[Union[str, Line]] = field(default_factory=list)
    task: Optional[str] = None


def words(plan: SearchPlan, ranking: Dict[str, str], found: Optional[int] = None) -> str:
    return _read_as(plan, _root(plan, ranking, found))


def not_this(plan: SearchPlan) -> str:
    return _read_as(plan, f'Tables that {_clause(plan.root, None, None)}. At most {plan.k.value} tables are shown.')


def search(node: Union[Part, Combiner]) -> str:
    return _clause(node, None, None)


def _read_as(plan: SearchPlan, sentence: str) -> str:
    if plan.structure == 'certain':
        return sentence
    shape = SHAPES[plan.root.kind] if isinstance(plan.root, Combiner) else 'one search'
    return f'Read as {shape} (assumed): {sentence}'


def _root(plan: SearchPlan, ranking: Dict[str, str], found: Optional[int]) -> str:
    root, k = plan.root, plan.k.value
    sentences = [f'Tables that {_clause(root, root.k or k, ranking)}.']
    if found is not None and found < k and (root.k or unranked(root, ranking)):
        sentences += [f'All {found} tables found are shown.'] if found else []
    elif unranked(root, ranking):
        sentences.append(f'Shown: {k} tables, unranked.')
    elif root.k:
        sentences.append(f'The first {k} are shown.')
    for combiner in walk(root):
        if not isinstance(combiner, Combiner) or combiner.kind not in ('intersection', 'difference'):
            continue
        first = ranked_by(combiner, ranking)
        passes = f'{combiner.id} passes on all {combiner.k or k} tables of {first.id}, unranked'
        if combiner.kind == 'difference':
            if unranked(first, ranking):
                sentences.append(f'{passes}.')
            continue
        filters = [child.id for child in combiner.inputs if child is not first]
        result = 'the result' if combiner is root else f'the result of {combiner.id}'
        sentences.append(f'{passes if unranked(first, ranking) else f"{first.id} ranks {result}"}; '
                         f'{" and ".join(filters)} only filter{"s" if len(filters) == 1 else ""}.')
    return ' '.join(sentences)


def _clause(node: Union[Part, Combiner], k: Optional[int], ranking: Optional[Dict[str, str]]) -> str:
    if isinstance(node, Part):
        return _part(node, k)

    def inner(child: Union[Part, Combiner], child_k: Optional[int]) -> str:
        clause = _clause(child, child_k if ranking is not None else None, ranking)
        return clause if isinstance(child, Part) else f'({clause})'

    if node.kind == 'difference':
        kept, removed = node.inputs
        if node.removal == 'exclusion':
            return (f'{inner(kept, k)}, except the tables that {inner(removed, removed.k)} '
                    f'(an exclusion: every such table is left out)')
        return (f'{inner(kept, k)}, minus the tables that {inner(removed, removed.k)} '
                f'(a comparison: the best matches of that search are taken away)')
    if node.kind == 'intersection' and ranking is None:
        return ' and that '.join(inner(child, None) for child in node.inputs)
    if node.kind == 'intersection':
        first = ranked_by(node, ranking)
        also = ' and that '.join(inner(child, child.k) for child in node.inputs if child is not first)
        return f'{inner(first, k)}, taken only from tables that also {also}'
    either = ', or that '.join(inner(child, child.k) for child in node.inputs)
    if node.kind == 'union':
        return f'{either}, up to {k} tables together, unranked' if k else either
    if ranking is None:
        return f'{either}, ranked by how many of these searches find them'
    seekers = set().union(*(repeating_seekers(child, ranking) for child in node.inputs))
    matching = ' and '.join(name for seeker, name in (('single_column', 'columns'), ('correlation', 'column pairs'))
                            if seeker in seekers)
    counts = f' (counts matching {matching}: a table counts once for each of its {matching} that matches)' \
        if matching else ''
    return f'{either}, ranked by how many of these searches find them, the {k} found most{counts}'


def _part(part: Part, k: Optional[int]) -> str:
    sentence = _search(part, k)
    if part.seeker_status == 'certain':
        return sentence
    return f'{sentence} (kind of search assumed: {part.fill.seeker.replace("_", " ")})'


def _search(part: Part, k: Optional[int]) -> str:
    fill = part.fill
    best = f', the {k} best' if k else ''
    matches = f', among the {k} best column matches, each table once' if k else ''
    if fill.seeker == 'keyword':
        most = f', the {k} containing the most of them' if k else ''
        return f'contain the words {", ".join(fill.words.value)}{most}'
    if fill.seeker == 'natural_language':
        return f'match the description {fill.description.value!r}{best}'
    if fill.seeker == 'multi_column':
        if fill.rows:
            return (f'have rows matching the rows you typed for your columns {_value(fill.columns, " and ")}, '
                    f'{typed(fill)}{best}')
        selected = f', in {fill.rows_where.describe()}' if fill.rows_where else ''
        return f'have rows matching your columns {_value(fill.columns, " and ")} together{selected}{best}'
    if fill.seeker == 'correlation':
        return (f'have a column that correlates with your column {_value(fill.target)}, joined on your column '
                f'{_value(fill.join)}{matches}')
    if fill.rows:
        return (f'have a column that shares values with the rows you typed for your column {_value(fill.column)}, '
                f'{typed(fill)}{matches}')
    if fill.rows_where:
        return (f'have a column that shares values with your {_value(fill.column)} values of '
                f'{fill.rows_where.describe()}{matches}')
    return f'have a column that shares values with your column {_value(fill.column)}{matches}'


def _value(slot: Slot, joiner: str = ', ') -> str:
    value = _plain(slot, joiner)
    return value if slot.status == 'certain' else f'{value} (assumed)'


def changes(before: SearchPlan, after: SearchPlan, said: Collection[str] = ()) -> List[Union[str, Line]]:
    old = {part.id: part for part in before.parts()}
    matched = _matched(before.root, after.root)
    kept = {part.id for part in after.parts()} | {was.id for was in matched.values()}
    lines = [f'k changed from {before.k.value} to {after.k.value}.'] if before.k.value != after.k.value else []
    shared = SHARED.get(after.template, {}) if after.template == before.template else {}
    covered = after.covered() if shared else set()
    for node in walk(after.root):
        was = matched.get(node.id) if isinstance(node, Combiner) else old.get(node.id)
        if isinstance(node, Combiner) and was is not None and _kind(was) != _kind(node):
            lines.append(f'{node.id} changed from {_kind(was)} to {_kind(node)}.')
        elif was is None:
            lines.append(Line('{} was added.', _label(node)))
        elif isinstance(node, Combiner):
            if node.kind == 'difference' and _part_ids(node.inputs[0]) != _part_ids(was.inputs[0]):
                lines.append(f'{node.id} now keeps {node.inputs[0].id} and removes {node.inputs[1].id}.')
            continue
        elif was.fill.seeker != node.fill.seeker:
            lines.append(f'{node.id} changed from {_seeker(was)} to {_seeker(node)}.')
        else:
            lines += [f'{node.id}.{name} changed from {_plain(getattr(was.fill, name))} to {_plain(slot)}.'
                      for name, slot in node.fill
                      if isinstance(slot, Slot) and slot.value != getattr(was.fill, name).value
                      and f'{node.id}.{name}' not in covered]
            if getattr(node.fill, 'rows', ()) != getattr(was.fill, 'rows', ()):
                lines.append(f"{node.id}'s typed rows changed from {typed(was.fill) or 'none'} to "
                             f"{typed(node.fill) or 'none'}.")
            if getattr(node.fill, 'rows_where', None) != getattr(was.fill, 'rows_where', None) and not shared:
                lines.append(f"{node.id}'s rows changed from {_rows(was.fill)} to {_rows(node.fill)}.")
    lines += [f't.{name} changed from {_plain(before.shared(name))} to {_plain(after.shared(name))}.'
              for name in shared if before.shared(name).value != after.shared(name).value]
    task = [f"The plan is no longer the {before.task.replace('_', ' ')} task."] \
        if before.task and not after.task else []
    return lines + [Line('{} was removed.', _label(node)) for node in walk(before.root)
                    if node.id not in kept and node.id not in said] + task


def _rows(fill: Any) -> str:
    return fill.rows_where.describe() if getattr(fill, 'rows_where', None) else 'all rows'


def _matched(before: Union[Part, Combiner], after: Union[Part, Combiner]) -> Dict[str, Combiner]:
    by_id = {node.id: node for node in walk(before) if isinstance(node, Combiner)}
    new = [node for node in walk(after) if isinstance(node, Combiner)]
    by_parts = {_same(node): node for node in by_id.values()}
    matched = {node.id: by_parts[_same(node)] for node in new if _same(node) in by_parts}
    taken = {was.id for was in matched.values()}
    return matched | {node.id: by_id[node.id] for node in new
                      if node.id not in matched and node.id in by_id and node.id not in taken}


def _same(combiner: Combiner) -> Tuple[str, FrozenSet[str]]:
    return _kind(combiner), _part_ids(combiner)


def _part_ids(node: Union[Part, Combiner]) -> FrozenSet[str]:
    return frozenset(inner.id for inner in walk(node) if isinstance(inner, Part))


def _seeker(part: Part) -> str:
    slots = [f'{name} {_plain(slot)}' for name, slot in part.fill if isinstance(slot, Slot)]
    slots += [f'typed rows {typed(part.fill)}'] if getattr(part.fill, 'rows', ()) else []
    return f'a {part.fill.seeker.replace("_", " ")} search ({", ".join(slots)})'


def _label(node: Union[Part, Combiner]) -> Union[str, Line]:
    if isinstance(node, Part):
        return Line('{} ({!r})', node.id, node.words)
    return f'{node.id} ({_kind(node)} of {", ".join(child.id for child in node.inputs)})'


def _kind(combiner: Combiner) -> str:
    return f'{combiner.kind} ({combiner.removal})' if combiner.removal else combiner.kind


def _plain(slot: Slot, joiner: str = ', ') -> str:
    return joiner.join(slot.value) if isinstance(slot.value, tuple) else str(slot.value)


def discrepancy(plan: SearchPlan, found: Any) -> Union[str, Line]:
    if found.kind == 'missing':
        return Line('Possibly missing from the plan: {!r}.', found.description)
    if found.kind == 'number_not_applied':
        return f'Possibly not the number of tables the request asks for: k = {plan.k.value}.'
    if found.target == 'k':
        label = f'k = {plan.k.value}'
    else:
        node_id, _, slot = found.target.partition('.')
        node = next(node for node in walk(plan.root) if node.id == node_id)
        label = Line('{} of {}', slot, _label(node)) if slot else _label(node)
    return Line('Possibly not asked for in the request: {}.' if found.kind == 'extra'
                else 'Possibly reversed against the request: {}.', label)


def finding(plan: SearchPlan, found: Any) -> Union[str, Line]:
    node_id, _, slot = getattr(found, 'target', '').partition('.')
    part = next((node for node in walk(plan.root) if isinstance(node, Part) and node.id == node_id), None)
    if part is None:
        return discrepancy(plan, found)
    return Line('{} The plan has: {}.', discrepancy(plan, found), _plain(getattr(part.fill, slot)) if slot
                else _seeker(part))


def notices(plan: SearchPlan, missing: List[Tuple[str, str, str]], counts: Dict[str, int],
            empty: bool) -> List[Union[str, Line]]:
    parts = {part.id: part for part in plan.parts()}
    lines = []
    for part_id, slot, name in missing:
        if slot == 'words':
            lines.append(f'The word {name!r} does not occur in the lake.')
        elif slot == 'rows':
            lines.append(f'No table in the lake shares values with the rows typed for column {name}.')
        elif getattr(parts[part_id].fill, slot).status == 'certain':
            lines.append(f'No table in the lake shares values with column {name}.')
        else:
            lines.append(f'None of the likely columns shares values with the lake, so {part_id} runs with {name} '
                         f'(assumed).')
    for node in walk(plan.root):
        if isinstance(node, Combiner) and node.kind == 'difference' and counts.get(node.inputs[1].id) == 0:
            lines.append(Line('The excluded part matched no tables, so nothing was removed: {}.',
                              outline(node.inputs[1])))
    if empty:
        nothing = [_label(part) for part in parts.values() if counts.get(part.id) == 0]
        uncounted = [part.id for part in parts.values() if part.id not in counts]
        if nothing:
            lines.append(Line(f'{{}} {"finds" if len(nothing) == 1 else "find"} no tables on '
                              f'{"its" if len(nothing) == 1 else "their"} own.', joined(' and ', nothing)))
        elif isinstance(plan.root, Combiner) and plan.root.kind == 'difference' and counts.get(plan.root.inputs[0].id):
            lines.append('The excluded part removed every table the kept part found.')
        else:
            lines.append(f'Each {"counted " if uncounted else ""}part finds tables, but no table matches all of them.')
        if uncounted:
            lines.append(f'{" and ".join(uncounted)} {"was" if len(uncounted) == 1 else "were"} not counted: each '
                         f'natural-language run costs judge calls.')
    return list(dict.fromkeys(lines))
