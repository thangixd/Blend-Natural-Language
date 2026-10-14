import json

from src.Agent import Examples, Templates
from src.Agent.Plans import COLUMN_SLOTS, GOALS, SHARED, Combiner, Part, Slot, typed, walk
from src.Agent.Text import Opaque

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
from src.Agent.Plans import SearchPlan
from src.Agent.Planner import Role
from src.Agent.Profile import QueryTable
from src.Agent.Text import ModelText, UserText

NO_QUESTION = 'Your previous reply asked the user something. Only the system asks; say it without asking anything.'

HEADER = 'Examples of verified answers, each for a query table of its own:'

WAY_OUT = ('If every plan the request supports is one the user rejected, answer NoDifferentPlan; never change a field '
           'only to avoid a rejected plan.')

FINDINGS = ('A check of the plan shown against the request found these possible discrepancies. Fix what the request '
            'supports and keep what the check did not question; a requirement missing from the plan becomes a new '
            'part:')

SEMANTICS = "The semantic profile of the query table's columns, inferred by a model, is as follows:"

OTHER_PARTS = ("The plan's other parts, context only: each is filled on its own, and the plan combines them with this "
               "part. Fill only what this part's own words ask for:")


def _content(text: Union[str, Opaque]) -> str:
    return text.for_prompt() if isinstance(text, Opaque) else text


def router_view(message: UserText, turn: int, last_turns: List[Tuple[int, UserText]], plan: Optional[str],
                decisions: Sequence[Union[str, Opaque]], columns: str, declined: Sequence[Union[str, Opaque]] = (),
                current: Optional[SearchPlan] = None) -> str:
    lines = [f'New message (turn {turn}): {_content(message)!r}', '', 'Last turns:']
    lines += [f'  Turn {number}: {_content(text)!r}' for number, text in last_turns] or ['  (none)']
    lines += ['', f'Current plan: {plan}' if plan else 'Current plan: none yet.']
    lines += ['Its fields by id, which edit_plan names:', *fields(current)] if current else []
    lines += ['', 'Open questions:']
    lines += [_content(line) for line in decisions] or ['  (none)']
    if declined:
        lines += ['', 'Closed question, the user chose none of these:', *map(_content, declined)]
    lines += ['', f'Query table columns: {columns}']
    return '\n'.join(lines)


def fields(plan: SearchPlan) -> List[str]:
    lines, covered = [f'  k = {plan.k.value} (how many tables are shown)'], plan.covered()
    for node in walk(plan.root):
        if isinstance(node, Combiner):
            removal = f' ({node.removal})' if node.removal else ''
            lines.append(f'  {node.id} = {node.kind}{removal} of {", ".join(child.id for child in node.inputs)}')
        else:
            rows = typed(node.fill)
            lines.append(f'  {node.id} = {node.fill.seeker} search for {_content(node.words)!r}'
                         + (f' over the typed rows {rows}' if rows else ''))
            lines += [f'  {node.id}.{name} = {slot.value!r}' for name, slot in node.fill
                      if isinstance(slot, Slot) and f'{node.id}.{name}' not in covered]
    return lines + [f'  t.{name} = {plan.shared(name).value!r} (one value of {", ".join(parts)})'
                    for name in SHARED.get(plan.template, {})
                    for parts in [sorted({part_id for part_id, _, _ in plan.places(name)})]]


def request(messages: List[Tuple[int, UserText]]) -> List[str]:
    return (["Request (the user's messages, verbatim, newest last):"]
            + [f'Turn {turn}: {_content(text)}' for turn, text in messages])


def decomposer_view(messages: List[Tuple[int, UserText]], table: QueryTable, facts: Sequence[Union[str, Opaque]] = (),
                    examples: Sequence[str] = (), failed_check: Optional[str] = None,
                    rejected: Sequence[str] = (), way_out: bool = False,
                    findings: Sequence[Union[str, Opaque]] = ()) -> str:
    lines = (_examples(examples) + request(messages) + ([''] + list(map(_content, facts)) if facts else [])
             + (['', FINDINGS, *map(_content, findings)] if findings else []) + _rejected(rejected)
             + ['', _table(table)])
    return '\n'.join(lines + _repair(failed_check, way_out))


def _table(table: QueryTable) -> str:
    if not table.semantics:
        return table.describe()
    lines = [table.describe(), '', SEMANTICS]
    for column, semantics in table.semantics.items():
        said = [f'Represents {_quoted(semantics.entity_type)}.' if semantics.entity_type else '',
                f'Contains temporal data (resolution: {semantics.temporal}).'
                if semantics.temporal not in (None, 'none') else '',
                f'Contains spatial data (resolution: {semantics.spatial}).'
                if semantics.spatial not in (None, 'none') else '',
                f'Domain-specific type: {_quoted(semantics.domain)}.' if semantics.domain else '',
                f'Function/Usage context: {semantics.usage}.' if semantics.usage else '']
        lines += [f'**{column}**: {" ".join(filter(None, said))}'] if any(said) else []
    return '\n'.join(lines)


def _quoted(text: ModelText) -> str:
    return repr(_content(text).lower())


def _repair(failed_check: Optional[str], way_out: bool) -> List[str]:
    return ((['', f'Your previous answer failed a check: {failed_check}'] if failed_check else [])
            + ([WAY_OUT] if way_out else []))


def _examples(examples: Sequence[str]) -> List[str]:
    return [*examples, ''] if examples else []


def _rejected(rejected: Sequence[str]) -> List[str]:
    return ['', 'Plans the user rejected:', *rejected] if rejected else []


def edit_view(messages: List[Tuple[int, UserText]], plan: SearchPlan, targets: Sequence[str], table: QueryTable,
              failed_check: Optional[str], facts: Sequence[Union[str, Opaque]] = (),
              rejected: Sequence[str] = ()) -> str:
    lines = request(messages) + ['', f'Current plan (v{plan.version}):', *fields(plan), '',
                                 f'Change only: {", ".join(targets)}', *map(_content, facts), *_rejected(rejected), '',
                                 _table(table)]
    if failed_check:
        lines += ['', f'Your previous answer failed a check: {failed_check}']
    return '\n'.join(lines)


def filler_view(messages: List[Tuple[int, UserText]], part_id: Optional[str], words: ModelText, table: QueryTable,
                failed_check: Optional[str], facts: Sequence[Union[str, Opaque]] = (), examples: Sequence[str] = (),
                rejected: Sequence[str] = (), way_out: bool = False, others: Sequence[ModelText] = ()) -> str:
    asked = f'Fill part {part_id}: {_content(words)!r}' if part_id else f'Fill this part: {_content(words)!r}'
    lines = (_examples(examples) + request(messages) + ['', *_asked(asked, others), *map(_content, facts),
                                                        *_rejected(rejected), '', _table(table)])
    return '\n'.join(lines + _repair(failed_check, way_out))


def template_filler_view(messages: List[Tuple[int, UserText]], template: str, words: Optional[ModelText],
                         table: QueryTable, failed_check: Optional[str], facts: Sequence[Union[str, Opaque]] = (),
                         examples: Sequence[str] = (), rejected: Sequence[str] = (),
                         others: Sequence[ModelText] = ()) -> str:
    asked = [f"Fill Blend's {template} task: {GOALS[template]}."]
    asked += _asked(f'It covers these words of the request: {_content(words)!r}', others) if words else []
    asked += ['Its slots:'] + [f'- {name}: {Templates.SLOTS[name]}' for name in Templates.TEMPLATES[template].slots]
    lines = (_examples(examples) + request(messages) + ['', *asked, *map(_content, facts), *_rejected(rejected), '',
                                                        _table(table)])
    return '\n'.join(lines + _repair(failed_check, False))


def _asked(asked: str, others: Sequence[ModelText]) -> List[str]:
    return [asked] + ([OTHER_PARTS, *[f'  {_content(words)!r}' for words in others]] if others else [])


def verifier_view(messages: List[Tuple[int, UserText]], plan: SearchPlan, words: str,
                  chosen: Sequence[Union[str, Opaque]], left: Sequence[Union[str, Opaque]], rejected: Sequence[str],
                  previous: Optional[str], table: QueryTable) -> str:
    lines = request(messages) + ['', f'Plan (v{plan.version}): {words}', 'Its fields by id, which a discrepancy names:',
                                 *fields(plan)]
    lines += ['', f'Previous plan, before the newest message changed it: {previous}'] if previous else []
    lines += ['', *[f'The user chose: {_content(line)}' for line in chosen]] if chosen else []
    lines += ['', *[f'The user left this to the system, which confirms nothing: {_content(line)}' for line in left]] \
        if left else []
    lines += ['', *[f'The user rejected this plan: {words}' for words in rejected]] if rejected else []
    return '\n'.join(lines + ['', table.describe()])


def chat_view(say: Sequence[Union[str, Opaque]], snapshot: Sequence[Union[str, Opaque]],
              last_turns: List[Tuple[int, UserText]], replies: List[Tuple[int, Union[str, Opaque]]],
              older: Sequence[Union[str, Opaque]], regenerated: bool = False) -> str:
    lines = ['Say this turn:', *[f'  - {_content(line)}' for line in say], '', *map(_content, snapshot), '',
             'Last turns:']
    for number, text in last_turns:
        lines += [f'  Turn {number}: {_content(text)!r}'] + [f'    You said: {_content(reply)!r}'
                                                             for turn, reply in replies if turn == number]
    lines += [] if last_turns else ['  (none)']
    lines += ['', 'Events so far:', *[f'  {_content(line)}' for line in older]] if older else []
    lines += ['', NO_QUESTION] if regenerated else []
    return '\n'.join(lines)


def decomposer_examples() -> List[str]:
    return [HEADER] + [line for name in Examples.DECOMPOSER
                         for line in _shown(Examples.EXAMPLES[name], decomposition(Examples.EXAMPLES[name]))]


def single_call_examples() -> List[str]:
    return [HEADER] + [line for example in Examples.EXAMPLES.values() for line in _shown(example, planned(example))]


def filler_examples(role: Role) -> List[str]:
    return [HEADER] + [line for example, part in Examples.filler_parts(role)
                         for line in _shown(example, filling(example, part),
                                            *_asked(f'Part: {_content(part.words)!r}',
                                                    [other.words for other in example.plan.parts()
                                                     if other.id != part.id]))]


def template_filler_examples(template: str) -> List[str]:
    return [HEADER] + [line for example in Examples.EXAMPLES.values() if example.template == template
                       for line in _shown(example, template_filling(example))]


def template_filling(example: Examples.Example) -> Dict[str, Any]:
    kinds = Templates.TEMPLATES[example.template].slots
    slots = {name: value if kinds[name] == 'rows' or isinstance(value, dict) else {'value': value, 'status': 'certain'}
             for name, value in example.slots.items()}
    return slots | {name: _listed(slots[name]) for name in slots if kinds[name] not in ('rows', 'words', 'description')}


def decomposition(example: Examples.Example) -> Dict[str, Any]:
    k = example.plan.k
    readings = {'readings': [_tree(reading) for reading in example.readings]} if example.readings else {}
    tree = {'template': example.template} if example.template and not example.plan.template_parts \
        else _tree(example.plan.root, example.plan)
    return {'tree': tree, **readings, 'fit': 'fits',
            'k': None if k.status == 'not_stated' else k.value, 'k_status': k.status}


def planned(example: Examples.Example) -> Dict[str, Any]:
    k = example.plan.k
    fills = []
    for part in example.plan.parts():
        fill = part.fill.model_dump(mode='json', exclude_defaults=True)
        fill.update({name: {'value': fill[name]['value']} for name, slot in part.fill if isinstance(slot, Slot)})
        fills.append({'part': part.id, 'fill': fill})
    return {'tree': _tree(example.plan.root), 'fills': fills, 'k': None if k.status == 'not_stated' else k.value}


def filling(example: Examples.Example, part: Part) -> Dict[str, Any]:
    fill = part.fill.model_dump(mode='json', exclude_defaults=True)
    return {'fill': fill | {name: _listed(fill[name]) for name in COLUMN_SLOTS if name in fill},
            'seeker_status': part.seeker_status,
            'seeker_alternatives': [other.fill.seeker for other in example.alternatives if other.id == part.id]}


def _listed(slot: Dict[str, Any]) -> Dict[str, Any]:
    return {'value': slot['value'], 'alternatives': list(slot.get('alternatives', [])), 'status': slot['status']}


def _tree(node: Union[Part, Combiner], plan: Optional[SearchPlan] = None) -> Dict[str, Any]:
    ids = {inner.id for inner in walk(node) if isinstance(inner, Part)}
    if plan and plan.template_parts and ids == set(plan.template_parts):
        return {'template': plan.template, 'words': _content(walk(node)[0].words)}
    if isinstance(node, Part):
        return {'id': node.id, 'words': _content(node.words)}
    if node.kind == 'difference':
        return {'combiner': 'difference', 'kept': _tree(node.inputs[0], plan), 'removed': _tree(node.inputs[1], plan),
                'removal': node.removal}
    return {'combiner': node.kind, 'parts': [_tree(child, plan) for child in node.inputs]}


def _shown(example: Examples.Example, answer: Dict[str, Any], *asked: str) -> List[str]:
    return ['', f'Query table {example.table}: {Examples.TABLES[example.table]}', f'Request: {example.request!r}',
            *asked, f'Answer: {json.dumps(answer)}']
