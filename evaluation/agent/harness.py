import argparse
import configparser
import dataclasses
import itertools
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

os.environ.setdefault('PYDANTIC_AI_NO_BANNER', '1')
os.environ.setdefault('TQDM_DISABLE', '1')
os.environ.setdefault('HF_HUB_VERBOSITY', 'error')

import yaml
from pydantic import AfterValidator, BaseModel, ConfigDict, Field, TypeAdapter, ValidationError, model_validator
from pydantic_ai.models.function import FunctionModel

from evaluation.nlseeker import datasets
from evaluation.nlseeker.prepare import CACHE_DIR, DATASETS_DIR, prepare_dataset
from src.Agent import Compiler, Planner, Profiler
from src.Agent.Compiler import k_rule
from src.Agent.Config import PLANNERS, ROLES, AgentConfig
from src.Agent.Dialogue import Click, DialogueManager, LoadTable, Message, NoPreference, Other
from src.Agent.Plans import GOALS, Combiner, Part, SearchPlan, Slot, walk
from src.Agent.Profile import QueryTable
from src.Agent.Record import to_json
from src.Agent.Sampling import build_model
from src.DBHandler import DBHandler

from typing import Annotated, Any, Callable, Dict, Hashable, List, Literal, Optional, Sequence, Tuple, Union
from pydantic_ai.models import Model
from pydantic_ai.messages import ModelResponse
from pydantic_ai.models.function import AgentInfo
from src.Agent.Dialogue import Snapshot
from src.Agent.Questions import Question

LAKES = {'chembl': 'config/chembl.ini', 'chicago': 'config/chicago_open.ini',
         'adventure_works': 'config/adventure_works.ini'}
DATASETS = {'chembl': 'chembl_10K', 'chicago': 'chicago_10K', 'adventure_works': 'adventure_works'}
BUDGETS = range(4)
ROOT = Path(__file__).resolve().parents[2]
STRUCTURAL = ('task', 'combiner', 'direction', 'removal', 'part_count')
SEEKER_TEMPLATES = {'single_column': 'single_column_join', 'multi_column': 'multi_column_join', 'keyword': 'keyword',
                    'correlation': 'correlation', 'natural_language': 'nl_search'}
SENTENCES = {('combiner', 'intersection'): 'I mean only the tables that every one of these searches finds.',
             ('combiner', 'union'): 'I mean the tables that any one of these searches finds.',
             ('combiner', 'counter'): 'I mean the tables these searches find, ranked by how many of them find each.',
             ('removal', 'exclusion'): 'I mean removing every table the other search finds, not only its best matches.',
             ('removal', 'comparison'): "I mean removing only the other search's best matches, not every table it "
                                        'finds.'}
TRIGGER_THRESHOLD, CALIBRATED = 'any_disagreement', False
CONDITION = 'main'
MAX_ANSWERS = 20


class Strict(BaseModel):
    model_config = ConfigDict(extra='forbid')


LabelledField = Literal['seeker', 'column', 'columns', 'join', 'target', 'key', 'fill', 'task', 'combiner',
                        'direction', 'removal', 'part_count']


class OpenField(Strict):
    part: str
    field: LabelledField
    intended: Any
    opened_as: Literal['not_stated', 'ambiguous']
    valid_values: List[Any]
    why_open: str
    named_first: Optional[Literal['kept', 'removed']] = None

    @model_validator(mode='after')
    def _consistent(self) -> 'OpenField':
        if len(self.valid_values) < 2 or not any(_same(value, self.intended) for value in self.valid_values):
            raise ValueError(f'{self.part}.{self.field}: the intended value is not one of at least two valid values')
        if self.field in ('direction', 'removal') and self.opened_as != 'ambiguous':
            raise ValueError(f'{self.part}.{self.field}: a direction or a removal is opened only as ambiguous')
        if (self.field == 'direction') != (self.named_first is not None):
            raise ValueError(f'{self.part}.{self.field}: named_first is given exactly for a direction')
        return self


class Inferable(Strict):
    part: str
    field: LabelledField
    why: str


class Case(Strict):
    id: str
    level: Union[Literal[0, 1, 2, 3], Literal['inferable']]
    rounds_needed: int
    description: str
    open_fields: List[OpenField]
    inferable: Optional[Inferable] = None

    @model_validator(mode='after')
    def _consistent(self) -> 'Case':
        opened = 0 if self.level == 'inferable' else self.level
        if len(self.open_fields) != opened or (self.level == 'inferable') != (self.inferable is not None):
            raise ValueError(f'{self.id}: a level opens that many fields, an inferable case none and names its field')
        if self.rounds_needed != rounds_needed(self.open_fields):
            raise ValueError(f'{self.id}: rounds_needed is not the rounds its open fields need')
        return self


def rounds_needed(open_fields: List[OpenField]) -> int:
    structure = any(field.field in STRUCTURAL for field in open_fields)
    return structure + sum(field.field not in STRUCTURAL for field in open_fields)


class Stripped(Strict):
    path: str
    name: str


class GoldPlan(Strict):
    gold_plan_id: str
    lake: Literal[tuple(LAKES)]
    template: str
    tier: Literal['simple', 'tasks', 'free form']
    class_: Literal['seen', 'partly seen', 'new', 'task'] = Field(alias='class')
    query_table: str
    stripped: Optional[Stripped] = None
    gold_plan: Dict[str, Any]
    cases: List[Case] = Field(min_length=1)

    @model_validator(mode='after')
    def _labels_match(self) -> 'GoldPlan':
        try:
            gold = gold_plan(self, 1)
        except ValidationError:
            return self
        for case in self.cases:
            for field in case.open_fields:
                if not _same(_gold_value(gold, field), field.intended):
                    raise ValueError(f'{case.id}: {field.part}.{field.field} has not the intended value in the gold plan')
            if case.inferable and _gold_value(gold, case.inferable) is None:
                raise ValueError(f'{case.id}: the inferable field is no field of the gold plan')
        return self


def _gold_value(gold: SearchPlan, field: Union[OpenField, Inferable]) -> Any:
    if field.part == 't':
        slot = gold.shared(field.field)
        return slot and slot.value
    node = next((node for node in walk(gold.root) if node.id == field.part), None)
    if node is None:
        return None
    if field.field == 'part_count':
        return sum(isinstance(part, Part) for part in walk(node))
    if field.field == 'task':
        return (gold.task or 'tree') if node is gold.root else None
    if isinstance(node, Part):
        slot = node.fill.seeker if field.field == 'seeker' else getattr(node.fill, field.field, None)
        return slot.value if isinstance(slot, Slot) else slot
    if field.field == 'direction':
        return {'kept': node.inputs[0].id, 'removed': node.inputs[1].id} if node.kind == 'difference' else None
    return node.kind if field.field == 'combiner' else node.removal


def _unique(entries: List[GoldPlan]) -> List[GoldPlan]:
    ids = [case.id for entry in entries for case in entry.cases]
    if len(ids) != len(set(ids)):
        raise ValueError('two cases share an id, and the outputs are named by case id')
    return entries


CASE_FILE = TypeAdapter(Annotated[List[GoldPlan], AfterValidator(_unique)])


def load(path: Union[str, Path]) -> List[GoldPlan]:
    return CASE_FILE.validate_python(yaml.safe_load(Path(path).read_text()))


def find(path: Union[str, Path], case_id: str) -> Tuple[GoldPlan, Case]:
    for entry in load(path):
        for case in entry.cases:
            if case.id == case_id:
                return entry, case
    raise KeyError(f'no case {case_id!r} in {path}')


def gold_plan(entry: GoldPlan, k_default: int) -> SearchPlan:
    return SearchPlan.model_validate({'version': 0, 'k': {'value': k_default, 'status': 'not_stated'},
                                      **entry.gold_plan})


def correct(plan: SearchPlan, gold: SearchPlan, config: AgentConfig, table: QueryTable, lake: DBHandler) -> bool:
    return _canonical(plan, config, table, lake) == _canonical(gold, config, table, lake)


def _canonical(plan: SearchPlan, config: AgentConfig, table: QueryTable, lake: DBHandler) -> Hashable:
    return plan.k.value, _node(k_rule(plan.root, plan.k.value, config, table, lake))


def _node(node: Union[Part, Combiner]) -> Hashable:
    if isinstance(node, Part):
        return 'part', tuple(sorted(_fields(node).items())), node.k
    inputs = tuple(_node(child) for child in node.inputs)
    return node.kind, node.removal, node.k, inputs if node.kind == 'difference' else tuple(sorted(inputs, key=repr))


def _fields(part: Part) -> Dict[str, Hashable]:
    found = {'seeker': part.fill.seeker}
    for name, value in part.fill:
        if name == 'seeker' or part.fill.seeker == 'natural_language' and name == 'description':
            continue
        value = value.value if isinstance(value, Slot) else value
        if name == 'words':
            value = tuple(sorted({DBHandler.normalize_token(word) for word in value} - {''}))
        elif name == 'rows':
            columns = part.fill.row_columns()
            value = tuple(sorted({tuple(sorted(zip(columns, map(DBHandler.normalize_token, row)))) for row in value}))
        elif isinstance(value, tuple):
            value = tuple(sorted(value))
        found[name] = value
    return found


def align(plan: SearchPlan, gold: SearchPlan, open_fields: List[OpenField]) -> Optional[Dict[str, str]]:
    ours, theirs = _positions(plan.root), _positions(gold.root)
    open_ = {(field.part, field.field) for field in open_fields}

    def equal(our: Part, their: Part) -> int:
        mine = _fields(our) | {'position': ours[our.id]}
        wanted = _fields(their) | {'position': theirs[their.id]}
        return sum(name in mine and mine[name] == value for name, value in wanted.items()
                   if (their.id, name) not in open_)

    found = best_matching(plan.parts(), gold.parts(), equal)
    return None if found is None else {our.id: their.id for our, their in found}


def best_matching(ours: Sequence, theirs: Sequence, equal: Callable[[Any, Any], int]) -> Optional[List[Tuple]]:
    matchings = [list(zip(ours, chosen)) for chosen in itertools.permutations(theirs, min(len(ours), len(theirs)))] \
        if len(ours) <= len(theirs) else \
        [list(zip(chosen, theirs)) for chosen in itertools.permutations(ours, len(theirs))]
    scored = [(sum(equal(our, their) for our, their in matching), matching) for matching in matchings]
    best = max(score for score, _ in scored)
    winners = [matching for score, matching in scored if score == best]
    if len(winners) > 1:
        return None
    return [(our, their) for our, their in winners[0] if equal(our, their)]


def _positions(node: Union[Part, Combiner], path: Tuple = ()) -> Dict[str, Tuple]:
    if isinstance(node, Part):
        return {node.id: path}
    return {part: position for index, child in enumerate(node.inputs)
            for part, position in _positions(child, path + ((node.kind, index if node.kind == 'difference' else None),))
            .items()}


def answer(question: Question, mapping: Dict[str, str], open_fields: List[OpenField], gold: SearchPlan,
           version: Optional[int]) -> Union[Click, Other, NoPreference]:
    about = asked_about(question.kind, question.field, mapping, open_fields, gold)
    if not about:
        return NoPreference(question.id, version=version)
    if question.field == 'structure':
        fitting = [option for option in question.options if all(_fits(option.value, field, gold) for field in about)]
        if len(fitting) == 1:
            return Click(question.id, fitting[0].id, version=version)
        return Other(question.id, ' '.join(_stated(field, gold, open_fields) for field in about), version=version)
    labelled, = about
    option = next((option for option in question.options if _same(option.value, labelled.intended)), None)
    if option:
        return Click(question.id, option.id, version=version)
    sentence = f'I mean the {labelled.intended} column.' if labelled.field != 'seeker' \
        else f'I mean a {labelled.intended.replace("_", " ")} search.'
    return Other(question.id, sentence, version=version)


def asked_about(kind: str, field: Optional[str], mapping: Dict[str, str], open_fields: List[OpenField],
                gold: SearchPlan) -> List[OpenField]:
    if kind != 'clarification':
        return []
    structural = [found for found in open_fields
                  if found.field in STRUCTURAL or (gold.task and found.field == 'seeker')]
    if field == 'structure' and structural:
        return structural
    part, _, slot = (field or '').partition('.')
    return [found for found in open_fields
            if (found.part, found.field) == ('t' if part == 't' else mapping.get(part), slot)][:1]


def _fits(tree: Planner.Tree, field: OpenField, gold: SearchPlan) -> bool:
    node = _at(tree, _path(gold.root, field.part))
    if field.field == 'removal':
        return getattr(node, 'removal', None) == field.intended
    if field.field == 'direction':
        kept, removed = (_shape(_find(gold.root, field.intended[side])) for side in ('kept', 'removed'))
        return isinstance(node, Planner.DecomposedDifference) and kept != removed \
            and (_shape(node.kept), _shape(node.removed)) == (kept, removed)
    if field.field == 'part_count':
        single = isinstance(node, Planner.TemplateChoice) and node.template in SEEKER_TEMPLATES.values()
        return node is not None and (1 if single else len(Planner.parts(node))) == field.intended
    if field.field == 'seeker':
        return isinstance(node, Planner.TemplateChoice) and node.template == SEEKER_TEMPLATES[field.intended]
    if field.field == 'task':
        return node is not None and (node.template if isinstance(node, Planner.TemplateChoice) else 'tree') \
            == field.intended
    return getattr(node, 'combiner', None) == field.intended


def _stated(field: OpenField, gold: SearchPlan, open_fields: List[OpenField]) -> str:
    if field.field == 'direction':
        kept, removed = (_find(gold.root, field.intended[side]) for side in ('kept', 'removed'))
        if any(other.part in {node.id for side in (kept, removed) for node in walk(side)} for other in open_fields):
            raise NotImplementedError('a side with an open field cannot be named by its words')
        return f'I mean the results for {_named(kept)}, without the results for {_named(removed)}.'
    if field.field == 'part_count':
        return f"I mean it as {'one search' if field.intended == 1 else f'{field.intended} searches'}."
    if field.field == 'seeker':
        return f'I mean a {field.intended.replace("_", " ")} search.'
    if field.field == 'task':
        return f'I mean: {GOALS[field.intended]}.' if field.intended != 'tree' \
            else "I mean it built from separate searches, not as one of Blend's tasks."
    return SENTENCES[field.field, field.intended]


def _named(node: Union[Part, Combiner]) -> str:
    return ' and '.join(f"'{part.words.for_prompt()}'" for part in walk(node) if isinstance(part, Part))


def _shape(node: Union[Part, Combiner, Planner.Group]) -> Hashable:
    if isinstance(node, (Part, Planner.DecomposedPart)):
        return 'part'
    if isinstance(node, Planner.TemplatePart):
        return 'task', node.template
    return tuple(sorted(map(_shape, node.inputs if isinstance(node, Combiner) else Planner.children(node)), key=repr))


def _find(root: Union[Part, Combiner], id: str) -> Union[Part, Combiner]:
    return next(node for node in walk(root) if node.id == id)


def _path(root: Union[Part, Combiner], id: str) -> Tuple[int, ...]:
    return () if root.id == id else (next(index for index, child in enumerate(root.inputs) if child.id == id),)


def _at(tree: Union[Planner.Tree, Planner.Group], path: Tuple[int, ...]) -> Optional[Planner.Group]:
    for index in path:
        inputs = Planner.children(tree) \
            if isinstance(tree, (Planner.DecomposedCombiner, Planner.DecomposedDifference)) else []
        tree = inputs[index] if index < len(inputs) else None
    return tree


def _same(value: Any, intended: Any) -> bool:
    if isinstance(value, (list, tuple)) and isinstance(intended, (list, tuple)):
        return sorted(value) == sorted(intended)
    return value == intended


def cache_only(messages: list, info: AgentInfo) -> ModelResponse:
    raise RuntimeError('the evaluation reads semantic profiles from the frozen cache only')


def _converse(manager: DialogueManager, case: Case, gold: SearchPlan, asked: List[dict]) -> None:
    for _ in range(MAX_ANSWERS):
        if not manager.decisions:
            return
        question = manager.decisions[0]
        plan = manager.pending.plan if manager.pending else manager.plan
        mapping = align(plan, gold, case.open_fields) if plan else None
        event = answer(question, mapping or {}, case.open_fields, gold,
                       manager.plan.version if manager.plan else None)
        asked.append({'question': question.id, 'kind': question.kind, 'field': question.field,
                      'trigger': question.trigger, 'round': manager.questions.rounds, 'alignment': mapping,
                      'tie': plan is not None and mapping is None,
                      'unmatched': [part.id for part in plan.parts() if part.id not in (mapping or {})] if plan else [],
                      'options': [(option.id, option.value) for option in question.options],
                      'answer': type(event).__name__, 'option': getattr(event, 'option', None),
                      'text': getattr(event, 'text', None)})
        manager.handle(event)
    raise RuntimeError(f'{case.id}: still asking after {MAX_ANSWERS} answers')


def evaluated(lake: str, budget: int = max(BUDGETS), **settings) -> AgentConfig:
    return dataclasses.replace(AgentConfig.load(LAKES[lake]), **{
        'question_rounds': budget, 'clarification_rounds': budget, 'open_at_once': 1,
        'trigger_threshold': TRIGGER_THRESHOLD, 'fill_sides': False, 'verify': False, 'semantic_profile': True}
        | settings)


def step_costs(entries: List[dict]) -> List[dict]:
    return [{'role': entry['step'], 'seconds': max(sample['seconds'] for sample in entry['samples']),
             'input_tokens': sum(sample['input_tokens'] for sample in entry['samples']),
             'output_tokens': sum(sample['output_tokens'] for sample in entry['samples'])}
            for entry in entries if 'step' in entry and not entry.get('replayed')]


def run(entry: GoldPlan, case: Case, out: Path, repeat: int = 0, budgets: Sequence[int] = BUDGETS,
        models: Optional[Dict[str, Model]] = None, **settings) -> List[dict]:
    lake_config = LAKES[entry.lake]
    config = evaluated(entry.lake, max(budgets), record_dir=str(out / 'records'), **settings)
    models = (models or {role: build_model(config, role) for role in ROLES}) | {'profiler': FunctionModel(cache_only)}
    points: List[Tuple[Snapshot, int, float]] = []
    manager = DialogueManager(config, lake_config, models, snapshot=lambda snapshot: points.append(
        (snapshot, len(step_costs(manager.record.entries())), time.perf_counter() - start)))
    gold = gold_plan(entry, config.k_default)
    base = {'case': case.id, 'gold_plan_id': entry.gold_plan_id, 'lake': entry.lake,
            'template': entry.template, 'tier': entry.tier, 'class': entry.class_, 'level': case.level,
            'rounds_needed': case.rounds_needed, 'condition': CONDITION, 'repeat': repeat,
            'record': str(manager.record.path), 'verify': config.verify, 'planner': config.planner}

    start = time.perf_counter()
    manager.handle(LoadTable(entry.query_table))
    base['semantic_profile'] = manager.loaded['semantic_profile']
    if base['semantic_profile'] != ('on' if config.semantic_profile else 'off'):
        return [base | {'failed': 'profile'}]
    asked = []
    manager.handle(Message(case.description))
    _converse(manager, case, gold, asked)
    seconds = time.perf_counter() - start
    rounds = manager.questions.rounds
    steps = step_costs(manager.record.entries())

    rows = []
    for budget in budgets:
        finished, spent, wall, finishing, finishing_asked = manager, steps, seconds, 0.0, []
        if budget < rounds:
            snapshot, before, elapsed = next(point for point in points if point[0].rounds == budget)
            began = time.perf_counter()
            finished, _ = snapshot.resume()
            _converse(finished, case, gold, finishing_asked)
            finishing = time.perf_counter() - began
            spent, wall = steps[:before] + step_costs(finished.record.entries()), elapsed + finishing
        plan = finished.plan
        rows.append(base | {'budget': budget, 'rounds': rounds,
                            'correct': plan is not None and correct(plan, gold, config, manager.table, manager.lake),
                            'plan': plan, 'assumed': finished.card.assumed if finished.card else None,
                            'tables': finished.card.tables if finished.card else None,
                            'questions': [question for question in asked if question['round'] <= budget],
                            'finishing_questions': finishing_asked, 'seconds': wall, 'steps': spent,
                            'model_seconds': sum(step['seconds'] for step in spent),
                            'input_tokens': sum(step['input_tokens'] for step in spent),
                            'output_tokens': sum(step['output_tokens'] for step in spent),
                            'run_seconds': seconds, 'finishing_seconds': finishing,
                            'finishing_record': str(finished.record.path)})
    return rows


def gold_tables(entry: GoldPlan, out: Path) -> List[Tuple[int, str]]:
    path = out / 'gold' / f'{entry.lake}.{entry.gold_plan_id}.json'
    if not path.exists():
        config = evaluated(entry.lake)
        table = QueryTable.load(entry.query_table, config.example_length)
        lake = DBHandler()
        lake.load_config(LAKES[entry.lake])
        gold = gold_plan(entry, config.k_default)
        ruled = gold.model_copy(update={'root': k_rule(gold.root, gold.k.value, config, table, lake)})
        _write(path, Compiler.run(ruled, table, LAKES[entry.lake]).tables)
    return [tuple(table) for table in json.loads(path.read_text())]


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix('.partial')
    partial.write_text(to_json(value))
    os.replace(partial, path)


def open_cases(paths: Sequence[Union[str, Path]]) -> List[Tuple[Path, List[GoldPlan]]]:
    return [(Path(path), load(path)) for path in paths]


def build_profiles(entries: List[GoldPlan]) -> None:
    for entry in entries:
        config = evaluated(entry.lake)
        profiled = Profiler.profile(QueryTable.load(entry.query_table, config.example_length), config,
                                    build_model(config, 'profiler'))
        if not profiled.complete:
            errors = {column: profiled.records[column].error for column in profiled.missing}
            raise SystemExit(f'refused: no semantic profile for {entry.query_table}, columns {errors}; '
                             f'check the profiler endpoint and run again')


def output(out: Path, entry: GoldPlan, case: Case, repeat: int) -> Path:
    return out / 'rows' / f'{entry.lake}.{case.id}.{CONDITION}.{repeat}.json'


def dispatch(commands: Dict[Path, List[str]], parallel: int, logs: Path) -> List[Path]:
    logs.mkdir(parents=True, exist_ok=True)

    def one(item: Tuple[Path, List[str]]) -> None:
        output, command = item
        with (logs / f'{output.stem}.log').open('a') as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, cwd=ROOT)

    with ThreadPoolExecutor(parallel) as pool:
        list(pool.map(one, [(output, command) for output, command in commands.items() if not output.exists()]))
    return [output for output in commands if not output.exists()]


def jaccard(tables: Sequence[Tuple[int, str]], gold: Sequence[Tuple[int, str]], query_table: str) -> float:
    ours, theirs = ({tuple(table) for table in found if table[1] != Path(query_table).name} for found in (tables, gold))
    return len(ours & theirs) / len(ours | theirs) if ours | theirs else 1.0


def summary(rows: List[dict], gold: Dict[Tuple[str, str], Tuple[GoldPlan, Sequence[Tuple[int, str]]]]
            ) -> Dict[Tuple[str, int, str, Any], Dict[str, float]]:
    groups: Dict[Tuple[str, int, str, Any], List[dict]] = {}
    for row in rows:
        if 'failed' not in row:
            groups.setdefault((row['lake'], row['budget'], row['tier'], row['level']), []).append(row)

    def mean(values: List[float]) -> float:
        return sum(values) / len(values)

    def score(row: dict) -> float:
        entry, tables = gold[row['lake'], row['gold_plan_id']]
        return jaccard(row['tables'] or [], tables, entry.query_table)

    def answered_from(row: dict) -> Tuple[List[List[OpenField]], List[OpenField]]:
        entry, _ = gold[row['lake'], row['gold_plan_id']]
        open_fields = next(case for case in entry.cases if case.id == row['case']).open_fields
        plan = gold_plan(entry, 1)
        return [asked_about(question['kind'], question['field'], question['alignment'] or {}, open_fields, plan)
                for question in row['questions']], open_fields

    def question_counts(found: List[dict]) -> Dict[str, int]:
        counted = {'useful': 0, 'unnecessary': 0, 'missed': 0, 'open': 0}
        for about, open_fields in map(answered_from, found):
            settled = sum(about, [])
            useful = sum(map(bool, about))
            counted['useful'] += useful
            counted['unnecessary'] += len(about) - useful
            counted['missed'] += sum(field not in settled for field in open_fields)
            counted['open'] += len(open_fields)
        return counted

    return {key: {'cases': len(found), 'correct': sum(row['correct'] for row in found),
                  'seconds': mean([row['seconds'] for row in found]),
                  'model_seconds': mean([row['model_seconds'] for row in found]),
                  'tokens': mean([row['input_tokens'] + row['output_tokens'] for row in found]),
                  'questions': mean([len(row['questions']) for row in found]),
                  'jaccard': mean([score(row) for row in found]),
                  'identical': sum(score(row) == 1 for row in found)} | question_counts(found)
            for key, found in groups.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--cases', nargs='+', required=True, help='Case files.')
    parser.add_argument('--case', nargs='*', help='Only these case ids.')
    parser.add_argument('--repeats', type=int, default=1)
    parser.add_argument('--parallel', type=int, default=12,
                        help='Sessions at once.')
    parser.add_argument('--out', type=Path, default=Path('data/Evaluation/Agent/main'))
    parser.add_argument('--summary', action='store_true', help="Only print the summary of --out's rows; runs no case.")
    parser.add_argument('--effort', choices=('low', 'medium', 'xhigh'), help='Reasoning effort of every thinking role.')
    parser.add_argument('--no-thinking', action='store_true', help='Thinking off for every role but the profiler.')
    parser.add_argument('--model', help='Model of every role but the profiler, and of the NL judge; needs --url.')
    parser.add_argument('--url', help='Endpoint of --model.')
    parser.add_argument('--tokenizer', help="The NL judge's tokenizer for --model.")
    parser.add_argument('--no-profile', action='store_true', help='Semantic profile off.')
    parser.add_argument('--planner', choices=PLANNERS, help='staged (the system), single_call or text (the baselines).')
    parser.add_argument('--specified', action='store_true', help='Only the fully specified cases: level 0 and inferable.')
    parser.add_argument('--unit', help=argparse.SUPPRESS)
    parser.add_argument('--repeat', type=int, default=0, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.unit:
        _unit(args)
        return
    files = open_cases(args.cases)
    gold_plans = [entry for _, found in files for entry in found]
    if not args.summary:
        for lake in {entry.lake for entry in gold_plans}:
            prepare_dataset(datasets.DATASETS[DATASETS[lake]], DATASETS_DIR, CACHE_DIR)
            write_settings(lake, args)
    use_settings(args.out)
    if args.summary:
        report([row for found in sorted((args.out / 'rows').glob('*.json')) for row in json.loads(found.read_text())],
               gold_plans, args.out)
        return
    if not args.no_profile:
        for _, entries in files:
            build_profiles(entries)
    commands = {}
    for path, entries in files:
        for entry in entries:
            chosen = [case for case in entry.cases if (not args.case or case.id in args.case)
                      and (not args.specified or case.level in (0, 'inferable'))]
            if chosen:
                gold_tables(entry, args.out)
            for case in chosen:
                for repeat in range(args.repeats):
                    commands[output(args.out, entry, case, repeat)] = [
                        sys.executable, '-m', 'evaluation.agent.harness', '--cases', str(path), '--unit', case.id,
                        '--repeat', str(repeat), '--out', str(args.out)] + [
                        '--no-profile'] * args.no_profile
    missing = dispatch(commands, args.parallel, args.out / 'logs')
    rows = [row for found in commands if found.exists() for row in json.loads(found.read_text())]
    report(rows, gold_plans, args.out)
    print(f"failed: {', '.join(row['case'] for row in rows if 'failed' in row) or 'none'}; "
          f"without an output (see {args.out / 'logs'}): {', '.join(found.stem for found in missing) or 'none'}")


def report(rows: List[dict], entries: List[GoldPlan], out: Path) -> None:
    named = {(row['lake'], row['gold_plan_id']) for row in rows if 'failed' not in row}
    gold = {(entry.lake, entry.gold_plan_id): (entry, gold_tables(entry, out)) for entry in entries
            if (entry.lake, entry.gold_plan_id) in named}
    for (lake, budget, tier, level), found in sorted(summary(rows, gold).items(), key=str):
        print(f"{lake} budget {budget} {tier} level {level}: {found['correct']}/{found['cases']} correct, "
              f"Jaccard {found['jaccard']:.2f}, {found['identical']}/{found['cases']} identical, "
              f"{found['seconds']:.0f} s, {found['model_seconds']:.0f} model s, {found['tokens']:.0f} tokens, "
              f"{found['questions']:.1f} questions ({found['useful']} useful, {found['unnecessary']} unnecessary), "
              f"{found['missed']}/{found['open']} open fields never asked")


def _unit(args: argparse.Namespace) -> None:
    path, = args.cases
    entry, case = find(path, args.unit)
    use_settings(args.out)
    _write(output(args.out, entry, case, args.repeat),
           run(entry, case, args.out, args.repeat, semantic_profile=not args.no_profile))


def write_settings(lake: str, args: argparse.Namespace) -> None:
    if args.model and not args.url:
        raise SystemExit('--model needs --url')
    parser = configparser.ConfigParser(interpolation=None)
    parser.read(LAKES[lake])
    agent, nl = parser['Agent'], parser['NLSeeker']
    for role in ROLES:
        if role != 'profiler':
            if args.model:
                agent[f'{role}_model'], agent[f'{role}_base_url'] = args.model, args.url
            if args.no_thinking:
                agent[f'{role}_thinking'] = 'false'
        if args.effort and agent.getboolean(f'{role}_thinking'):
            agent[f'{role}_reasoning_effort'] = args.effort
    if args.planner:
        agent['planner'] = args.planner
    if args.model:
        nl['llm_model'], nl['llm_base_url'] = args.model, args.url
        nl['llm_tokenizer'] = args.tokenizer or nl['llm_tokenizer']
    path = args.out / 'config' / f'{lake}.ini'
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w') as file:
        parser.write(file)


def use_settings(out: Path) -> None:
    for lake in LAKES:
        if (out / 'config' / f'{lake}.ini').exists():
            LAKES[lake] = str(out / 'config' / f'{lake}.ini')


if __name__ == '__main__':
    main()
