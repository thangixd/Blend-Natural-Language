from collections import Counter
from copy import deepcopy
from dataclasses import dataclass, field, replace
from itertools import count
from time import perf_counter

from pydantic import PositiveInt, ValidationError

from src.Agent import Chat, Clarify, Compiler, Planner, Profiler, Prompts, Render, Router, Templates, Verifier
from src.Agent.Config import ROLES
from src.Agent.Plans import (COLUMN_SLOTS, GOALS, REQUESTED, WORDING, Combiner, Part, SearchPlan, Seeker, Slot, held,
                             still_task, unranked, walk)
from src.Agent.Profile import QueryTable
from src.Agent.Questions import NONE_OF_THESE, OTHER_KINDS, Option, Question, Questions
from src.Agent.Record import Record
from src.Agent.Render import PlanCard
from src.Agent.Sampling import build_model, canonical, draw, groups, model_settings
from src.Agent.Text import Line, ModelText, UserText, joined
from src.DBHandler import DBHandler

from pathlib import Path
from typing import (Any, Callable, Collection, Dict, FrozenSet, Hashable, Iterator, List, Optional, Sequence, Set,
                    Tuple, Union)
from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.models import Model
from src.Agent.Config import AgentConfig
from src.Agent.Sampling import Sample

Answer = Tuple[str, Union[str, Line], UserText, str]
Select = Callable[[List[Clarify.Field], int], List[Clarify.Field]]
Finding = Tuple[str, Union[str, Line]]

COULD_NOT_PLAN = 'I could not turn this request into a search. Please rephrase it.'
NOT_UNDERSTOOD = 'I could not tell what your message asks for. Please rephrase it.'
NOT_OPEN = 'That question is not open, so there is nothing to answer.'
ONE_OPTION = 'That question takes one option.'
NOT_A_VALUE = 'That is no valid value for {}, so the question stays open.'
NO_PLAN = 'There is no plan yet, so there is nothing to change.'
NO_WORDING = 'That part of the plan has no wording to change.'
NO_OTHER_READING = 'I have no other reading of your request. Please rephrase the request.'
NO_DIFFERENT_PLAN = "I couldn't turn your change into a different plan. Please rephrase the request."
CANT_EXPRESS = "Blend can't express this as one plan. Please rephrase the request."
FACT = '{} Build the search from parts instead, or rephrase the request?'
FROM_PARTS = 'build it from parts'
NOT_THE_TASK = 'Do not choose a task whose fact fails, write a tree of parts: {}'
DROPPED_PART = 'That part was removed by your earlier choice. To include it again, ask to add it.'
NO_CHANGE = "That doesn't change the plan."
NO_EDIT_AT_CAP = ("Revisions for this request are used up, and I couldn't apply this as a change. Rephrase the change or "
                  "start a new request.")
VERBATIM = (NO_OTHER_READING, NO_DIFFERENT_PLAN, CANT_EXPRESS, DROPPED_PART, NO_CHANGE, NO_EDIT_AT_CAP)
REPEATED = 'Each further reading the Planner gave finds the same tables as a plan the user rejected.'
ABSTAINED = 'The Planner answered that the request has no reading besides the plans the user rejected.'
NOTHING_NEW = "The Planner answered that the request, with the user's change, has no plan besides the plans the user " \
              "rejected."
CLARIFICATION, CLICK, EDITOR = 'clarification', 'click', 'editor'
ABSTENTIONS = {Planner.NoOtherReading: ABSTAINED, Planner.NoDifferentPlan: NOTHING_NEW}
EXHAUSTED = 'On an earlier Not it of this plan the Planner already had no further reading of the request.'
INVALID = "None of the Planner's answers had the form a plan must have."
UNCHANGED = 'The change the Edit gave leaves every field of the plan as it was.'
SAME_TABLES = 'Your change is applied. It finds the same tables as before.'
NO_EDIT = ("I couldn't apply your change, so the plan shown is unchanged. Reject it with Not it and say what you want, "
           "or rephrase the change.")
REVISIONS_USED = 'This request has used its {} revisions, so Not it is off. Please start a new request.'
STALE = 'That result is no longer current, so nothing was changed.'
DROPPED = '{} ({!r}) is no longer in the plan, with the {} you set.'
SLOT_DROPPED = '{} is now a {} search, so the {} you set for it is no longer in the plan.'
SHARED_DROPPED = 'No part searches {} any more, so the value you set for it is no longer in the plan.'
REPLACED = 'This new request replaced plan v{} and the choices made for it.'
ADDS_PART = ('The newest message adds a part to the plan shown: keep its parts with their ids and the way they '
             'combine, and place the new part in the tree.')
REPEATS = 'The plan equals a plan the user rejected: {}'
REPEATS_TABLES = 'The plan finds the same tables as a plan the user rejected: {}'
CHANGES_STRUCTURE = 'The plan changes the structure the user picked: {}'
CHANGES_TASK = 'The plan leaves the task the user picked: {}'
KEEPS_PLAN = 'The plan does not keep the parts of the plan shown and the way they combine: {}'
RESTORES = 'The plan brings back a value the user changed: {} = {!r}'
KEEPS_OLD = 'The change keeps the value the user asked to change: {} = {!r}'
ALTERS_LOCKED = 'The change alters a value the user set: {} = {!r}'
BRINGS_BACK = 'The plan brings back a part the user dropped: {}'
TASK_FACTS = "The task's slots do not fit the query table: {}"
SAME_SEARCH = 'The plan searches the same twice: {} and {} both find tables that {}.'
DOES_NOT_FIT_TASK = "The request does not fit the slots of these Blend tasks: {}."
DOES_NOT_FIT = 'The plan does not hold the whole request.'
NO_SEEKER = 'No Blend search fits a part of the plan.'
NOT_FILLED = 'A part of the plan could not be filled.'
LOCKED_ROWS = 'Each typed row needs one value per column the user set: {}.'
WHICH = 'Which do you mean?'
WHICH_QUESTION = 'Which question are you answering?'
EMPTY_MEAN_NOTE = "numpy's mean of an empty slice, the correlation seeker's all-empty target"
REJECT_OVER_EDIT = 'A rejection with text over an edit: both re-plan with the message.'
EDIT_AT_CAP = 'An edit over a rejection at the revision cap: the rejection would be refused.'
READ_AS_EDIT = 'read as a change to the plan, not as a rejection (revisions are used up)'
CANT_DO = {'too_deep': "Blend can't do that: the request needs more than two levels of combined searches.",
           'no_combiner': "Blend can't do that: no way of combining searches in Blend relates these parts.",
           'none': "Blend can't do that: no Blend search fits part of the request."}
CANT_DO_PICK = ' Which of these should run?'
REPHRASE = 'rephrase the request'
UNPLACED = 'These words were not used: {}. Continue without them, or rephrase?'
CONTINUE = 'continue without them'
PLEASE_REPHRASE = 'Please rephrase the request.'
WAITS = 'I have a question about this too; it waits until you answer one of the {} open questions.'
ANSWERED = 'The user answered the question {!r} with: {!r}'
NO_PREFERENCE = 'no preference'
STRUCTURE = 'Which of these searches do you mean?'
SEEKER = 'Which kind of search should {} ({!r}) be?'
EXPLAIN = ('Explain the plan card: what was searched and what came back.',
           'The card shows the tables found, with the plan in words.')
SAY = 'Say this in your own words: {}'
AFTER = ('The user was just told: {!r} The reason: {} Explain that reason in plain words, without repeating what the '
         'user was told.')
ANSWER = ('Answer the user about {!r} from the snapshot; the answer changes nothing.',
          'I can only answer from the plan card and the open questions shown.')
ASK_BACK = ('Explain the open question {}, {!r}, and its options; it stays open.',
            'The question {!r} stays open; pick one of its options or type your answer.')
OUT_OF_SCOPE = ('Reply briefly to a message that is not about finding tables.',
                'I can only help find tables in the lake.')
NOT_CHECKED = 'This plan was not checked against the request.'
FINDING = 'A check of the plan against the request found this about the part; fix it if the request supports it: {}'
VERIFY_OFF = 'verify is off'
SINGLE_CALL_UNCHECKED = 'the single-call planner has no Verifier'
PROFILED = ('decomposer', 'filler', 'edit')
NO_SEMANTICS = 'No semantic profile for {}: the profiler gave no answer for them, so they are planned from the data ' \
               'profile alone.'


class Check(str):
    kind: str

    def __new__(cls, kind: str, *values: Any) -> 'Check':
        check = super().__new__(cls, kind.format(*values))
        check.kind = kind
        return check


@dataclass
class LoadTable:
    path: Union[str, Path]
    header: bool = True
    name: Optional[str] = None


@dataclass
class Message:
    text: UserText

    def __post_init__(self) -> None:
        self.text = _user(self.text)


@dataclass(kw_only=True)
class Control:
    version: Optional[int]


@dataclass
class Click(Control):
    decision: str
    option: Union[str, Tuple[str, ...]]


@dataclass
class SetK(Control):
    k: int


@dataclass
class SetWording(Control):
    part: str
    value: Union[str, Tuple[str, ...]]


@dataclass
class NoPreference(Control):
    decision: str


@dataclass
class Other(Control):
    decision: str
    text: UserText

    def __post_init__(self) -> None:
        self.text = _user(self.text)


@dataclass
class NotIt(Control):
    text: Optional[UserText] = None

    def __post_init__(self) -> None:
        self.text = None if self.text is None else _user(self.text)


@dataclass
class Reply:
    text: Union[str, ModelText, Line]
    fact: Optional[str] = None
    reason: Optional[str] = None


@dataclass
class Rejected:
    plan: SearchPlan
    tables: FrozenSet[int]
    words: str


@dataclass
class Verdict:
    discrepancies: List[BaseModel] = field(default_factory=list)
    failed: bool = False


@dataclass
class Pending:
    request: List[Tuple[int, UserText]]
    decompositions: List[Planner.Decomposition]
    plan: SearchPlan
    asked: List[str]
    answers: List[Answer] = field(default_factory=list)


@dataclass
class Response:
    shown: Union[None, PlanCard, Question, Reply]
    decisions: List[Question] = field(default_factory=list)
    raised: List[Question] = field(default_factory=list)
    chat: Union[None, ModelText, Line, str] = None


Event = Union[LoadTable, Message, Click, NoPreference, Other, SetK, SetWording, NotIt]
Drawn = Tuple[str, str, List[Sample]]


@dataclass
class Snapshot:
    rounds: int
    manager: 'DialogueManager'
    event: Event
    drawn: List[Drawn]

    def resume(self) -> Tuple['DialogueManager', Response]:
        manager = self.manager._resumed(self.rounds, self.drawn)
        response = manager.handle(self.event)
        if manager.replay:
            raise RuntimeError(f'The resumed copy left {len(manager.replay)} steps of the run unreplayed.')
        return manager, response


class DialogueManager:
    def __init__(self, config: AgentConfig, lake_config: Union[str, Path],
                 models: Optional[Dict[str, Model]] = None, select: Select = Clarify.select,
                 snapshot: Optional[Callable[[Snapshot], None]] = None) -> None:
        self.config, self.select, self.snapshot = config, select, snapshot
        self.lake_config = lake_config
        self.models = models or {role: build_model(config, role) for role in ROLES}
        self.record = Record(config.record_dir)
        self.lake = DBHandler()
        self.lake.load_config(lake_config)
        self.table: Optional[QueryTable] = None
        self.loaded: Dict[str, Any] = {}
        self.plan: Optional[SearchPlan] = None
        self.version = 0
        self.replaced: Optional[int] = None
        self.card: Optional[PlanCard] = None
        self.words: Optional[str] = None
        self.seen: Optional[SearchPlan] = None
        self.verdict = Verdict()
        self.showing = False
        self.locked: Dict[str, Any] = {}
        self.lock_sources: Dict[str, str] = {}
        self.undone: Dict[str, Any] = {}
        self.dropped_parts: Dict[str, ModelText] = {}
        self.readings: List[Planner.StoredReading] = []
        self.request: List[Tuple[int, UserText]] = []
        self.rejected: List[Rejected] = []
        self.exhausted: Optional[SearchPlan] = None
        self.messages: List[Tuple[int, UserText]] = []
        self.questions = Questions(config, self.record)
        self.questions.on_round = self._snapshot
        self.before: Optional[Tuple[DialogueManager, Event]] = None
        self.drawn: List[Drawn] = []
        self.replay: List[Drawn] = []
        self.request_turn: Optional[int] = None
        self.fillings: Dict[str, Tuple[ModelText, List[List[BaseModel]]]] = {}
        self.pending: Optional[Pending] = None
        self.sides: List[Tuple[ModelText, Optional[Hashable]]] = []
        self.declined: Set[str] = set()
        self.misfits: List[str] = []
        self.task_repairs = 0
        self.replan_answers: List[Line] = []
        self.not_used: List[ModelText] = []
        self.chosen: List[Tuple[Optional[str], Line, bool]] = []
        self.replanned: Set[int] = set()
        self.checked: Optional[Planner.StoredReading] = None
        self.swaps: List[Tuple[Part, Part]] = []
        self.commands_assumed = False
        self.read_as_edit = False
        self.to_say: List[Tuple[Union[str, Line], Union[str, Line]]] = []
        self.events: List[Line] = []
        self.answered: List[Line] = []
        self.replies: List[Tuple[int, Union[str, ModelText, Line]]] = []

    @property
    def decisions(self) -> List[Question]:
        return self.questions.open

    def handle(self, event: Event) -> Response:
        self.before, self.drawn = None, []
        if self.snapshot:
            self.before = self._copy(), event
        if isinstance(event, LoadTable):
            return self._load(event)
        before = {question.id for question in self.questions.open}
        self.questions.new_event()
        self.to_say, self.read_as_edit = [], False
        if isinstance(event, Control) and event.version != (self.plan.version if self.plan else None):
            self.record.write(event='stale', turn=len(self.messages), control=event)
            response = self._respond(Reply(STALE))
        elif isinstance(event, Click):
            self.record.write(event='click', decision=event.decision, option=event.option)
            response = self._resolve(len(self.messages), event.decision, event.option)
        elif isinstance(event, NoPreference):
            self.record.write(event='no_preference', decision=event.decision)
            response = self._resolve(len(self.messages), event.decision, f'{event.decision}.none')
        elif isinstance(event, Other):
            response = self._other(event.decision, event.text)
        elif isinstance(event, SetK):
            response = self._set_k(len(self.messages), event.k)
        elif isinstance(event, SetWording):
            response = self._set_wording(len(self.messages), event.part, event.value)
        elif isinstance(event, NotIt) and event.text is not None:
            turn = self._said(event.text)
            response = self._execute(turn, (turn, event.text), [Router.Reject(command='reject')])
        elif isinstance(event, NotIt):
            response = self._not_it(len(self.messages))
        else:
            response = self._message(event.text)

        raised = self.questions.raised(len(self.messages), before)
        response = replace(response, decisions=self.questions.open[::-1], raised=raised)
        line = _event_line(len(self.messages), event, response)
        response = self._chat(len(self.messages), response)
        self.events.append(line)
        self.showing = isinstance(response.shown, PlanCard)
        return response

    def _load(self, event: LoadTable) -> Response:
        self.table = QueryTable.load(event.path, self.config.example_length, event.header, event.name)
        shown, semantic = self.table.describe(), {'semantic_profile': 'off'}
        if self.config.semantic_profile:
            profiled = Profiler.profile(self.table, self.config, self.models['profiler'])
            self.table.semantics = profiled.columns
            semantic = {'semantic_profile': 'on' if profiled.complete else 'partial' if profiled.columns else 'missing',
                        'semantics': profiled.columns, 'semantic_records': profiled.records,
                        'contradictions': profiled.contradictions, 'cache_hit': profiled.cache_hit}
            shown += f'\n\n{NO_SEMANTICS.format(", ".join(profiled.missing))}' if profiled.missing else ''
        self.loaded = {'event': 'load_table', 'name': self.table.name, 'profile': self.table.profile,
                       'verifier': 'on' if self.config.verify else 'off', **semantic}
        self.record.write(**self.loaded)
        self.events.append(f'The query table {self.table.name} was loaded.')
        return self._respond(Reply(shown))

    def _message(self, text: UserText) -> Response:
        turn = self._said(text)
        return self._route(turn, text)

    def _other(self, decision: str, text: UserText) -> Response:
        turn = self._said(text)
        self.record.write(event='other', turn=turn, decision=decision)
        question = self.questions.find(decision)
        if question and question.about_request():
            self.request = [*self.request, (turn, text)]
        return self._resolve(turn, decision, f'{decision}.other', (turn, text), path='other')

    def _said(self, text: UserText) -> int:
        turn = len(self.messages) + 1
        self.messages.append((turn, text))
        self.record.write(event='message', turn=turn, text=text)
        return turn

    def _route(self, turn: int, text: UserText, declined: Optional[Question] = None) -> Response:
        showing = 'its results are showing' if self.showing \
            else 'its results are not showing: a reply or question came after them'
        plan = f'v{self.card.version} ({showing}): {self.words}' if self.card else None
        view = Prompts.router_view(text, turn, self.messages[-3:-1], plan, self.questions.lines(),
                                   self.table.short_context(), self.questions.lines([declined]) if declined else [],
                                   self.plan if self.card else None)
        schema = Router.routing_type({question.id: [option.id for option in question.options]
                                      + ([f'{question.id}.other'] if question.kind in OTHER_KINDS else [])
                                      for question in self.questions.open},
                                     [*self.plan.edit_targets(), Router.NEW_PART] if self.plan else [])
        samples, routings = self._step('router', turn, Router.router, view, schema,
                                       lambda routing: Router.commands_key(routing.commands))
        if not routings:
            return self._say(NOT_UNDERSTOOD)

        readings = {}
        for reading in _one_granularity([reading for group in routings for reading in _readings(group[0].commands)],
                                 self.locked):
            readings.setdefault(Router.commands_key(reading), reading)
        if len(readings) == 1:
            return self._execute(turn, (turn, text), next(iter(readings.values())))
        split = {command.command for reading in readings.values() for command in reading} == {'reject', 'edit_plan'}
        if split and len(self.rejected) < self.config.revision_rounds:
            self.record.write(event='split_resolved', turn=turn, readings=list(readings.values()),
                              rule=REJECT_OVER_EDIT)
            return self._execute(turn, (turn, text), [Router.Reject(command='reject')])
        repeated = declined is not None and set(readings) == {Router.commands_key(option.commands)
                                                              for option in declined.options}
        if repeated:
            self.record.write(event='not_asked_again', turn=turn, decision=declined.id)
        elif not self.questions.capped(turn, 'which'):
            return self._ask_which(turn, text, list(readings.values()))
        first = next(sample.output for sample in samples if sample.output is not None)
        commands = _readings(first.commands)[0]
        if split:
            edit = next(reading for reading in (commands, *readings.values())
                        if any(command.command == 'edit_plan' for command in reading))
            commands = [command for command in edit if command.command == 'edit_plan']
            self.record.write(event='split_resolved', turn=turn, readings=list(readings.values()), rule=EDIT_AT_CAP)
            self.read_as_edit = True
        response = self._execute(turn, (turn, text), commands, assumed=True)
        if split and isinstance(response.shown, Reply) and response.shown.fact in (NO_EDIT, CANT_EXPRESS):
            return self._say(NO_EDIT_AT_CAP, response.shown.reason)
        return response

    def _execute(self, turn: int, message: Tuple[int, UserText], commands: List[BaseModel],
                 assumed: bool = False) -> Response:
        acted = sorted({Router.commands_key([command]): command for command in commands}.values(),
                       key=lambda command: command.command != 'resolve')
        about_request = {question.id for question in self.questions.open if question.about_request()}
        if any(command.command == 'resolve' and command.decision in about_request for command in acted):
            self.request = [*self.request, message]
        response = self._respond(None)
        for command in acted:
            mark, self.commands_assumed = self.commands_assumed, assumed
            if command.command == 'chat':
                self.to_say.append((Line(ANSWER[0], command.topic), ANSWER[1]))
            elif command.command == 'ask_back':
                question = self.questions.find(command.decision)
                self.to_say.append((Line(ASK_BACK[0], question.id, question.text), Line(ASK_BACK[1], question.text))
                                   if question else (SAY.format(NOT_OPEN), NOT_OPEN))
            elif command.command == 'out_of_scope':
                self.to_say.append(OUT_OF_SCOPE)
            elif command.command == 'new_request':
                self._new_request(turn, message[0])
                self.request = [message]
                response = self._plan(turn, [message])
            elif command.command == 'reject':
                response = self._not_it(turn, message[1])
            elif command.command == 'edit_plan':
                response = self._edit(turn, message, command.targets)
            else:
                response = self._resolve(turn, command.decision, command.option, message, path='router')
            if not isinstance(response.shown, (PlanCard, Question)):
                self.commands_assumed = mark
        return response

    def _new_request(self, turn: int, request_turn: int) -> None:
        self.locked, self.lock_sources, self.undone, self.not_used, self.rejected = {}, {}, {}, [], []
        self.request_turn = request_turn
        self.chosen, self.replanned, self.checked = [], set(), None
        self.dropped_parts, self.pending, self.declined, self.fillings, self.sides = {}, None, set(), {}, []
        self.misfits, self.task_repairs, self.replan_answers = [], 0, []
        replaced = self.plan is not None
        if replaced:
            self.replaced, self.plan, self.card, self.seen, self.readings = self.plan.version, None, None, None, []
        self.questions.new_request(turn, request_turn, replaced)

    def _resolve(self, turn: int, decision: str, option: Union[str, Tuple[str, ...]],
                 message: Optional[Tuple[int, UserText]] = None, path: str = 'click') -> Response:
        question = self.questions.find(decision)
        if question and option == f'{decision}.none':
            return self._no_preference(turn, question, path)
        options = {choice.id: choice for choice in question.options} if question else {}
        picked = [options.get(choice) for choice in (option if isinstance(option, tuple) else (option,))]
        chosen = picked[0]
        other = message is not None and question is not None and question.kind in OTHER_KINDS \
            and option == f'{decision}.other'
        if None in picked and not other:
            return self._respond(Reply(NOT_OPEN))
        if len(picked) > 1 and question.kind != 'column':
            return self._respond(Reply(ONE_OPTION))
        if question.kind == 'column' and not other:
            value = self._column(question.field.split('.')[1], chosen.value if len(picked) == 1 else tuple(
                column for choice in picked for column in (choice.value if isinstance(choice.value, tuple)
                                                           else (choice.value,))))
            if value is None:
                return self._respond(Reply(NOT_A_VALUE.format(question.field)))
        self.questions.close(question)
        self.record.write(event='resolved', turn=turn, decision=decision, option=option, path=path)
        chose = Line('none of the options, {!r}', message[1]) if other \
            else joined(', ', [choice.name or choice.label for choice in picked])
        self.answered.append(Line('{} {}: {}', decision, question.text, chose))
        self.chosen.append((_set_field(question), Line('{}: {}', question.text, chose), False))
        if question.kind == 'clarification':
            return self._clarified(turn, question, chosen, message[1] if other else None)
        if other and question.kind == 'column':
            return self._fill_all(turn, question.request, question.decompositions,
                                  [(question.field, question.text, message[1], CLICK)])
        if other:
            if self.questions.left():
                self.questions.count(question)
            return self._route(turn, message[1], question)
        if chosen.label == REPHRASE:
            return self._respond(Reply(PLEASE_REPHRASE))
        if question.kind == 'unplaced':
            self.not_used += _constraints(question.decompositions)
            return self._decomposed(turn, question.request, question.decompositions)
        if question.kind == 'cant_do':
            filled = isinstance(chosen.plan, SearchPlan)
            if filled or Planner.task(chosen.plan.tree) is None:
                self._lock('structure', Planner.structure(chosen.plan.root if filled else chosen.plan.tree), CLICK)
            ids = {part.id for part in (chosen.plan.parts() if filled else Planner.parts(chosen.plan.tree))}
            self.dropped_parts = {part_id: words for part_id, words in question.parts.items() if part_id not in ids}
            if isinstance(chosen.plan, SearchPlan):
                return self._run_new(turn, chosen.plan, chosen.readings)
            return self._fill_all(turn, question.request, [chosen.plan])
        if question.kind == 'column':
            self._lock(question.field, value, CLICK)
            return self._fill_all(turn, question.request, question.decompositions)
        if question.kind == 'fact':
            return self._from_parts(turn, question.request, chosen.plan, chosen.value)
        return self._execute(turn, question.message, chosen.commands)

    def _no_preference(self, turn: int, question: Question, path: str) -> Response:
        self.questions.close(question)
        self.record.write(event='resolved', turn=turn, decision=question.id, option=f'{question.id}.none', path=path)
        self.answered.append(Line('{} {}: {}', question.id, question.text, NO_PREFERENCE))
        self.chosen.append((_set_field(question), Line('{}', question.text), True))
        if question.kind == 'clarification':
            return self._clarified(turn, question, None, None)
        proposal = question.options[0]
        if question.kind == 'which':
            return self._execute(turn, question.message, proposal.commands, assumed=True)
        if question.kind == 'unplaced':
            self.not_used += _constraints(question.decompositions)
            return self._decomposed(turn, question.request, question.decompositions)
        if question.kind == 'fact':
            return self._from_parts(turn, question.request, proposal.plan, proposal.value)
        if question.kind == 'cant_do':
            if proposal.plan is None:
                return self._say(COULD_NOT_PLAN)
            filled = isinstance(proposal.plan, SearchPlan)
            ids = {part.id for part in (proposal.plan.parts() if filled else Planner.parts(proposal.plan.tree))}
            self.not_used += [words for part_id, words in question.parts.items() if part_id not in ids]
            if filled:
                return self._run_new(turn, proposal.plan, proposal.readings)
            return self._fill_all(turn, question.request, [proposal.plan])
        self.declined.add(question.field)
        return self._fill_all(turn, question.request, question.decompositions, stored=True)

    def _ask_which(self, turn: int, text: UserText, alternatives: List[List[BaseModel]]) -> Response:
        answered = [{command.decision for command in commands if command.command in ('resolve', 'ask_back')}
                    for commands in alternatives]
        asked = WHICH_QUESTION if all(answered) and len(set().union(*answered)) > 1 else WHICH
        labels = {question.id: question.text for question in self.questions.open}
        labels.update({option.id: option.name or option.label for question in self.questions.open
                       for option in question.options})
        labels.update({f'{question.id}.other': NONE_OF_THESE for question in self.questions.open
                       if question.kind in OTHER_KINDS})
        choices = [(joined(' and ', [Router.describe(command, labels) for command in commands]), commands)
                   for commands in alternatives]
        return self._ask(turn, asked, choices, (turn, text))

    def _plan(self, turn: int, request: List[Tuple[int, UserText]], facts: Sequence[Union[str, Line]] = (),
              failed_check: Optional[str] = None) -> Response:
        if self.config.planner != 'staged':
            return self._single_call(turn, request, facts)
        failed = failed_check
        for _ in range(self.config.repair_rounds + 1):
            samples, decompositions = self._step(
                'decomposer', turn, Planner.decomposer,
                Prompts.decomposer_view(request, self.table, facts, examples=Prompts.decomposer_examples(),
                                        failed_check=failed),
                None, Planner.decomposition_key)
            if decompositions:
                break
            failed = samples[0].error
        else:
            return self._say(COULD_NOT_PLAN)
        outputs = [output for group in decompositions for output in group]
        constraints = _constraints(outputs)
        if constraints and not self.questions.capped(turn, 'unplaced'):
            decision = self.questions.new_id()
            return self._open(turn, Question(decision, Line(UNPLACED, joined(', ', constraints, '{!r}')),
                                             [Option(f'{decision}.o1', CONTINUE), Option(f'{decision}.o2', REPHRASE)],
                                             'unplaced', ('request', self.request_turn), turn, request=request,
                                             decompositions=outputs))
        self.not_used += constraints
        return self._decomposed(turn, request, outputs)

    def _single_call(self, turn: int, request: List[Tuple[int, UserText]], facts: Sequence[Line] = ()) -> Response:
        text, plan_type, failed = self.config.planner == 'text', Planner.single_call_type(self.table), None
        for _ in range(self.config.repair_rounds + 1):
            samples, plans = self._step('decomposer', turn, Planner.text_call if text else Planner.single_call,
                                        Prompts.decomposer_view(request, self.table, facts,
                                                                examples=Prompts.single_call_examples(),
                                                                failed_check=failed),
                                        None if text else plan_type, str if text else Planner.single_call_key,
                                        n=self.config.single_call_samples, step='single_call')
            if text and plans:
                try:
                    plans = [[Planner.from_text(plans[0][0], plan_type)]]
                except ValidationError as e:
                    failed = str(e)
                    self.record.write(event='unparsed', turn=turn, error=failed)
                    continue
            if plans:
                break
            failed = samples[0].error
        else:
            return self._say(COULD_NOT_PLAN)
        output = plans[0][0]
        root = _assemble(output.tree, iter(Planner.planned_parts(output)), (f'c{number}' for number in count(1)))
        k = Slot(value=output.k or self.config.k_default, status='certain' if output.k else 'not_stated')
        return self._run_new(turn, SearchPlan(version=0, k=k, root=root), [])

    def _decomposed(self, turn: int, request: List[Tuple[int, UserText]],
                    decompositions: List[Planner.Decomposition]) -> Response:
        decomposition = decompositions[0]
        if decomposition.fit != 'fits':
            self.record.write(event='cant_do', turn=turn, reason=decomposition.fit, tree=decomposition.tree,
                              sub_plans=decomposition.sub_plans)
            fitting = [decomposition.model_copy(update={'tree': tree, 'fit': 'fits', 'sub_plans': [], 'readings': []})
                       for tree in [decomposition.tree, *decomposition.sub_plans]]
            if self.questions.capped(turn, 'cant_do'):
                self.not_used += [part.words for sub_plan in fitting[1:] for part in Planner.parts(sub_plan.tree)]
                return self._fill_all(turn, request, fitting[:1])
            return self._cant_do(turn, request, decomposition.fit,
                                 [(Planner.outline(sub_plan.tree), sub_plan, []) for sub_plan in fitting],
                                 {part.id: part.words for sub_plan in fitting for part in Planner.parts(sub_plan.tree)})
        return self._fill_all(turn, request, decompositions)

    def _fill_all(self, turn: int, request: List[Tuple[int, UserText]], decompositions: List[Planner.Decomposition],
                  answers: Sequence[Answer] = (), stored: bool = False) -> Response:
        filled = self._filled(turn, request, decompositions, answers, stored=stored)
        if isinstance(filled, Templates.DoesNotFit):
            return self._misfit(turn, request, decompositions)
        return filled if isinstance(filled, Response) else self._clarify(turn, request, decompositions, *filled)

    def _clarify(self, turn: int, request: List[Tuple[int, UserText]], decompositions: List[Planner.Decomposition],
                 plan: SearchPlan, readings: List[Planner.StoredReading]) -> Response:
        room = self.questions.clarification_room() if self.plan is None else 0
        fitting = [output for output in decompositions if output.fit == 'fits']
        candidates = [field for field in Clarify.fields(plan, fitting, self._samples(plan),
                                                        self.config.trigger_threshold, self.table,
                                                        self._structure_key(fitting))
                      if field.name not in self.locked and field.name not in self.questions.fields]
        chosen = self.select(candidates, room) if room else []
        self.record.write(event='clarification', turn=turn, room=room, chosen=[field.name for field in chosen],
                          fields=[{'field': field.name, 'trigger': field.trigger, 'entropy': field.entropy,
                                   'chosen': field.chosen, 'statuses': field.statuses, 'valid': field.valid}
                                  for field in candidates])
        if not chosen:
            self.pending = None
            return self._run_new(turn, plan, readings)
        questions = [self._clarification(turn, field, plan, fitting) for field in chosen]
        self.pending = Pending(request, decompositions, plan, [question.id for question in questions])
        for question in questions:
            self.questions.ask(turn, question)
        return self._respond(questions[0])

    def _clarification(self, turn: int, field: Clarify.Field, plan: SearchPlan,
                       decompositions: List[Planner.Decomposition]) -> Question:
        counts, name = Counter(field.chosen), lambda value: None
        if field.kind == 'structure':
            proposal, key = decompositions[0].tree, self._structure_key(decompositions)
            produced = [[output.tree, *output.readings] for output in decompositions]
            filled = Line("{} (Blend's {} task)", Render.not_this(plan.model_copy(update={'structure': 'certain'})),
                          plan.task.replace('_', ' ')) if plan.task else None
            text, label = STRUCTURE, lambda tree: filled if filled and tree is proposal else Planner.outline(tree)
        elif field.part == 't':
            slot = field.name[2:]
            outputs = [output for group in self._samples(plan)['t'] for output in group]
            proposal, key = plan.shared(slot).value, canonical
            produced = [[getattr(output, slot).value, *getattr(output, slot).alternatives] for output in outputs]
            text, label, name = f'Which column should {field.name} use?', self.table.option, _named
        else:
            part = next(part for part in plan.parts() if part.id == field.part)
            outputs = [output for group in self._samples(plan)[part.id] for output in group]
            if field.kind == 'seeker':
                proposal, key = part.fill.seeker, lambda seeker: seeker
                produced = [[output.fill.seeker, *output.seeker_alternatives] for output in outputs]
                text, label = Line(SEEKER, part.id, part.words), lambda seeker: f'{seeker.replace("_", " ")} search'
            else:
                slot = field.name.split('.')[1]
                proposal, key = getattr(part.fill, slot).value, canonical
                produced = [[getattr(output.fill, slot).value, *getattr(output.fill, slot).alternatives]
                            for output in outputs if output.fill.seeker == part.fill.seeker]
                text = f'Which columns should {field.name} use (at least two)?' if slot == 'columns' \
                    else f'Which column should {field.name} use?'
                label, name = self.table.option, _named
        others = [value for _, value, _ in sorted(Planner.tally(produced, key, proposal), key=lambda other: -other[0])]
        values = [value for value in [proposal, *others] if value != 'none'][:4]
        decision = self.questions.new_id()
        options = [Option(f'{decision}.o{i}', label(value), value=value, samples=counts[key(value)], name=name(value))
                   for i, value in enumerate(values, 1)] + [Option(f'{decision}.none', NO_PREFERENCE)]
        return Question(decision, text, options, 'clarification', ('request', self.request_turn), turn,
                        field=field.name, trigger=field.trigger, valid=field.valid, statuses=tuple(field.statuses))

    def _clarified(self, turn: int, question: Question, chosen: Optional[Option], text: Optional[UserText]
                   ) -> Response:
        pending = self.pending
        if question.field == 'structure' and text is not None:
            self.not_used = []
            self.replan_answers = [Line(ANSWERED, question.text, text)]
            return self._plan(turn, pending.request, self.replan_answers)
        if question.field == 'structure' and chosen:
            task = Planner.task(chosen.value)
            if task is None:
                self._lock('structure', Planner.structure(chosen.value), CLARIFICATION)
            elif isinstance(task, Planner.TemplateChoice):
                self._lock('task', task.template, CLARIFICATION)
            picked = [output.model_copy(update={'tree': chosen.value, 'readings': []})
                      for output in pending.decompositions]
            return self._fill_all(turn, pending.request, picked)
        if text is not None:
            pending.answers.append((question.field, question.text, text, CLARIFICATION))
        elif chosen:
            self._lock(question.field, chosen.value, CLARIFICATION)
        waiting = [self.questions.find(decision) for decision in pending.asked if self.questions.find(decision)]
        if waiting:
            return self._respond(waiting[0])
        return self._fill_all(turn, pending.request, pending.decompositions, pending.answers, stored=True)

    def _filled(self, turn: int, request: List[Tuple[int, UserText]], decompositions: List[Planner.Decomposition],
                answers: Sequence[Answer] = (), ask: bool = True, failed_check: Optional[str] = None,
                mapped: bool = False, kept: Sequence[Part] = (), way_out: bool = False, stored: bool = False,
                findings: Sequence[Finding] = ()
                ) -> Union[None, Response, Check, Planner.NoDifferentPlan, Templates.DoesNotFit,
                           Tuple[Optional[SearchPlan], List[Planner.StoredReading]]]:
        tree = decompositions[0].tree
        task = Planner.task(tree)
        parts, seeker_readings, column_readings = [], [], []
        decomposed, roles, siblings = Planner.parts(tree), Planner.roles(tree), Planner.siblings(tree)
        words = [task.words] if isinstance(task, Planner.TemplatePart) else []
        kept = {part.id: part for part in kept}
        for part in decomposed:
            if part.id in kept:
                parts.append(kept[part.id])
                continue
            own = [answer for answer in answers if answer[0].split('.')[0] == part.id]
            locks = self._locks(part.id)
            samples = [[output for output in (_locked(filling, locks) for filling in group) if output]
                       for group in self._samples(tree, part.id)] if stored else []
            samples = [group for group in samples if group]
            if stored and any(field.split('.')[0] == part.id for field in self.declined):
                filled = None, [], []
            elif samples and not own:
                filled = self._part_from(turn, part.id, part.words, samples)[:3]
            else:
                filled = self._fill(turn, request, decompositions, part.id, part.words, roles[part.id], own, ask,
                                    failed_check=failed_check, mapped=mapped, taken={done.id for done in parts if done},
                                    way_out=way_out, beside=[done for done in parts
                                                             if done and done.id in siblings.get(part.id, ())],
                                    others=[other.words for other in decomposed if other.id != part.id] + words,
                                    findings=[line for node_id, line in findings if node_id == part.id])
            if isinstance(filled, Planner.NoDifferentPlan):
                return filled
            if isinstance(filled, Response):
                return filled if ask else None
            part, seekers, columns = filled
            parts.append(part)
            seeker_readings += seekers
            column_readings += columns
        task_root, template = None, {}
        if task:
            first = max([int(part.id[1:]) for part in [*decomposed, *filter(None, parts)]], default=0)
            filled = self._task(turn, request, decompositions, task,
                                [answer for answer in answers if answer[0].split('.')[0] not in roles], ask, stored,
                                [part.words for part in decomposed], first, _combiners(tree))
            if not isinstance(filled, tuple):
                return filled if ask or isinstance(filled, Check) else None
            task_root, task_readings, template = filled
            column_readings += task_readings
        root = _assemble(tree, iter(parts), (f'c{number}' for number in count(1)), task_root)
        if None in parts and not ask:
            return None, []
        if None in parts:
            kept = {node.id for node in walk(root)} if root else set()
            left_out = {part.id: part.words for part in decomposed if part.id not in kept}
            return self._cant_do_rest(turn, request, root, seeker_readings + column_readings, self._k(decompositions),
                                      left_out, template)
        same = self._same_search(turn, root, task, template.get('template_parts', ()))
        if same:
            return self._replanned(turn, request, failed_check=same) if ask else same
        if ask and self.config.fill_sides:
            self._fill_sides(turn, request, decompositions, parts)
        structural = Planner.tally([[output.tree, *output.readings] for output in decompositions],
                                   self._structure_key(decompositions) if ask else Planner.structure_key, tree)
        readings = [_reading(count, root, reading, alternative, template) for count, reading, alternative
                    in sorted(structural + seeker_readings + column_readings, key=lambda reading: -reading[0])]
        plan = SearchPlan(version=0, k=self._k(decompositions), root=root,
                          structure='ambiguous' if structural else 'certain', **template)
        return plan, readings

    def _task(self, turn: int, request: List[Tuple[int, UserText]], decompositions: List[Planner.Decomposition],
              task: Union[Planner.TemplateChoice, Planner.TemplatePart], answers: Sequence[Answer], ask: bool,
              stored: bool, others: Sequence[ModelText], first: int, combiners: int
              ) -> Union[None, Response, Templates.DoesNotFit,
                         Tuple[Union[Part, Combiner], List[Tuple[int, Union[Part, Combiner], bool]], Dict[str, Any]]]:
        template, part = task.template, isinstance(task, Planner.TemplatePart)
        words = task.words if part else ModelText(GOALS[template])
        locked = {name[2:]: value for name, value in self.locked.items() if name.startswith('t.')}
        facts = [f'Set by the user, keep it: t.{name} = {value!r}' for name, value in locked.items()]
        facts += [Line(ANSWERED, answer[1], answer[2]) for answer in answers]
        groups = self.fillings.get('t', (None, []))[1] if stored and not answers else []
        failed, found, drawn = None, None, False
        for _ in range(self.config.repair_rounds + 1):
            if not groups:
                drawn = True
                samples, groups = self._step(
                    'filler', turn, Planner.template_filler,
                    Prompts.template_filler_view(request, template, words if part else None, self.table, failed, facts,
                                                 Prompts.template_filler_examples(template), self._not_this(), others),
                    Templates.filling_type(template, self.table), Templates.filling_key(template),
                    step='template_filler')
                if not groups:
                    failed = samples[0].error
                    continue
                fitting = [group for group in groups if not isinstance(group[0], Templates.DoesNotFit)]
                if not fitting:
                    self.record.write(event='does_not_fit', turn=turn, template=template)
                    return groups[0][0] if ask else Check(DOES_NOT_FIT_TASK, template)
                groups = fitting
            groups = [[_set_slots(output, locked) for output in group] for group in groups]
            groups[0], found = Templates.first_meeting(template, groups[0], self.table, self.lake)
            if not found:
                differ = Templates.copies_differ(template, groups[0]) if drawn else []
                if differ:
                    self.record.write(event='copied_values_differ', turn=turn, template=template, slots=differ)
                break
            failed, groups = Check(TASK_FACTS, ' '.join(fact for _, fact in found)), []
        else:
            if found is None:
                return self._say(COULD_NOT_PLAN) if ask else None
            return self._ask_fact(turn, request, decompositions, template, found) if ask else failed
        self.fillings['t'] = (None, groups)
        proposal, readings = Templates.readings(template, groups, self.table, self.lake)

        def expanded(slots: BaseModel) -> Union[Part, Combiner]:
            node = Templates.expand(template, slots, words)
            return _renamed(node, first, combiners) if part else node

        sampled = proposal
        proposal, readings = self._task_overlapping(turn, proposal, readings, expanded)
        root = expanded(proposal)
        ids = [node.id for node in walk(root) if isinstance(node, Part)]
        expansions = [[expanded(output) for output in group] for group in groups]
        for part_id in ids:
            self.fillings[part_id] = words, [[node for output in group for node in walk(output)
                                              if isinstance(node, Part) and node.id == part_id] for group in expansions]
        for name, _, _, source in answers:
            value = getattr(proposal, name[2:]).value if name.startswith('t.') else _field(root, None, name, None)
            self._lock(name, value, source)
        root = self._honoured(root)
        if root is None:
            return self._say(COULD_NOT_PLAN) if ask else None
        if proposal is not sampled:
            olds = [node for node in walk(self._honoured(expanded(sampled))) if isinstance(node, Part)]
            self.swaps += [(new, old) for new, old in zip([node for node in walk(root) if isinstance(node, Part)], olds)
                           if new != old]
        return root, [(count, expanded(slots), alternative) for count, slots, alternative in readings], \
            {'template': template, 'template_parts': tuple(ids) if part else ()}

    def _task_overlapping(self, turn: int, proposal: BaseModel, readings: List[Tuple[int, BaseModel, bool]],
                          expanded: Callable[[BaseModel], Union[Part, Combiner]]
                          ) -> Tuple[BaseModel, List[Tuple[int, BaseModel, bool]]]:
        proposed = expanded(proposal)
        unshared = self._assumed_unshared(proposed)
        if not unshared:
            return proposal, readings
        fields = [f'{part_id}.{slot}' for part_id, slot, _ in unshared]
        ranked = ((reading, expanded(reading[1])) for reading in sorted(readings, key=lambda reading: -reading[0]))
        taken, swapped = next(((reading, root) for reading, root in ranked
                               if all(_field(root, None, field, None) != _field(proposed, None, field, None)
                                      for field in fields) and not self._assumed_unshared(root)), (None, None))
        for part_id, slot, name in unshared:
            self.record.write(event='no_overlap', turn=turn, part=part_id, slot=slot, value=name,
                              taken=_field(swapped, None, f'{part_id}.{slot}', None) if swapped else None)
        if taken is None:
            return proposal, readings
        return taken[1], [reading for reading in readings if reading is not taken]

    def _assumed_unshared(self, root: Union[Part, Combiner]) -> List[Tuple[str, str, str]]:
        parts = {node.id: node for node in walk(root) if isinstance(node, Part)}
        return [(part_id, slot, name) for part_id, slot, name in Compiler.missing(list(parts.values()), self.table,
                                                                                 self.lake)
                if slot not in REQUESTED and getattr(parts[part_id].fill, slot).status != 'certain']

    def _same_search(self, turn: int, root: Union[Part, Combiner], task: Union[None, Planner.TemplateChoice,
                                                                               Planner.TemplatePart] = None,
                     task_parts: Collection[str] = ()) -> Optional[Check]:
        same = Planner.repeated(root)
        return self._same_check(turn, *same, task, task_parts) if same else None

    def _same_check(self, turn: int, earlier: Union[Part, Combiner], later: Union[Part, Combiner],
                    task: Union[None, Planner.TemplateChoice, Planner.TemplatePart] = None,
                    task_parts: Collection[str] = ()) -> Check:
        check = Check(SAME_SEARCH, *(_named_node(node, task, task_parts) for node in (earlier, later)),
                      Render.search(earlier))
        self.record.write(event='same_search', turn=turn, parts=[earlier.id, later.id], check=check)
        return check

    def _ask_fact(self, turn: int, request: List[Tuple[int, UserText]], decompositions: List[Planner.Decomposition],
                  template: str, found: List[Tuple[str, str]]) -> Response:
        self.misfits.append(template)
        trees = [tree for tree in _other_trees(decompositions, self.misfits)
                 if not isinstance(tree, Planner.TemplateChoice)]
        parts = decompositions[0].model_copy(update={'tree': trees[0], 'readings': trees[1:]}) if trees else None
        fact = ' '.join(fact for _, fact in found)
        if self.questions.capped(turn, 'fact'):
            return self._from_parts(turn, request, parts, fact)
        decision = self.questions.new_id()
        return self._open(turn, Question(decision, FACT.format(fact), [
            Option(f'{decision}.o1', FROM_PARTS, plan=parts, value=fact), Option(f'{decision}.o2', REPHRASE)],
            'fact', ('request', self.request_turn), turn, request=request, decompositions=decompositions))

    def _from_parts(self, turn: int, request: List[Tuple[int, UserText]],
                    parts: Optional[Planner.Decomposition], fact: str) -> Response:
        if parts:
            return self._fill_all(turn, request, [parts])
        return self._replanned(turn, request, [NOT_THE_TASK.format(fact)])

    def _misfit(self, turn: int, request: List[Tuple[int, UserText]],
                decompositions: List[Planner.Decomposition]) -> Response:
        template = Planner.task(decompositions[0].tree).template
        self.misfits.append(template)
        others = _other_trees(decompositions, self.misfits)
        if others:
            return self._fill_all(turn, request, [decompositions[0].model_copy(update={'tree': others[0],
                                                                                      'readings': others[1:]})])
        return self._replanned(turn, request, failed_check=Check(DOES_NOT_FIT_TASK, ', '.join(self.misfits)))

    def _replanned(self, turn: int, request: List[Tuple[int, UserText]], facts: Sequence[Union[str, Line]] = (),
                   failed_check: Optional[str] = None) -> Response:
        if self.task_repairs >= self.config.repair_rounds:
            return self._say(COULD_NOT_PLAN)
        self.task_repairs += 1
        return self._plan(turn, request, [*self.replan_answers, *facts], failed_check)

    def _fill_sides(self, turn: int, request: List[Tuple[int, UserText]], decompositions: List[Planner.Decomposition],
                    parts: List[Part]) -> None:
        tree, sides = decompositions[0].tree, Planner.single_part_sides(decompositions[0].tree)
        if not sides:
            return
        self.sides += [(part.words, Planner.filling_key(part)) for part in parts
                       if part.id in sides and not self._side(part.words)]
        for index, output in enumerate(decompositions[1:], 1):
            if Planner.structure_key(output.tree) != Planner.structure_key(tree):
                continue
            own, roles = Planner.single_part_sides(output.tree), Planner.roles(output.tree)
            start, filled = perf_counter(), []
            for part in Planner.parts(output.tree):
                if part.id not in own or self._side(part.words):
                    continue
                side = self._fill(turn, request, decompositions, part.id, part.words, roles[part.id], ask=False,
                                  side=True, others=[other.words for other in Planner.parts(output.tree)
                                                     if other.id != part.id])
                self.sides.append((part.words, Planner.filling_key(side[0])
                                   if isinstance(side, tuple) and side[0] else None))
                filled.append(part.id)
            if filled:
                self.record.write(event='sides_filled', turn=turn, sample=index, parts=filled,
                                  seconds=round(perf_counter() - start, 1))

    def _side(self, words: ModelText) -> Optional[Tuple[ModelText, Optional[Hashable]]]:
        return next((side for side in self.sides if side[0] is words), None)

    def _structure_key(self, decompositions: List[Planner.Decomposition]) -> Callable[[Planner.Tree], Hashable]:
        if not self.config.fill_sides:
            places = {id(tree): _places(output.tree, {}) for output in decompositions
                      for tree in [output.tree, *output.readings]}
            return lambda tree: Planner.structure_key(tree, _places(tree, places.get(id(tree), {})))
        sides = {}
        for output in decompositions:
            sides_of = Planner.single_part_sides(output.tree)
            own = {part.id: self._side(part.words)[1] for part in Planner.parts(output.tree)
                   if part.id in sides_of and self._side(part.words)}
            for kept, removed in zip(sides_of[::2], sides_of[1::2]):
                if kept in own and own[kept] == own.get(removed):
                    own[kept], own[removed] = (own[kept], 'kept'), (own[removed], 'removed')
            sides.update({id(tree): own for tree in [output.tree, *output.readings]})
        return lambda tree: Planner.structure_key(tree, sides.get(id(tree)))

    def _run_new(self, turn: int, plan: SearchPlan, readings: List[Planner.StoredReading]) -> Response:
        self.readings = readings
        self.record.write(event='readings', turn=turn, readings=self.readings)
        self.plan = plan.model_copy(update={'version': self.version + 1, 'root': self._ruled(plan.root, plan.k.value)})
        verdict = self._verify(turn, self.plan)
        return self._repaired(turn, self.plan, verdict) or self._run(turn, verdict=verdict)

    def _fill(self, turn: int, request: List[Tuple[int, UserText]], decompositions: List[Planner.Decomposition],
              part_id: str, words: ModelText, role: Planner.Role, answers: Sequence[Answer] = (), ask: bool = True,
              seeker: Optional[Seeker] = None, failed_check: Optional[str] = None,
              mapped: bool = False, taken: Collection[str] = (), way_out: bool = False,
              others: Sequence[ModelText] = (), side: bool = False, findings: Sequence[Union[str, Line]] = (),
              beside: Sequence[Part] = ()) -> Union[Response, Planner.NoDifferentPlan, Tuple]:
        locked = {} if mapped or side else self._locks(part_id)
        seeker = seeker or locked.get('seeker')
        facts = [f'Set by the user, keep it: {part_id}.{name} = {value!r}' for name, value in locked.items()]
        if mapped:
            facts = [Line('Part {} of the plan shown: {!r}', part.id, part.words) for part in self.plan.parts()]
            facts += self._dropped_facts()
            facts += [f'Set by the user, keep it: {name} = {value!r}' for name, value in self.locked.items()
                      if '.' in name]
        answered = [answer[0].split('.', 1)[1] for answer in answers]
        facts += [Line(ANSWERED, answer[1], answer[2]) for answer in answers]
        facts += [Line(FINDING, line) for line in findings]
        if seeker:
            facts.append(f'The seeker is fixed: {seeker}.')
        output_type = Planner.filling_type(self.table, [*locked, *answered], seeker, mapped)
        if output_type is None:
            return None, [], []
        abstaining_type = Planner.filling_type(self.table, [*locked, *answered], seeker, mapped, True) \
            if way_out else None
        failed, original, originals, wrong = failed_check, None, None, []
        for _ in range(self.config.repair_rounds + 1):
            offered = way_out and failed is failed_check
            samples, fillings = self._step('filler', turn, Planner.filler,
                                           Prompts.filler_view(request, None if mapped else part_id, words, self.table,
                                                               failed, facts, Prompts.filler_examples(role),
                                                               self._not_this(), offered, others),
                                           abstaining_type if offered else output_type, Planner.filling_key)
            if not fillings:
                first = samples[0]
                location = next((location for location in first.error_locations
                                 if len(location) > 2 and location[0] == 'fill' and location[2] in COLUMN_SLOTS), None)
                failed, wrong = first.error, wrong + samples
                continue
            if isinstance(fillings[0][0], Planner.NoDifferentPlan):
                return fillings[0][0]
            fillings = [group for group in fillings if not isinstance(group[0], Planner.NoDifferentPlan)]

            if mapped:
                outputs = [(group, output) for group in fillings for output in group]
                checks = [self._mapping_check(output, taken) for _, output in outputs]
                if all(checks):
                    self.record.write(event='mapping_failed', turn=turn, part=outputs[0][1].id, check=checks[0])
                    return original or (None, [], [])
                group, chosen = outputs[checks.index(None)]
                fillings = [[chosen, *[output for output in group if output is not chosen]],
                            *[other for other in fillings if other is not group]]
                part_id = chosen.id
                locked = self._locks(part_id)
            for name, answer in zip(answered, answers):
                if hasattr(fillings[0][0].fill, name):
                    locked[name] = self._lock(answer[0], _value(fillings[0][0].fill, name), answer[3])
            fillings = [group for group in ([output for output in (_locked(filling, locked) for filling in group)
                                             if output] for group in fillings) if group]
            if not fillings:
                location, failed = None, _locked_rows(part_id, locked)
                continue
            if fillings[0][0].fill.seeker == 'none':
                if not side:
                    self.record.write(event='cant_do', turn=turn, reason='none', part=part_id, words=words)
                return original or (None, [], [])
            if not side:
                self.fillings[part_id] = words, fillings
            part, seekers, column, failed = self._part_from(turn, part_id, words, fillings, repair=original is not None)
            same = next((other for other in beside if Planner.same_search(other, part)), None)
            if failed is None and same:
                failed = self._same_check(turn, same, part)
            if failed is None:
                return part, seekers, column
            original, originals = original or (part, seekers, column), originals or fillings
        if original:
            if not side:
                self.fillings[part_id] = words, originals
            return original
        if not ask or location is None:
            return self._say(COULD_NOT_PLAN)
        asked = f'{part_id}.{location[2]}'
        if asked in self.declined or self.questions.capped(turn, 'column', asked):
            self.declined.add(asked)
            return None, [], []
        return self._ask_column(turn, request, decompositions, part_id, *location[1:3], wrong)

    def _part_from(self, turn: int, part_id: str, words: ModelText, fillings: List[List[BaseModel]],
                   repair: bool = False
                   ) -> Tuple[Part, List[Tuple[int, Any, bool]], List[Tuple[int, Part, bool]], Optional[str]]:
        proposal = fillings[0][0]
        part = Part(id=part_id, words=words, fill=proposal.fill.model_dump(), seeker_status=proposal.seeker_status)
        outputs = [output for group in fillings for output in group]
        part, seekers = Planner.seeker_readings(part, outputs)
        part, column = Planner.column_readings(part, [output.fill for output in outputs], self.table)
        part, column, failed = self._overlapping(turn, part, column, repair)
        return part, seekers, column, failed

    def _samples(self, node: Union[Planner.Tree, SearchPlan], part_id: Optional[str] = None) -> Any:
        parts = node.parts() if isinstance(node, SearchPlan) else Planner.parts(node)
        samples = {part.id: groups for part in parts for words, groups in [self.fillings.get(part.id, (None, []))]
                   if words is part.words}
        shared = {'t': self.fillings.get('t', (None, []))[1]} if getattr(node, 'template', None) else {}
        return samples.get(part_id, []) if part_id \
            else {part.id: samples.get(part.id, []) for part in parts} | shared

    def _mapping_check(self, filling: BaseModel, taken: Collection[str]) -> Optional[str]:
        if filling.id in taken:
            return 'two parts are one part of the plan shown'
        locked = self._locks(filling.id)
        changed = next((name for name, value in locked.items() if _value(filling.fill, name) != value), None)
        return f'{filling.id}.{changed} is not {locked[changed]!r}' if changed else None

    def _overlapping(self, turn: int, part: Part, readings: List[Tuple[int, Part, bool]],
                     repair: bool = False) -> Tuple[Part, List[Tuple[int, Part, bool]], Optional[str]]:
        slot = next((slot for _, slot, _ in Compiler.missing([part], self.table, self.lake) if slot not in REQUESTED),
                    None)
        if slot is None or (getattr(part.fill, slot).status == 'certain' and not repair):
            return part, readings, None
        value = getattr(part.fill, slot).value
        ranked = [reading for _, reading, _ in sorted(readings, key=lambda reading: -reading[0])
                  if getattr(reading.fill, slot).value != value]
        taken = next((reading for reading in ranked if slot not in
                      {name for _, name, _ in Compiler.missing([reading], self.table, self.lake)}), None)
        self.record.write(event='no_overlap', turn=turn, part=part.id, slot=slot, value=value,
                          taken=getattr(taken.fill, slot).value if taken else None)
        if taken is None:
            return part, readings, (f'No table in the lake shares values with {slot} {value!r}, nor with any of its '
                                    f'alternatives.')
        new, kept = getattr(taken.fill, slot), []
        self.swaps.append((taken, part))
        for count, reading, alternative in readings:
            if reading is taken:
                continue
            if getattr(reading.fill, slot).value == value:
                swapped = reading.model_copy(update={'fill': reading.fill.model_copy(update={slot: new})})
                self.swaps.append((swapped, reading))
                reading = swapped
            kept.append((count, reading, alternative))
        return taken, kept, None

    def _honoured(self, root: Union[Part, Combiner]) -> Union[None, Part, Combiner]:
        for part in [node for node in walk(root) if isinstance(node, Part)]:
            locked = self._locks(part.id)
            honoured = _locked(part, locked) if set(locked) <= set(type(part.fill).model_fields) else None
            if honoured is None:
                return None
            root = _replaced(root, honoured)
        return root

    def _lock(self, name: str, value: Any, source: str) -> Any:
        self.locked[name], self.lock_sources[name] = value, source
        return value

    def _locks(self, part_id: str, locked: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        return {name.split('.', 1)[1]: value for name, value in (self.locked if locked is None else locked).items()
                if name.startswith(f'{part_id}.')}

    def _ruled(self, root: Union[Part, Combiner], k: int) -> Union[Part, Combiner]:
        return Compiler.k_rule(root, k, self.config, self.table, self.lake)

    def _set_k(self, turn: int, k: int) -> Response:
        if self.plan is None:
            return self._respond(Reply(NO_PLAN))
        self._lock('k', k, CLICK)
        self.record.write(event='set_k', turn=turn, k=k)
        self.plan = self.plan.model_copy(update={'version': self.plan.version + 1, 'root': self._ruled(self.plan.root, k),
                                                 'k': Slot[PositiveInt](value=k, status='certain')})
        return self._run(turn)

    def _set_wording(self, turn: int, part_id: str, value: Union[str, Tuple[str, ...]]) -> Response:
        if self.plan is None:
            return self._respond(Reply(NO_PLAN))
        part = next((part for part in self.plan.parts() if part.id == part_id), None)
        name = next((name for name in WORDING if part and hasattr(part.fill, name)), None)
        if name is None:
            return self._respond(Reply(NO_WORDING))
        self._lock(f'{part_id}.{name}', value, CLICK)
        self.record.write(event='set_wording', turn=turn, part=part_id, value=value)
        wording = type(getattr(part.fill, name))(value=value, status='certain')
        self.plan = self.plan.model_copy(update={'version': self.plan.version + 1, 'root': _replaced(
            self.plan.root, part.model_copy(update={'fill': part.fill.model_copy(update={name: wording})}))})
        return self._run(turn)

    def _edit(self, turn: int, message: Tuple[int, UserText], targets: List[str]) -> Response:
        targets = [target for target in targets if self.plan and target in [*self.plan.edit_targets(), Router.NEW_PART]]
        if not targets:
            return self._say(NO_EDIT) if self.plan else self._respond(Reply(NO_PLAN))
        request = self.request
        if message[0] not in {number for number, _ in request}:
            self.request = [*request, message]
        response = self._revise(turn, own=True, base=self.plan) if targets == [Router.NEW_PART] \
            else self._change(turn, targets)
        if not isinstance(response.shown, PlanCard):
            self.request = request
            self.record.write(event='edit_refused', turn=turn, message=message[0])
        return response

    def _change(self, turn: int, targets: List[str]) -> Response:
        output_type = Planner.edit_type(self.plan, targets, self.table)
        facts = [f'Set by the user, keep it: {name} = {value!r}' for name, value in self.locked.items()
                 if not self._targeted(name, value, targets) and name != 'task']
        failed, original = None, None
        for _ in range(self.config.repair_rounds + 1):
            repair = original is not None
            samples, edits = self._step('edit', turn, Planner.editor,
                                        Prompts.edit_view(self.request, self.plan, targets, self.table, failed, facts,
                                                          self._not_this()),
                                        output_type, Planner.edit_key)
            checks = [] if edits else [samples[0].error]
            outputs, proposals = [], []
            for edit in (edit for group in edits for edit in group):
                try:
                    plan = _applied(self.plan, edit)
                except ValidationError as error:
                    checks.append(str(error))
                    continue
                altered = self._altered(plan, targets)
                if altered:
                    checks.append(Check(ALTERS_LOCKED, altered, self.locked[altered]))
                    continue
                found = self._task_facts(plan) or self._same_search(turn, plan.root)
                if found:
                    checks.append(found)
                    continue
                outputs.append(edit)
                kept = self._kept(plan, targets)
                if kept:
                    checks.append(Check(KEEPS_OLD, kept, self.locked[kept]))
                else:
                    proposals.append((edit, plan))
            for edit, plan in proposals:
                if Planner.plan_key(plan) == Planner.plan_key(self.plan):
                    self.record.write(event='edit_no_change', turn=turn)
                    return self._say(NO_CHANGE, UNCHANGED)
                shown, check, no_overlap = self._edited(turn, targets, edit, outputs, repair)
                if shown:
                    return shown
                original = original or ((edit, outputs) if no_overlap else None)
                checks.append(check)
            failed = checks[0]
            self.record.write(event='edit_failed', turn=turn, check=failed)
        shown = self._edited(turn, targets, *original, final=True)[0] if original else None
        return shown or self._say(NO_EDIT, failed)

    def _edited(self, turn: int, targets: List[str], edit: BaseModel, samples: List[BaseModel], repair: bool = False,
                final: bool = False) -> Tuple[Optional[Response], Optional[str], bool]:
        plan, applied = _applied(self.plan, edit), [_applied(self.plan, sample) for sample in samples]
        task = self.plan.template
        locked, undone, gone = self._relocked(plan, targets, applied)
        moved = _unshared(self.plan, plan.root, locked) if task and not plan.template \
            else {name: name for name in locked}
        locked = {name: locked[was] for name, was in moved.items()}
        found = [{node.id: node for node in walk(other.root)} for other in applied]
        if 'k' in locked:
            plan = plan.model_copy(update={'k': Slot[PositiveInt](value=locked['k'], status='certain')})
        elif any(other.k.value != plan.k.value for other in applied):
            plan = plan.model_copy(update={'k': plan.k.model_copy(update={'status': 'ambiguous'})})
        readings = []
        edited = {target.partition('.')[0] for target in targets} | {
            part_id for target in targets if target.startswith('t.') for part_id, _, _ in self.plan.places(target[2:])}
        for part in [part for part in plan.parts() if part.id in edited]:
            locks = self._locks(part.id, locked)
            versions = [version for version in (_locked(nodes[part.id], locks) for nodes in found) if version]
            honoured = _locked(part, locks)
            if honoured is None:
                return None, _locked_rows(part.id, locks), False
            part, seeker = Planner.seeker_readings(honoured, versions)
            part, column = Planner.column_readings(part, [version.fill for version in versions], self.table)
            part, column, failed = self._overlapping(turn, part, column, repair)
            if failed and not final:
                return None, failed, True
            slots = [target.partition('.')[2] for target in targets if target.startswith(f'{part.id}.')]
            column = [(count, reading, alternative) for count, reading, alternative in column
                      if part.id in targets or any(getattr(reading.fill, slot).value != getattr(part.fill, slot).value
                                                   for slot in slots)]
            plan = plan.model_copy(update={'root': _replaced(plan.root, part)})
            readings += seeker + column
        structural = []
        for combiner in [node for node in walk(plan.root) if isinstance(node, Combiner) and node.id in targets
                         and 'structure' not in locked]:
            others = Planner.tally([[nodes[combiner.id]] for nodes in found], lambda node: (node.kind, node.removal),
                                   combiner)
            structural += [(count, other.model_copy(update={'inputs': combiner.inputs}), alternative)
                           for count, other, alternative in others]
        plan = plan.model_copy(update={'structure': 'ambiguous'}) if structural else plan
        shown = self._show(turn, plan, own=True, changed=gone, edit=True, locked=locked)
        if isinstance(shown, Rejected):
            return None, Check(REPEATS, Render.not_this(shown.plan)), False
        self.lock_sources = {name: self.lock_sources[moved[name]] if value == self.locked[moved[name]] else EDITOR
                             for name, value in locked.items()}
        self.chosen, self.locked, self.undone = self._choices(locked), locked, undone
        self.readings = [stored for stored in (_reading(count, plan.root, reading, alternative)
                                               for count, reading, alternative
                                               in sorted(structural + readings, key=lambda reading: -reading[0]))
                         if not self._restores(stored.root)]
        self.record.write(event='readings', turn=turn, readings=self.readings)
        return shown, None, False

    def _task_facts(self, plan: SearchPlan) -> Optional[Check]:
        if plan.template is None:
            return None
        before, after = Templates.read(self.plan), Templates.read(plan)
        changed = [name for name, value in after.items() if value != before[name]]
        found = Templates.facts(plan.template, after, self.table, self.lake, changed) if changed else []
        return Check(TASK_FACTS, ' '.join(fact for _, fact in found)) if found else None

    def _targeted(self, name: str, value: Any, targets: List[str], root: Union[None, Part, Combiner] = None) -> bool:
        if name != 'structure':
            return name in targets or name.endswith('.seeker') and name.split('.')[0] in targets
        picked = set(Planner.structure_parts(value))
        return any(node.id in targets and sum(any(inner.id in picked for inner in walk(child))
                                              for child in node.inputs) > 1
                   for node in walk(root or self.plan.root) if isinstance(node, Combiner))

    def _kept(self, plan: SearchPlan, targets: List[str]) -> Optional[str]:
        return next((name for name, value in self.locked.items() if self._targeted(name, value, targets)
                     and not name.endswith('.seeker')
                     and _field(plan.root, plan.k.value, name, value, self.plan) == value), None)

    def _altered(self, plan: SearchPlan, targets: List[str]) -> Optional[str]:
        return next((name for name, value in self.locked.items() if not self._targeted(name, value, targets)
                     and _field(plan.root, plan.k.value, name, value, self.plan) not in (None, value)), None)

    def _relocked(self, plan: SearchPlan, targets: List[str], applied: List[SearchPlan]
                  ) -> Tuple[Dict[str, Any], Dict[str, Any], List[str]]:
        locked, gone = {}, []
        undone = {name: value for name, value in self.undone.items() if not self._targeted(name, value, targets)}
        for name, value in self.locked.items():
            new = _field(plan.root, plan.k.value, name, value, self.plan)
            if new is None and name.startswith('t.'):
                gone.append(SHARED_DROPPED.format(name))
            elif new is None and '.' in name:
                part_id, _, slot = name.partition('.')
                seeker = next(part.fill.seeker for part in plan.parts() if part.id == part_id)
                gone.append(SLOT_DROPPED.format(part_id, seeker.replace('_', ' '), slot))
            elif not self._targeted(name, value, targets):
                locked[name] = value
            elif all(_field(other.root, other.k.value, name, value, self.plan) == new for other in applied):
                locked[name] = new
            else:
                undone[name] = value
        return locked, undone, gone

    def _restores(self, root: Union[Part, Combiner]) -> Optional[str]:
        for name, value in self.undone.items():
            now = _field(root, None, name, value, self.plan)
            if now == value:
                return name
        return None

    def _not_it(self, turn: int, text: Optional[UserText] = None) -> Response:
        if self.plan is None:
            return self._respond(Reply(NO_PLAN))
        if text is None and self.exhausted is self.plan:
            return self._say(NO_OTHER_READING, EXHAUSTED)
        if len(self.rejected) >= self.config.revision_rounds:
            self.record.write(event='revision_cap', turn=turn)
            return self._say(REVISIONS_USED.format(self.config.revision_rounds))
        self.rejected.append(Rejected(self.plan, frozenset(table_id for table_id, _ in self.card.tables), self.words))
        self.record.write(event='not_it', turn=turn, version=self.plan.version, text=text)
        if text is not None:
            self.request = [*self.request, (turn, text)]
            return self._revise(turn, own=True)
        while self.readings:
            reading = self.readings.pop(0)
            if reading is self.checked:
                self.replanned.add(len(self.rejected))
            root, template = self._reading_root(turn, reading)
            root = self._honoured(root) if root else None
            if root is None:
                self.record.write(event='reading_skipped', turn=turn, reading=reading, reason='not filled')
                continue
            if self._restores(root):
                self.record.write(event='reading_skipped', turn=turn, reading=reading, reason='restores a change')
                continue
            if {node.id for node in walk(root)} & set(self.dropped_parts):
                self.record.write(event='reading_skipped', turn=turn, reading=reading, reason='dropped part')
                continue
            if Planner.repeated(root):
                self.record.write(event='reading_skipped', turn=turn, reading=reading, reason='same search twice')
                continue
            picked = self.locked.get('structure')
            if picked and _field(root, None, 'structure', picked) != picked:
                self.record.write(event='reading_skipped', turn=turn, reading=reading,
                                  reason='changes the picked structure')
                continue
            if 'task' in self.locked and _whole_task(template) != self.locked['task']:
                self.record.write(event='reading_skipped', turn=turn, reading=reading, reason='leaves the picked task')
                continue
            if self._assumes_no_overlap(root):
                self.record.write(event='reading_skipped', turn=turn, reading=reading, reason='no overlap')
                continue
            shown = self._show(turn, self.plan.model_copy(update={'root': root, **template}))
            if isinstance(shown, Response):
                return shown
            self.record.write(event='reading_skipped', turn=turn, reading=reading, reason='same tables')
        return self._revise(turn, own=False)

    def _assumes_no_overlap(self, root: Union[Part, Combiner]) -> bool:
        return bool(set(self._assumed_unshared(root)) - set(Compiler.missing(self.plan.parts(), self.table, self.lake)))

    def _revise(self, turn: int, own: bool, base: Optional[SearchPlan] = None, kept: Sequence[Part] = (),
                findings: Sequence[Finding] = ()) -> Response:
        failed, invalid, checks, reason = None, None, [], None
        for attempt in range(self.config.repair_rounds + 1):
            facts = [Line('Part {} of the plan shown: {!r}', part.id, part.words) for part in self.plan.parts()]
            facts += self._dropped_facts() + ([ADDS_PART] if base else [])
            facts += [f'Set by the user, keep it: {name} = {value!r}' for name, value in self.locked.items()]
            way_out = own and not base and failed is not None and failed.kind in (REPEATS, REPEATS_TABLES)
            samples, answers = self._step('decomposer', turn, Planner.decomposer,
                                          Prompts.decomposer_view(self.request, self.table, facts,
                                                                  examples=Prompts.decomposer_examples(),
                                                                  failed_check=invalid or failed,
                                                                  rejected=self._not_this(), way_out=way_out,
                                                                  findings=[line for _, line in findings]),
                                          Planner.FURTHER if not own else Planner.REVISION_REPAIR if way_out
                                          else None if base or findings else Planner.REVISION,
                                          Planner.decomposition_key)
            if not answers:
                failed, invalid, checks, reason = None, samples[0].error, [], INVALID
                self.record.write(event='revision_failed', turn=turn, check=invalid)
                continue
            invalid = None
            if type(answers[0][0]) in ABSTENTIONS:
                shown, checks = answers[0][0], []
            else:
                shown, checks = self._revised(turn, [answer for group in answers for answer in group
                                                     if type(answer) not in ABSTENTIONS], own, failed, base, way_out,
                                                    base.parts() if base else kept, findings)
            if type(shown) in ABSTENTIONS:
                failed, reason = None, ABSTENTIONS[type(shown)]
                break
            if shown:
                return shown
            self.record.write(event='revision_failed', turn=turn, check=next(iter(checks), None))
            repeats = not own and any(check.kind == REPEATS_TABLES for check in checks)
            failed = next((check for check in checks if own or check.kind != REPEATS_TABLES), None)
            invalid = next((sample.error for sample in samples if sample.error), None) \
                if repeats and not failed else None
            reason = failed or (INVALID if invalid else REPEATED if checks else None)
            if repeats and (attempt > 0 or not (failed or invalid)):
                failed, invalid, reason = None, None, REPEATED
                break
            if failed is None and invalid is None:
                break
        self.record.write(event='no_part_added' if base else 'no_verifier_replan' if findings
                          else 'no_different_plan' if own else 'no_other_reading', turn=turn, check=failed,
                          reason=reason)
        if not own:
            self.exhausted = self.plan
        if failed is not None and failed.kind in (DOES_NOT_FIT, NO_SEEKER):
            return self._say(CANT_EXPRESS, failed)
        if own and not base and checks and all(check.kind == BRINGS_BACK for check in checks):
            return self._say(DROPPED_PART, failed)
        return self._say(NO_EDIT if base else NO_DIFFERENT_PLAN if own else NO_OTHER_READING, reason)

    def _revised(self, turn: int, outputs: List[BaseModel], own: bool, failed: Optional[Check],
                 base: Optional[SearchPlan] = None, way_out: bool = False, kept: Sequence[Part] = (),
                 findings: Sequence[Finding] = ()
                 ) -> Tuple[Union[None, Response, Planner.NoDifferentPlan], List[Check]]:
        decompositions = outputs if own else [Planner.decomposition(output.tree) for output in outputs]
        fits = outputs[0].fit == 'fits'
        self.swaps = []
        checks, plan, readings, abstained = [Check(DOES_NOT_FIT)], None, [], None
        if fits:
            filled = self._filled(turn, self.request, decompositions, ask=False, failed_check=failed, kept=kept,
                                  way_out=way_out and isinstance(decompositions[0].tree, Planner.DecomposedPart),
                                  findings=findings)
            if isinstance(filled, Planner.NoDifferentPlan):
                checks, abstained = [], filled
            elif isinstance(filled, Check):
                checks = [filled]
            elif filled is None:
                checks = [Check(NOT_FILLED)]
            elif filled[0] is None:
                checks = [Check(NO_SEEKER)]
            else:
                checks, (plan, readings) = [], filled
        if plan is None:
            others = Planner.tally([[decomposition.tree, *decomposition.readings] for decomposition, output
                                    in zip(decompositions, outputs) if output.fit == 'fits'],
                                   Planner.structure_key, outputs[0].tree if fits else None)
            readings = [Planner.StoredReading(samples=count, root=tree, alternative=alternative)
                        for count, tree, alternative in sorted(others, key=lambda other: -other[0])]
            plan = self.plan.model_copy(update={'k': self._k(decompositions), 'structure': 'ambiguous'})
        plan = plan if own and not base else plan.model_copy(update={'k': self.plan.k})
        fallback = [reading for reading in readings if not (own and not base and reading.alternative)]
        candidates = ([] if checks or abstained else [None]) \
            + [reading for reading in fallback if isinstance(reading.root, (Part, Combiner)) and not reading.unfilled] \
            + [reading for reading in fallback if not isinstance(reading.root, (Part, Combiner))]
        structure = Planner.structure(base.root) if base else None
        not_used = self.not_used
        if own:
            self.not_used = _constraints(outputs)
        swaps = self.swaps
        for candidate in candidates:
            self.swaps = [] if candidate and not isinstance(candidate.root, (Part, Combiner)) else swaps
            found, template = self._reading_root(turn, candidate, kept, findings) if candidate else (plan.root, {})
            root = self._honoured(found) if found else None
            if root is None:
                continue
            same = self._same_search(turn, root)
            if same:
                checks.append(same)
                continue
            picked = self.locked.get('structure')
            if picked and Planner.structure(root, Planner.structure_parts(picked)) != picked:
                checks.append(Check(CHANGES_STRUCTURE, picked))
                continue
            fields = template if candidate else {'template': plan.template, 'template_parts': plan.template_parts}
            if 'task' in self.locked and _whole_task(fields) != self.locked['task']:
                checks.append(Check(CHANGES_TASK, self.locked['task']))
                continue
            back = sorted({node.id for node in walk(root)} & set(self.dropped_parts)) if not base else []
            if back:
                checks.append(Check(BRINGS_BACK, ', '.join(back)))
                continue
            if base and Planner.structure(root, [part.id for part in base.parts()]) != structure:
                checks.append(Check(KEEPS_PLAN, structure))
                continue
            restored = None if base else self._restores(root)
            if restored:
                checks.append(Check(RESTORES, restored, self.undone[restored]))
                continue
            unswapped = self._honoured(_unswapped(found, self.swaps)) if own and not base else None
            repeated = self._repeated(plan.model_copy(update={'root': self._ruled(unswapped, plan.k.value)})) \
                if unswapped else None
            if repeated:
                checks.append(Check(REPEATS, Render.not_this(repeated.plan)))
                continue
            before, self.readings = self.readings, [reading for reading in readings if reading is not candidate]
            by_user = own and not findings
            shown = self._show(turn, plan.model_copy(update={'root': root, **template}), by_user,
                               edit=base is not None)
            if isinstance(shown, Response):
                self.dropped_parts = {part_id: words for part_id, words in self.dropped_parts.items()
                                      if part_id not in {node.id for node in walk(root)}}
                self.record.write(event='readings', turn=turn, readings=self.readings)
                return shown, []
            self.readings = before
            checks.append(Check(REPEATS if by_user else REPEATS_TABLES, Render.not_this(shown.plan)))
        self.not_used = not_used
        return abstained, checks

    def _show(self, turn: int, plan: SearchPlan, own: bool = False, changed: Sequence[str] = (), edit: bool = False,
              locked: Optional[Dict[str, Any]] = None) -> Union[Response, Rejected]:
        plan = plan.model_copy(update={'version': self.plan.version + 1, 'root': self._ruled(plan.root, plan.k.value)})
        repeated = self._repeated(plan) if own else None
        if repeated:
            return repeated
        verdict = self._verify(turn, plan, self.words if edit else None, locked)
        repaired = None if edit else self._repaired(turn, plan, verdict)
        if repaired:
            return repaired
        result = self._blend_result(turn, plan)
        same = next((rejected for rejected in self.rejected
                     if rejected.tables == frozenset(table_id for table_id, _ in result.tables)), None)
        if same and not own:
            return same
        dropped = self._dropped(plan.root)
        changed = (Render.changes(self.seen, plan, dropped) if self.seen else []) + list(dropped.values()) \
            + list(changed)
        self.plan = plan
        return self._run(turn, result, changed, [SAME_TABLES] if same else [], verdict)

    def _repeated(self, plan: SearchPlan) -> Optional[Rejected]:
        return next((rejected for rejected in self.rejected
                     if Planner.plan_key(rejected.plan) == Planner.plan_key(plan)), None)

    def _dropped(self, root: Union[Part, Combiner]) -> Dict[str, Line]:
        kept = {node.id for node in walk(root)}
        return {part.id: Line(DROPPED, part.id, part.words, ', '.join(locked))
                for part in (self.seen or self.plan).parts()
                if part.id not in kept for locked in [self._locks(part.id)] if locked}

    def _dropped_facts(self) -> List[Line]:
        return [Line('Part {}, which the user dropped: {!r}', part_id, words)
                for part_id, words in self.dropped_parts.items()]

    def _not_this(self) -> List[str]:
        return list(dict.fromkeys(f'The user rejected this plan. It is NOT what they want: '
                                  f'{Render.not_this(rejected.plan)} Find a different reading of their request.'
                                  for rejected in self.rejected))

    def _reading_root(self, turn: int, reading: Planner.StoredReading, kept: Sequence[Part] = (),
                      findings: Sequence[Finding] = ()) -> Tuple[Union[None, Part, Combiner], Dict[str, Any]]:
        template = {'template': reading.template, 'template_parts': reading.template_parts}
        if reading.unfilled:
            part = next(node for node in walk(reading.root)
                        if isinstance(node, Part) and node.id == reading.unfilled.part)
            filled = self._fill(turn, self.request, [], part.id, part.words, Planner.roles(reading.root)[part.id],
                                ask=False, seeker=reading.unfilled.seeker,
                                others=[other.words for other in walk(reading.root)
                                        if isinstance(other, Part) and other.id != part.id])
            if not isinstance(filled, tuple) or filled[0] is None:
                return None, template
            return _replaced(reading.root, filled[0].model_copy(update={'seeker_status': 'ambiguous'})), template
        if isinstance(reading.root, (Part, Combiner)):
            return reading.root, template
        filled = self._filled(turn, self.request, [Planner.decomposition(reading.root)], ask=False, mapped=True,
                              kept=kept, findings=findings)
        if not isinstance(filled, tuple) or filled[0] is None:
            return None, template
        return filled[0].root, {'template': filled[0].template, 'template_parts': filled[0].template_parts}

    def _blend_result(self, turn: int, plan: SearchPlan) -> Compiler.Result:
        result = Compiler.run(plan, self.table, self.lake_config)
        if result.empty_means:
            self.record.write(event='suppressed_warnings', turn=turn, version=plan.version, warning=EMPTY_MEAN_NOTE,
                              count=result.empty_means)
        return result

    def _run(self, turn: int, result: Optional[Compiler.Result] = None, changed: Sequence[Union[str, Line]] = (),
             notices: Sequence[str] = (), verdict: Optional[Verdict] = None) -> Response:
        result = result or self._blend_result(turn, self.plan)
        self.verdict = verdict or replace(self.verdict, discrepancies=[
            found for found in self.verdict.discrepancies if not self._reopened(self.plan, found, self.locked)])
        missing = Compiler.missing(self.plan.parts(), self.table, self.lake)
        notices = list(notices) + ([NOT_CHECKED] if self.verdict.failed else []) \
            + Render.notices(self.plan, missing, result.counts, not result.tables)
        changed, self.replaced = ([REPLACED.format(self.replaced)] if self.replaced else []) + list(changed), None
        counted = {'k': self.plan.k.value} if any(found.kind == 'number_not_applied'
                                                   for found in self.verdict.discrepancies) else {}
        self.words = Render.words(self.plan, result.ranking)
        card = PlanCard(self.plan.version, result.tables, Render.words(self.plan, result.ranking, len(result.tables)),
                        result.nodes,
                        (['commands'] if self.commands_assumed else []) + self.plan.assumed(),
                        self.plan.preselected() | counted,
                        list(self.not_used), notices, ([READ_AS_EDIT] if self.read_as_edit else []) + list(changed),
                        len(self.rejected) < self.config.revision_rounds, not unranked(self.plan.root, result.ranking),
                        [Render.discrepancy(self.plan, found) for found in self.verdict.discrepancies],
                        self.plan.task and self.plan.task.replace('_', ' '))
        self.card, self.version, self.seen = card, card.version, self.plan
        self.record.write(event='plan_card', turn=turn, plan=self.plan, card=card, missing=missing,
                          counts=result.counts)
        return self._respond(card)

    def _verify(self, turn: int, plan: SearchPlan, previous: Optional[str] = None,
                locked: Optional[Dict[str, Any]] = None) -> Verdict:
        locked = self.locked if locked is None else locked
        if not self.config.verify or self.config.planner != 'staged':
            self.record.write(event='not_checked', turn=turn, version=plan.version,
                              reason=SINGLE_CALL_UNCHECKED if self.config.verify else VERIFY_OFF)
            return Verdict()
        words = Render.words(plan, Compiler.rankings(plan.root, self.table, self.lake))
        choices = self._choices(locked)
        chosen = [line for _, line, left in choices if not left]
        chosen += [f'{name} = {value!r}' for name, value in locked.items()]
        view = Prompts.verifier_view(self.request, plan, words, chosen, [line for _, line, left in choices if left],
                                     list(dict.fromkeys(rejected.words for rejected in self.rejected)), previous,
                                     self.table)
        samples, outputs = self._step('verifier', turn, Verifier.compare, view, Verifier.comparison_type(plan),
                                      lambda output: (), n=1)
        if not outputs:
            self.record.write(event='not_checked', turn=turn, version=plan.version, reason=samples[0].error)
            return Verdict(failed=True)
        found = [(discrepancy, self._why_dropped(plan, discrepancy, locked))
                 for discrepancy in outputs[0][0].discrepancies]
        self.record.write(event='compare', turn=turn, version=plan.version,
                          discrepancies=[discrepancy for discrepancy, _ in found],
                          dropped=[{'discrepancy': discrepancy, **why} for discrepancy, why in found if why])
        return Verdict([discrepancy for discrepancy, why in found if not why])

    def _why_dropped(self, plan: SearchPlan, found: BaseModel, locked: Dict[str, Any]) -> Dict[str, str]:
        name = self._reopened(plan, found, locked)
        if name:
            return {'locked': name}
        node = next((node for node in walk(plan.root) if node.id == getattr(found, 'target', None)), None)
        if found.kind == 'reversed' and isinstance(node, Combiner) and node.kind != 'difference':
            return {'no_direction': node.kind}
        return {}

    def _repaired(self, turn: int, plan: SearchPlan, verdict: Verdict) -> Optional[Response]:
        attempt = len(self.rejected)
        if not verdict.discrepancies or attempt in self.replanned:
            return None
        self.replanned.add(attempt)
        findings = [(getattr(found, 'target', '').partition('.')[0], Render.finding(plan, found))
                    for found in verdict.discrepancies]
        pointed = {node_id for node_id, _ in findings}
        before, self.plan, readings = self.plan, plan.model_copy(update={'version': plan.version - 1}), self.readings
        response = self._revise(turn, own=True, kept=[part for part in plan.parts() if part.id not in pointed],
                                findings=findings)
        entry = {'turn': turn, 'version': plan.version, 'checked': plan, 'discrepancies': verdict.discrepancies}
        if not isinstance(response.shown, PlanCard):
            self.plan = before
            self.record.write(event='verifier_replan', **entry, outcome='failed',
                              reason=response.shown.reason or response.shown.fact)
            return None
        remaining = self.verdict.discrepancies
        if remaining and Planner.plan_key(plan) != Planner.plan_key(self.plan):
            self.checked = Planner.StoredReading(samples=0, root=plan.root, template=plan.template,
                                                 template_parts=plan.template_parts)
            self.readings = [self.checked, *self.readings]
        ran = {node.id for node in walk(self.plan.root)}
        removed = [{'part': part.id, 'field': name, 'source': self.lock_sources[f'{part.id}.{name}']}
                   for part in plan.parts() if part.id not in ran for name in self._locks(part.id)]
        self.record.write(event='verifier_replan', **entry, replanned=self.plan, remaining=remaining,
                          outcome='not_checked' if self.verdict.failed else 'remaining' if remaining else 'clean',
                          readings=self.readings, replaced_readings=readings, removed_locks=removed)
        return response

    def _choices(self, locked: Dict[str, Any]) -> List[Tuple[Optional[str], Line, bool]]:
        return [(name, line, left) for name, line, left in self.chosen
                if name not in self.locked or locked.get(name) == self.locked[name]]

    def _reopened(self, plan: SearchPlan, found: BaseModel, locked: Dict[str, Any]) -> Optional[str]:
        target = getattr(found, 'target', None)
        return next((name for name, value in locked.items() if target is not None and (
            self._targeted(name, value, [target], plan.root)
            or name == 'structure' and target in Planner.structure_parts(value))), None)

    def _k(self, decompositions: List[Planner.Decomposition]) -> Slot:
        if 'k' in self.locked:
            return Slot(value=self.locked['k'], status='certain')
        proposal = decompositions[0]
        if any(decomposition.k != proposal.k for decomposition in decompositions):
            status = 'ambiguous'
        else:
            status = proposal.k_status if proposal.k else 'not_stated'
        return Slot(value=proposal.k or self.config.k_default, status=status)

    def _ask_column(self, turn: int, request: List[Tuple[int, UserText]], decompositions: List[Planner.Decomposition],
                    part_id: str, seeker: Seeker, slot: str, failed: List[Sample]) -> Response:
        values, counts, first = {}, Counter(), None
        for index, sample in enumerate(failed):
            for position, value in enumerate(_given(sample.raw, seeker, slot)):
                value = self._column(slot, value)
                if value is None:
                    continue
                key = frozenset(value) if isinstance(value, tuple) else value
                values.setdefault(key, value)
                counts[key] += 1
                first = key if index == position == 0 else first
        ranked = [values[key] for key in sorted(values, key=lambda key: (key != first, -counts[key]))][:4]
        columns = self.table.numeric_columns if slot == 'target' else self.table.columns
        ranked = ranked or [(column,) if slot == 'columns' else column for column in columns]
        text = f'Which columns should {part_id}.columns use (at least two)?' if slot == 'columns' \
            else f'Which column should {part_id}.{slot} use?'
        decision = self.questions.new_id()
        return self._open(turn, Question(decision, text, [Option(f'{decision}.o{i}', self.table.option(value),
                                                                 value=value, name=_named(value))
                                                          for i, value in enumerate(ranked, 1)],
                                         'column', ('request', self.request_turn), turn, field=f'{part_id}.{slot}',
                                         request=request, decompositions=decompositions))

    def _column(self, slot: str, value: Any) -> Any:
        try:
            return Planner.column_slot(self.table, slot)(value=value, status='certain').value
        except ValidationError:
            return None

    def _cant_do(self, turn: int, request: List[Tuple[int, UserText]], reason: str,
                 sub_plans: List[Tuple[Line, Union[Planner.Decomposition, SearchPlan], List[Planner.StoredReading]]],
                 parts: Dict[str, ModelText]) -> Response:
        decision = self.questions.new_id()
        options = [Option(f'{decision}.o{i}', label, plan=plan, readings=readings)
                   for i, (label, plan, readings) in enumerate(sub_plans[:4], 1)]
        options.append(Option(f'{decision}.o{len(options) + 1}', REPHRASE))
        return self._open(turn, Question(decision, CANT_DO[reason] + (CANT_DO_PICK if sub_plans else ''), options,
                                         'cant_do', ('request', self.request_turn), turn, request=request, parts=parts))

    def _cant_do_rest(self, turn: int, request: List[Tuple[int, UserText]], rest: Union[None, Part, Combiner],
                      readings: List[Tuple[int, Union[Part, Planner.Unfilled], bool]], k: Slot,
                      left_out: Dict[str, ModelText], template: Dict[str, Any]) -> Response:
        capped = any(field.split('.')[0] in left_out for field in self.declined) \
            or self.questions.capped(turn, 'cant_do')
        if capped:
            self.not_used += list(left_out.values())
        if rest is None:
            return self._say(COULD_NOT_PLAN) if capped else self._cant_do(turn, request, 'none', [], left_out)
        ids = {node.id for node in walk(rest)}
        kept = [_reading(samples, rest, reading, alternative, template)
                for samples, reading, alternative in sorted(readings, key=lambda item: -item[0])
                if (reading.part if isinstance(reading, Planner.Unfilled) else reading.id) in ids]
        plan = SearchPlan(version=0, k=k, root=rest, **template)
        if capped:
            return self._run_new(turn, plan, kept)
        return self._cant_do(turn, request, 'none', [(Planner.outline(rest), plan, kept)], left_out)

    def _ask(self, turn: int, text: str, choices: List[Tuple[Union[str, Line], List[BaseModel]]],
             message: Tuple[int, UserText]) -> Response:
        decision = self.questions.new_id()
        return self._open(turn, Question(decision, text, [Option(f'{decision}.o{i}', label, commands)
                                                          for i, (label, commands) in enumerate(choices, 1)],
                                         'which', ('message', message[0]), turn, message))

    def _open(self, turn: int, question: Question) -> Response:
        if self.questions.ask(turn, question):
            return self._respond(question)
        return self._respond(Reply(WAITS.format(len(self.questions.open))))

    def _respond(self, shown: Union[None, PlanCard, Question, Reply]) -> Response:
        return Response(shown, self.questions.open[::-1])

    def _say(self, fact: str, reason: Optional[str] = None) -> Response:
        return self._respond(Reply(fact, fact, reason))

    def _chat(self, turn: int, response: Response) -> Response:
        shown = response.shown
        said = list(self.to_say)
        verbatim = shown.fact if isinstance(shown, Reply) and shown.fact in VERBATIM else None
        if isinstance(shown, PlanCard):
            said.insert(0, EXPLAIN)
        elif verbatim:
            said[:0] = [(AFTER.format(verbatim, shown.reason), '')] if shown.reason else []
        elif isinstance(shown, Reply) and shown.fact:
            said.insert(0, (SAY.format(shown.fact), shown.fact))
        if not said:
            return response
        text = None
        snapshot = Chat.snapshot(self.table, self.card, self.questions.lines(), self.answered, self.locked,
                                 [rejected.words for rejected in self.rejected], self.not_used)
        for regenerated in (False, True):
            view = Prompts.chat_view([line for line, _ in said], snapshot, self.messages[-2:], self.replies,
                                     self.events, regenerated)
            _, outputs = self._step('chat', turn, Chat.chat, view, None, lambda output: output.asks_user, n=1)
            if outputs and not outputs[0][0].asks_user:
                text = outputs[0][0].text
                break
            if not outputs:
                break
        if text is None:
            templates = [template for _, template in said if template]
            text = joined(' ', templates) if templates else None
            self.record.write(event='chat_template', turn=turn, text=text or '')
        text = (verbatim or '') if text is None else Line('{} {}', verbatim, text) if verbatim else text
        self.replies.append((turn, text))
        if shown is None or isinstance(shown, Reply) and shown.fact:
            return replace(response, shown=Reply(text, shown.fact if shown else None))
        return replace(response, chat=text)

    def _snapshot(self, rounds: int) -> None:
        if self.snapshot:
            self.snapshot(Snapshot(rounds, *self.before, deepcopy(self.drawn)))

    def _copy(self) -> 'DialogueManager':
        return deepcopy(self, {id(shared): shared for shared in (self.lake, self.models, self.snapshot)})

    def _resumed(self, rounds: int, drawn: List[Drawn]) -> 'DialogueManager':
        manager = self._copy()
        manager.config = manager.questions.config = replace(self.config, question_rounds=rounds,
                                                            clarification_rounds=rounds)
        manager.record = manager.questions.record = Record(self.config.record_dir)
        manager.record.write(**self.loaded)
        manager.snapshot, manager.replay = None, deepcopy(drawn)
        return manager

    def _step(self, role: str, turn: int, agent: Agent, view: str, output_type: Any,
              key: Callable[[Any], Hashable], n: Optional[int] = None,
              step: Optional[str] = None) -> Tuple[List[Sample], List[List[Any]]]:
        settings = model_settings(self.config, role)
        marked = {'replayed': True} if self.replay else {}
        if self.replay:
            replayed, seen, samples = self.replay.pop(0)
            if (replayed, seen) != (role, view):
                raise RuntimeError(f"The resumed copy's {role} step differs from the run's {replayed} step.")
        else:
            samples = draw(agent, view, self.models[role], settings, n or self.config.samples, output_type,
                           timeout=self.config.sample_timeout,
                           instructions=Planner.SEMANTIC_PROFILE if role in PROFILED and self.table.semantics else None)
        self.drawn.append((role, view, samples))
        grouping = groups(samples, key)
        self.record.write(step=step or role, turn=turn, view=view, samples=samples, groups=grouping,
                          model=self.config.roles[role].model, settings=settings, **marked)
        return samples, [[samples[index].output for index in group] for group in grouping]


def _user(text: Union[str, UserText]) -> UserText:
    return text if isinstance(text, UserText) else UserText(text)


def _event_line(turn: int, event: Union[Message, Click, NoPreference, Other, SetK, SetWording, NotIt],
                response: Response) -> Line:
    if isinstance(event, Click):
        did = f'picked {event.option}'
    elif isinstance(event, NoPreference):
        did = f'answered {event.decision} with no preference'
    elif isinstance(event, Other):
        did = f'answered {event.decision} in their own words'
    elif isinstance(event, SetK):
        did = f'set k to {event.k}'
    elif isinstance(event, SetWording):
        did = f'set the wording of {event.part}'
    elif isinstance(event, NotIt):
        did = 'pressed Not it' + (' with a sentence' if event.text is not None else '')
    else:
        did = 'sent a message'
    shown = response.shown
    if isinstance(shown, PlanCard):
        outcome = f'plan v{shown.version} ran and found {len(shown.tables)} tables'
    elif isinstance(shown, Question):
        outcome = Line('asked {}: {}', shown.id, shown.text)
    elif shown is None:
        outcome = 'the Chat answered'
    else:
        outcome = Line('the reply was: {}', shown.fact or shown.text)
    return Line('Turn {}: the user {}; {}.', turn, did, outcome)


def _named(columns: Union[str, Tuple[str, ...]]) -> str:
    return ', '.join(columns) if isinstance(columns, tuple) else columns


def _named_node(node: Union[Part, Combiner], task: Union[None, Planner.TemplateChoice, Planner.TemplatePart],
                task_parts: Collection[str]) -> str:
    ids = [part.id for part in walk(node) if isinstance(part, Part)]
    if task and set(ids) <= set(task_parts):
        return f'the {task.template} task part'
    return node.id if isinstance(node, Part) else f'the {node.kind} of {", ".join(ids)}'


def _set_field(question: Question) -> Optional[str]:
    return 'structure' if question.kind == 'cant_do' else question.field


def _field(root: Union[Part, Combiner], k: Optional[int], name: str, value: Any, plan: Optional[SearchPlan] = None
           ) -> Any:
    if name == 'k':
        return k
    if name.startswith('t.'):
        return held(root, plan.places(name[2:]) if plan else ())
    if name == 'structure':
        return Planner.structure(root, Planner.structure_parts(value))
    part_id, _, slot = name.partition('.')
    part = next((node for node in walk(root) if isinstance(node, Part) and node.id == part_id), None)
    return _value(part.fill, slot) if part else None


def _value(fill: Any, name: str) -> Any:
    return fill.seeker if name == 'seeker' else getattr(getattr(fill, name, None), 'value', None)


def _constraints(decompositions: List[Planner.Decomposition]) -> List[ModelText]:
    listing = [decomposition for decomposition in decompositions
               if any(group.kind == 'constraint' for group in decomposition.unplaced)]
    if 2 * len(listing) <= len(decompositions):
        return []
    return [group.words for group in listing[0].unplaced if group.kind == 'constraint']


def _locked_rows(part_id: str, locked: Dict[str, Any]) -> Check:
    return Check(LOCKED_ROWS, ', '.join(f'{part_id}.{name} = {value!r}' for name, value in locked.items()))


def _readings(commands: List[BaseModel]) -> List[List[BaseModel]]:
    kinds = {command.command for command in commands}
    if 'unclear' in kinds:
        rest = [command for command in commands if command.command != 'unclear']
        return [rest + [alternative] for command in commands if command.command == 'unclear'
                for alternative in command.alternatives]
    if {'new_request', 'edit_plan'} <= kinds:
        return [[command for command in commands if command.command != dropped]
                for dropped in ('edit_plan', 'new_request')]
    return [commands]


def _one_granularity(readings: List[List[BaseModel]], locked: Dict[str, Any]) -> List[List[BaseModel]]:
    edits = [command.targets for commands in readings for command in commands if command.command == 'edit_plan']
    slots = {}
    for targets in edits:
        for part in {target.partition('.')[0] for target in targets if '.' in target}:
            slots.setdefault(part, set()).add(frozenset(target for target in targets if target.startswith(f'{part}.')))
    split = {}
    for part, (own, *others) in ((part, list(named)) for part, named in slots.items()):
        if not others and len({(part in targets, own <= set(targets)) for targets in edits
                                    if part in targets or own <= set(targets)}) > 1:
            split[part] = [part] if own & set(locked) else sorted(own)

    def resolved(command: BaseModel) -> BaseModel:
        if command.command != 'edit_plan':
            return command
        targets = [name for target in command.targets for name in split.get(target.partition('.')[0], [target])]
        return command.model_copy(update={'targets': list(dict.fromkeys(targets))})

    return [[resolved(command) for command in commands] for commands in readings]


def _given(raw: Any, seeker: Seeker, slot: str) -> List[Any]:
    fill = raw.get('fill') if isinstance(raw, dict) else None
    given = fill.get(slot) if isinstance(fill, dict) and fill.get('seeker') == seeker else None
    return [given.get('value'), *(given.get('alternatives') or [])] if isinstance(given, dict) else []


def _locked(filling: BaseModel, locked: Dict[str, Any]) -> Optional[BaseModel]:
    if filling.fill.seeker != locked.get('seeker', filling.fill.seeker):
        return None
    try:
        slots = {name: type(getattr(filling.fill, name))(value=value, status='certain') for name, value in locked.items()
                 if name != 'seeker' and hasattr(filling.fill, name)}
        fill = type(filling.fill).model_validate(filling.fill.model_copy(update=slots).model_dump())
    except ValidationError:
        return None
    update = {'fill': fill}
    if 'seeker' in locked:
        update |= {'seeker_status': 'certain'} | ({'seeker_alternatives': []}
                                                  if 'seeker_alternatives' in type(filling).model_fields else {})
    return filling.model_copy(update=update)


def _assemble(tree: Planner.Structure, parts: Iterator[Optional[Part]], ids: Iterator[str],
              task: Union[None, Part, Combiner] = None) -> Union[None, Part, Combiner]:
    if isinstance(tree, Planner.DecomposedPart):
        return next(parts)
    if isinstance(tree, (Planner.TemplateChoice, Planner.TemplatePart)):
        return task
    combiner_id = next(ids)
    inputs = [_assemble(child, parts, ids, task) for child in Planner.children(tree)]
    if tree.combiner == 'difference' and inputs[0] is None:
        return None
    inputs = [node for node in inputs if node is not None]
    if len(inputs) < 2:
        return next(iter(inputs), None)
    return Combiner(id=combiner_id, kind=tree.combiner, removal=getattr(tree, 'removal', None), inputs=tuple(inputs))


def _places(tree: Optional[Planner.Tree], sample: Dict[str, str]) -> Dict[str, str]:
    sides, places = Planner.single_part_sides(tree) if tree else [], {}
    for kept, removed in zip(sides[::2], sides[1::2]):
        both = kept in sample and removed in sample
        places |= {kept: sample[kept], removed: sample[removed]} if both else {kept: 'kept', removed: 'removed'}
    return places


def _reading(samples: int, root: Union[Part, Combiner], reading: Any, alternative: bool,
             template: Optional[Dict[str, Any]] = None) -> Planner.StoredReading:
    if isinstance(reading, Planner.Unfilled):
        return Planner.StoredReading(samples=samples, root=root, unfilled=reading, alternative=alternative,
                                     **(template or {}))
    return Planner.StoredReading(samples=samples, root=_replaced(root, reading), alternative=alternative,
                                 **((template or {}) if isinstance(reading, (Part, Combiner)) else {}))


def _other_trees(decompositions: List[Planner.Decomposition], tasks: Collection[str]) -> List[Planner.Structure]:
    fitting = [output for output in decompositions if output.fit == 'fits']
    others = Planner.tally([[output.tree, *output.readings] for output in fitting], Planner.structure_key,
                           decompositions[0].tree)
    return [tree for _, tree, _ in sorted(others, key=lambda other: -other[0])
            if getattr(Planner.task(tree), 'template', None) not in tasks]


def _renamed(node: Union[Part, Combiner], parts: int, combiners: int) -> Union[Part, Combiner]:
    if isinstance(node, Part):
        return node.model_copy(update={'id': f'p{parts + int(node.id[1:])}'})
    return node.model_copy(update={'id': f'c{combiners + int(node.id[1:])}',
                                   'inputs': tuple(_renamed(child, parts, combiners) for child in node.inputs)})


def _combiners(tree: Planner.Structure) -> int:
    if isinstance(tree, (Planner.DecomposedPart, Planner.TemplateChoice, Planner.TemplatePart)):
        return 0
    return 1 + sum(_combiners(child) for child in Planner.children(tree))


def _set_slots(output: BaseModel, locked: Dict[str, Any]) -> BaseModel:
    return output.model_copy(update={name: type(getattr(output, name))(value=value, status='certain')
                                     for name, value in locked.items() if name in type(output).model_fields})


def _whole_task(template: Dict[str, Any]) -> Optional[str]:
    return None if template.get('template_parts') else template.get('template')


def _unswapped(root: Union[Part, Combiner], swaps: List[Tuple[Part, Part]]) -> Union[Part, Combiner]:
    for node in list(walk(root)):
        root = next((_replaced(root, own) for swapped, own in swaps if swapped is node), root)
    return root


def _unshared(plan: SearchPlan, root: Union[Part, Combiner], locked: Dict[str, Any]) -> Dict[str, str]:
    moved = {name: name for name in locked if not name.startswith('t.') and name != 'task'}
    for name in [name for name in locked if name.startswith('t.')]:
        moved |= {f'{part_id}.{field}': name for part_id, field, position in plan.places(name[2:])
                  if position is None and field != 'rows_where'
                  and _field(root, None, f'{part_id}.{field}', None) is not None}
    return moved


def _applied(plan: SearchPlan, edit: BaseModel) -> SearchPlan:
    root, k = plan.root, plan.k
    for change in edit.changes:
        if isinstance(change, Planner.KChange):
            k = change.value
            continue
        node_id, _, name = change.target.partition('.')
        if node_id == 't':
            root = plan.model_copy(update={'root': root}).with_shared(name, change.value).root
            continue
        node = next(node for node in walk(root) if node.id == node_id)
        if isinstance(change, Planner.CombinerChange):
            root = _replaced(root, node.model_copy(update={'kind': change.kind, 'removal': change.removal}))
        elif isinstance(change, Planner.SlotChange):
            root = _replaced(root, Part(**node.model_dump() | {'fill': node.fill.model_dump() | {
                name: change.value.model_dump()}}))
        else:
            root = _replaced(root, Part(**node.model_dump() | {'fill': change.fill.model_dump(),
                                                               'seeker_status': change.seeker_status}))
    return still_task(plan, plan.model_copy(update={'root': root, 'k': k}))


def _replaced(root: Union[Part, Combiner], reading: Any) -> Any:
    if not isinstance(reading, (Part, Combiner)) or root.id == reading.id:
        return reading
    if isinstance(root, Part):
        return root
    return root.model_copy(update={'inputs': tuple(_replaced(node, reading) for node in root.inputs)})
