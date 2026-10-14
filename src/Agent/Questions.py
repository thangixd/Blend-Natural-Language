from dataclasses import dataclass, field

from src.Agent import Planner
from src.Agent.Plans import SearchPlan
from src.Agent.Text import Line, ModelText, UserText

from typing import Any, Callable, Collection, Dict, List, Literal, Optional, Set, Tuple, Union
from pydantic import BaseModel
from src.Agent.Config import AgentConfig
from src.Agent.Record import Record

OTHER_KINDS = ('which', 'column', 'clarification')
NONE_OF_THESE = 'none of these'


@dataclass
class Option:
    id: str
    label: Union[str, Line]
    commands: List[BaseModel] = field(default_factory=list)
    plan: Union[None, Planner.Decomposition, SearchPlan] = None
    readings: List[Planner.StoredReading] = field(default_factory=list)
    value: Any = None
    samples: Optional[int] = None
    name: Optional[str] = None


@dataclass
class Question:
    id: str
    text: Union[str, Line]
    options: List[Option]
    kind: Literal['which', 'column', 'cant_do', 'unplaced', 'clarification', 'fact']
    subject: Tuple
    turn: int
    message: Optional[Tuple[int, UserText]] = None
    request: List[Tuple[int, UserText]] = field(default_factory=list)
    decompositions: List[Planner.Decomposition] = field(default_factory=list)
    parts: Dict[str, ModelText] = field(default_factory=dict)
    raised: bool = False
    field: Optional[str] = None
    trigger: Optional[Literal['blocking', 'disagreement', 'status']] = 'blocking'
    threshold: Optional[str] = None
    valid: Optional[int] = None
    statuses: Tuple[str, ...] = ()

    def about_request(self) -> bool:
        return self.subject[0] == 'request'


class Questions:
    def __init__(self, config: AgentConfig, record: Record) -> None:
        self.config, self.record = config, record
        self.open: List[Question] = []
        self.waiting: List[Question] = []
        self._ids = 0
        self.rounds = 0
        self.clarified = 0
        self.fields: Set[str] = set()
        self.rounds_about: Dict[int, int] = {}
        self.opened = False
        self.on_round: Callable[[int], None] = lambda rounds: None

    def new_id(self) -> str:
        self._ids += 1
        return f'd{self._ids}'

    def find(self, decision: str) -> Optional[Question]:
        return next((question for question in self.open if question.id == decision), None)

    def new_event(self) -> None:
        self.opened = False

    def capped(self, turn: int, kind: str, field: Optional[str] = None) -> bool:
        asked = field or kind
        if asked in self.fields:
            self.record.write(event='asked_once', turn=turn, kind=kind, field=asked)
            return True
        if self.opened and len(self.open) < self.config.open_at_once or self.left():
            return False
        self.record.write(event='question_cap', turn=turn, kind=kind)
        return True

    def taken(self) -> int:
        return self.rounds + len(self.waiting)

    def left(self) -> bool:
        return self.taken() < self.config.question_rounds

    def clarification_room(self) -> int:
        if self.opened or self.waiting or self.clarified >= self.config.clarification_rounds or not self.left():
            return 0
        return max(self.config.open_at_once - len(self.open), 0)

    def count(self, question: Question, waited: bool = False) -> None:
        if not waited:
            self.on_round(self.taken())
        self.rounds, self.opened = self.rounds + 1, True
        self.clarified += question.kind == 'clarification'
        if question.subject[0] == 'message':
            self.rounds_about[question.subject[1]] = self.rounds_about.get(question.subject[1], 0) + 1

    def close(self, question: Question) -> None:
        self.open.remove(question)

    def ask(self, turn: int, question: Question) -> bool:
        if question.kind != 'which':
            self.fields.add(question.field or question.kind)
        if len(self.open) >= self.config.open_at_once:
            self.on_round(self.taken())
            self.waiting.append(question)
            self.record.write(event='question_waits', turn=turn, question=question)
            return False
        self._opens(question)
        self.record.write(event='question', turn=turn, question=question)
        return True

    def _opens(self, question: Question, waited: bool = False) -> None:
        if waited or not self.opened:
            self.count(question, waited)
        question.threshold = self.config.trigger_threshold
        self.open.append(question)

    def raised(self, turn: int, before: Collection[str]) -> List[Question]:
        raised = [question for question in self.open if question.id in before and not question.raised]
        for question in raised:
            question.raised = True
        while self.waiting and len(self.open) < self.config.open_at_once:
            self._opens(self.waiting.pop(0), waited=True)
            raised.append(self.open[-1])
        if raised:
            self.record.write(event='raised', turn=turn, decisions=[question.id for question in raised])
        return raised

    def new_request(self, turn: int, request_turn: int, replaced: bool) -> None:
        self.rounds, self.clarified, self.fields = self.rounds_about.get(request_turn, 0), 0, set()
        for questions in (self.open, self.waiting):
            for question in [question for question in questions
                             if question.about_request() and question.subject[1] != request_turn
                             or replaced and any(command.command == 'edit_plan' for option in question.options
                                                 for command in option.commands)]:
                questions.remove(question)
                self.record.write(event='expired', turn=turn, decision=question.id)

    def lines(self, questions: Optional[List[Question]] = None) -> List[Union[str, Line]]:
        lines = []
        for question in reversed(self.open) if questions is None else questions:
            lines += [Line('  {}: {}', question.id, question.text)] + [Line('    {}: {}', option.id, option.label)
                                                                       for option in question.options]
            lines += [f'    {question.id}.other: {NONE_OF_THESE}'] if question.kind in OTHER_KINDS else []
        return lines
