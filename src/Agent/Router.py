from typing import Annotated, List, Literal, Union

from pydantic import BaseModel, Field, create_model, model_validator
from pydantic_ai import Agent, NativeOutput

from src.Agent.Sampling import canonical
from src.Agent.Text import Line, ModelText

from typing import Dict, Hashable

INSTRUCTIONS = """You route one message of a conversation with a table-search assistant built on Blend.
Return every command the message carries:
- new_request: the user asks for a new search of tables.
- edit_plan: the user changes the current plan; name the plan fields or parts to change, or new_part alone when
  the user adds a part to the plan (another edit_plan carries any other change). A part the user removed earlier
  comes back the same way: "include the Paris part again" or "bring back the tables that join on my cities" is
  edit_plan with new_part, never reject. A part's own id (p1) changes its kind of search or typed rows but keeps a
  column the user set; when the user wants that column changed too, as in "use a different part with another column"
  or "search p1 another way, on my postal codes", name the column as well (p1 and p1.column).
- resolve: the user answers an open question; name the question and the option they chose.
- ask_back: the user asks about an open question instead of answering it.
- reject: the user says the result shown is not what they wanted.
- chat: the user asks about the plan, the results or the system, without changing the search.
- out_of_scope: the message has nothing to do with finding tables.
- unclear: the message could mean two or more of the commands above; list them as alternatives.
Commands that need a plan or an open question are offered only while there is one.
Write the rationale first."""


NEW_PART = 'new_part'


class NewRequest(BaseModel):
    command: Literal['new_request']


class EditPlan(BaseModel):
    command: Literal['edit_plan']
    targets: List[str] = Field(min_length=1)

    @model_validator(mode='after')
    def _new_part_alone(self) -> 'EditPlan':
        if NEW_PART in self.targets and len(self.targets) > 1:
            raise ValueError(f'{NEW_PART} is a command of its own')
        return self


class Resolve(BaseModel):
    command: Literal['resolve']
    decision: str
    option: str

    @model_validator(mode='after')
    def _option_of_its_decision(self) -> 'Resolve':
        if not self.option.startswith(f'{self.decision}.'):
            raise ValueError(f'{self.option} is not an option of {self.decision}')
        return self


class AskBack(BaseModel):
    command: Literal['ask_back']
    decision: str


class Reject(BaseModel):
    command: Literal['reject']


class Chat(BaseModel):
    command: Literal['chat']
    topic: ModelText


class OutOfScope(BaseModel):
    command: Literal['out_of_scope']


class Unclear(BaseModel):
    command: Literal['unclear']
    alternatives: List[BaseModel]


class Routing(BaseModel):
    rationale: ModelText
    commands: List[BaseModel]


router = Agent(name='router', instructions=INSTRUCTIONS, retries=0)


def routing_type(decisions: Dict[str, List[str]], targets: List[str]) -> NativeOutput:
    commands = [NewRequest, Reject, Chat, OutOfScope]
    if decisions:
        ids = Literal[tuple(decisions)]
        options = Literal[tuple(option for decision in decisions.values() for option in decision)]
        commands += [create_model('Resolve', __base__=Resolve, decision=(ids, ...), option=(options, ...)),
                     create_model('AskBack', __base__=AskBack, decision=(ids, ...))]
    if targets:
        commands.append(create_model('EditPlan', __base__=EditPlan,
                                     targets=(List[Literal[tuple(targets)]], Field(min_length=1))))
    command = Annotated[Union[tuple(commands)], Field(discriminator='command')]
    unclear = create_model('Unclear', __base__=Unclear, alternatives=(List[command], Field(min_length=2)))
    return NativeOutput(create_model(
        'Routing', __base__=Routing,
        commands=(List[Annotated[Union[tuple(commands) + (unclear,)], Field(discriminator='command')]],
                  Field(min_length=1))))


def commands_key(commands: List[BaseModel]) -> Hashable:
    return tuple(sorted({canonical(command, ('topic',)) for command in commands}, key=repr))


def describe(command: BaseModel, labels: Dict[str, Union[str, Line]]) -> Union[str, Line]:
    if command.command == 'new_request':
        return 'start a new search'
    if command.command == 'edit_plan' and command.targets == [NEW_PART]:
        return 'add a part to the current plan'
    if command.command == 'edit_plan':
        return f'change {", ".join(command.targets)} of the current plan'
    if command.command == 'resolve':
        return Line('answer "{}" with {}', labels[command.decision], labels[command.option])
    if command.command == 'ask_back':
        return Line('ask about the question "{}"', labels[command.decision])
    if command.command == 'reject':
        return 'reject the result shown'
    if command.command == 'chat':
        return Line('ask about {}', command.topic)
    return 'something other than finding tables'
