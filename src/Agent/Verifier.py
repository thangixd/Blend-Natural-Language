from typing import Annotated, List, Literal, Union

from pydantic import BaseModel, Field, create_model
from pydantic_ai import Agent, NativeOutput

from src.Agent.Text import ModelText

from src.Agent.Plans import SearchPlan

INSTRUCTIONS = """You check whether a table-search plan means what the user's request asks for.
Write the rationale first. Then list the requirements of the request, each in a few words, check each against the
plan, and report a discrepancy only where the plan does not do what the request asks:
- extra: a field or part of the plan that the request does not ask for; name its id.
- reversed: a field or part whose polarity or direction is the opposite of the request's (the kept and removed side
  of a difference swapped, a table left out that the request wants kept); name its id.
- number_not_applied: the request states how many tables it wants and k is another number.
- missing: something the request asks for that no field of the plan does; describe it in a few words.
Name only the ids the view lists. What the user chose is the user's decision, never a discrepancy. What the user left
to the system is no choice: check it against the request like any field the user did not set. The plans the user
rejected are readings the user does not mean: never report the plan for differing from one of them. When the plan
does what the request asks, report no discrepancy."""


class Extra(BaseModel):
    kind: Literal['extra']
    target: str


class Reversed(BaseModel):
    kind: Literal['reversed']
    target: str


class NumberNotApplied(BaseModel):
    kind: Literal['number_not_applied']
    target: Literal['k']


class Missing(BaseModel):
    kind: Literal['missing']
    description: ModelText = Field(description='What the request asks for that the plan does not do, in a few words.')


class Comparison(BaseModel):
    rationale: ModelText
    requirements: List[ModelText] = Field(description="The request's requirements, each in a few words.")
    discrepancies: List[BaseModel]


compare = Agent(name='verifier', instructions=INSTRUCTIONS, retries=0)


def comparison_type(plan: SearchPlan) -> NativeOutput:
    ids = Literal[tuple(plan.edit_targets())]
    found = (create_model('Extra', __base__=Extra, target=(ids, ...)),
             create_model('Reversed', __base__=Reversed, target=(ids, ...)), NumberNotApplied, Missing)
    return NativeOutput(create_model('Comparison', __base__=Comparison, discrepancies=(
        List[Annotated[Union[found], Field(discriminator='kind')]], ...)))
