from collections import Counter
from dataclasses import dataclass
from math import log2

from src.Agent import Planner
from src.Agent.Plans import COLUMN_SLOTS, SHARED, Slot
from src.Agent.Sampling import canonical

from typing import Callable, Dict, Hashable, List, Literal, Optional, Tuple, Union
from pydantic import BaseModel
from src.Agent.Plans import SearchPlan, Status
from src.Agent.Profile import QueryTable


@dataclass
class Field:
    name: str
    kind: Literal['structure', 'seeker', 'column']
    position: Tuple[int, int]
    chosen: List[Hashable]
    statuses: List[Union[Status, Literal['alternative']]]
    threshold: str
    valid: int
    one_value: bool = False

    @property
    def part(self) -> Optional[str]:
        return None if self.kind == 'structure' else self.name.split('.')[0]

    @property
    def trigger(self) -> Optional[Literal['disagreement', 'status']]:
        if self.one_value:
            return None
        doubted = any(status != 'certain' for status in self.statuses)
        distinct = len(set(self.chosen))
        split = distinct > 1 if self.threshold == 'any_disagreement' \
            else distinct == len(self.chosen) >= 3 or distinct > 1 and doubted
        return 'disagreement' if split else 'status' if doubted else None

    @property
    def entropy(self) -> float:
        return -sum(n / len(self.chosen) * log2(n / len(self.chosen)) for n in Counter(self.chosen).values())


def fields(plan: SearchPlan, decompositions: List[Planner.Decomposition], fillings: Dict[str, List[List[BaseModel]]],
           threshold: str, table: QueryTable, structure_key: Callable[[Planner.Tree], Hashable]) -> List[Field]:
    found = [Field('structure', 'structure', (0, 0), [structure_key(output.tree) for output in decompositions],
                   ['alternative' if output.readings else 'certain' for output in decompositions], threshold,
                   len(decompositions))]
    shared = [output for group in fillings.get('t', []) for output in group]
    found += [Field(f't.{name}', 'column', (0, slot), [canonical(getattr(output, name).value) for output in shared],
                    [getattr(output, name).status for output in shared], threshold, len(shared),
                    Planner.one_value(table, 'column'))
              for slot, name in enumerate(SHARED.get(plan.template, {}), 1)]
    covered, fixed = plan.covered(), set(plan.task_ids().values())
    for index, part in enumerate(plan.parts(), 1):
        outputs = [output for group in fillings[part.id] for output in group]
        if part.id not in fixed:
            found.append(Field(f'{part.id}.seeker', 'seeker', (index, 0), [output.fill.seeker for output in outputs],
                               [output.seeker_status for output in outputs], threshold, len(outputs)))
        if getattr(part.fill, 'rows', ()):
            continue
        same = [output.fill for output in outputs if output.fill.seeker == part.fill.seeker]
        found += [Field(f'{part.id}.{name}', 'column', (index, slot), [canonical(getattr(fill, name).value)
                                                                       for fill in same],
                        [getattr(fill, name).status for fill in same], threshold, len(outputs),
                        Planner.one_value(table, name))
                  for slot, (name, value) in enumerate(part.fill, 1)
                  if isinstance(value, Slot) and name in COLUMN_SLOTS and f'{part.id}.{name}' not in covered]
    return found


def select(candidates: List[Field], n: int) -> List[Field]:
    uncertain = [field for field in candidates if field.trigger]
    structure = [field for field in uncertain if field.kind == 'structure']
    if structure:
        return structure[:n]
    chosen = []
    for field in sorted(uncertain, key=lambda field: (-field.entropy, field.kind != 'seeker', field.position)):
        if len(chosen) == n:
            break
        if field.kind == 'seeker':
            chosen = [other for other in chosen if other.part != field.part]
        elif any(other.kind == 'seeker' and other.part == field.part for other in chosen):
            continue
        chosen.append(field)
    return chosen
