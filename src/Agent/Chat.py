from pydantic import BaseModel, Field
from pydantic_ai import Agent, NativeOutput

from src.Agent.Text import Line, ModelText, joined

from typing import Any, Dict, List, Optional, Union
from src.Agent.Profile import QueryTable
from src.Agent.Render import PlanCard

INSTRUCTIONS = """You are the voice of a table-search assistant built on Blend. You phrase what the view hands you
to say, in a few plain sentences, from the facts in the view only; you never decide anything and never ask.
- Explain the plan card: what was searched, in the plan's words, and what came back; name the assumed fields and
  the notices. Take counts, table ids and the assumed fields only from the snapshot's lines for them: a table's
  number is its id, never a score, and only the "Assumed fields" line says what was assumed. Call each line by its
  label: a Changed line is a change, not a notice.
- Answer the user's question about the plan, the results or the system from the snapshot; if the snapshot does not
  say, say so. Your answer changes nothing.
- Explain an open question the user asks about, with its options; it stays open, and the user answers it by picking
  an option or typing an answer.
- Reply in one sentence to a message that is not about finding tables: you help find tables in the lake.
- Say a sentence you are handed in your own words, keeping its meaning and any next step it names.
- After a sentence the user was just told, which is shown before your words, explain the reason you are handed for
  it, without repeating the sentence.
Never ask the user anything, not even whether to do something. Set asks_user if your text asks the user anything."""
SIZES = ('A number of tables in the plan\'s words ("the N best", "up to N tables together") is the size a search '
         'runs at, never a count of tables found; only the line on the tables shown counts them.')
ASSUMED = {'structure': 'the way the parts combine', 'commands': 'what the message asks for'}

class Utterance(BaseModel):
    text: ModelText
    asks_user: bool = Field(description='Whether the text asks the user anything.')


chat = Agent(name='chat', instructions=INSTRUCTIONS, output_type=NativeOutput(Utterance), retries=0)


def snapshot(table: QueryTable, card: Optional[PlanCard], open_lines: List[Union[str, Line]], answered: List[Line],
             locked: Dict[str, Any], rejected: List[str], not_used: List[ModelText]) -> List[Union[str, Line]]:
    lines = [f"Query table (the user's own table, never a result): {table.name}, columns: {table.short_context()}"]
    if card:
        order = 'ranked by Blend, best match first' if card.ranked else \
            "unranked: the plan's result has no order, so they are sorted by table id"
        lines += [f'Current plan: v{card.version}: {card.words}', SIZES,
                  f'Tables shown on the card: {len(card.tables)}, {order}. Blend gives no scores; the number after '
                  f'each position is the table id.' if card.tables else 'Tables shown on the card: none.']
        lines += [f'  {rank}. table id {table_id}: {name}' for rank, (table_id, name) in enumerate(card.tables, 1)]
        assumed = card.assumed
        lines.append(f"Assumed fields: {len(assumed)} of the plan's fields {'were' if assumed[1:] else 'was'} "
                     f"assumed: {', '.join(map(_assumed, assumed))}." if assumed
                     else 'Assumed fields: none; no field of the plan was assumed.')
        lines += [f'Preselected: {name} = {value}' for name, value in card.preselected.items()]
        lines += [Line('Changed: {}', change) for change in card.changed]
        lines += [Line('Notice: {}', notice) for notice in card.notices]
        lines += [Line('Discrepancy: {}', line) for line in card.discrepancies]
        lines += [] if card.not_it else ['Not it is off: this request used its revisions.']
    else:
        lines.append('Current plan: none yet.')
    lines += ['Open questions:', *(open_lines or ['  (none)'])]
    lines += ['Answered questions:', *[Line('  {}', line) for line in answered]] if answered else []
    lines += ['Set by the user:', *[f'  {name} = {value!r}' for name, value in locked.items()]] if locked else []
    lines += ['Rejected by the user:', *[f'  {words}' for words in rejected]] if rejected else []
    lines += [Line('Not used: {}', joined(', ', not_used, '{!r}'))] if not_used else []
    return lines


def _assumed(name: str) -> str:
    part_id, _, slot = name.partition('.')
    meaning = ASSUMED.get(name) or (f'the kind of search of {part_id}' if slot == 'seeker'
                                    else f'the {slot} of {part_id}')
    return f'{name} ({meaning})'
