import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Dict, List, Literal, Optional

from pydantic import BaseModel, BeforeValidator, PrivateAttr, create_model, model_validator
from pydantic_ai import Agent, NativeOutput

from src.Agent import AutoDDG
from src.Agent.Record import to_json
from src.Agent.Sampling import draw_each, model_settings
from src.Agent.Text import ModelText

from pydantic_ai.models import Model
from src.Agent.Config import AgentConfig
from src.Agent.Profile import QueryTable

Temporal = Literal['none', 'year', 'quarter', 'month', 'week', 'day', 'time', 'timestamp', 'other']
Spatial = Literal['none', 'country', 'state_province', 'county', 'city', 'district', 'postal_code', 'address',
                  'coordinates', 'other']
Usage = Literal['identifier', 'category', 'name', 'measurement', 'score', 'free_text', 'other']

ATTEMPTS = 3
NUMERIC_USAGES = ('measurement', 'score')
NULL_WORDS = ('null', 'none')
FREE_TEXT = ('entity_type', 'domain')


class Semantics(BaseModel):
    """One column's semantic profile; null means the model was unsure, or the type check dropped the field."""
    temporal: Optional[Temporal]
    spatial: Optional[Spatial]
    entity_type: Optional[ModelText]
    domain: Optional[ModelText]
    usage: Optional[Usage]
    _null_strings: int = PrivateAttr(0)

    @model_validator(mode='wrap')
    @classmethod
    def _null_words(cls, data, handler):
        words = [name for name in FREE_TEXT
                 if isinstance(data.get(name), str) and data[name].strip().lower() in NULL_WORDS]
        semantics = handler(data | dict.fromkeys(words))
        semantics._null_strings = len(words)
        return semantics


@dataclass
class ColumnRecord:
    cached: bool
    attempts: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    thinking_tokens: int = 0
    seconds: float = 0.0
    thinking_cut: bool = False
    error: Optional[str] = None
    null_strings: int = 0
    nulls: Dict[str, str] = field(default_factory=dict)


@dataclass
class SemanticProfile:
    columns: Dict[str, Semantics]
    records: Dict[str, ColumnRecord]

    @property
    def contradictions(self) -> int:
        return sum(list(record.nulls.values()).count('contradicted') for record in self.records.values())

    @property
    def cache_hit(self) -> bool:
        return all(record.cached for record in self.records.values())

    @property
    def missing(self) -> List[str]:
        return [column for column in self.records if column not in self.columns]

    @property
    def complete(self) -> bool:
        return not self.missing


profiler = Agent(name='profiler', instructions=AutoDDG.SYSTEM_MESSAGE, retries=0)


def semantics_type(words: int) -> type:
    def cut(text):
        kept = text.split() if isinstance(text, str) else []
        return ' '.join(kept[:words]) + '…' if len(kept) > words else text

    capped = Annotated[Optional[ModelText], BeforeValidator(cut)]
    return create_model('Semantics', __base__=Semantics, entity_type=(capped, ...), domain=(capped, ...))


def request(table: QueryTable, column: str, words: int) -> str:
    line = table.profile[column]
    others = ', '.join(f'{other.name} ({other.type})' for other in table.profile.values() if other.name != column)
    return AutoDDG.USER_PROMPT.format(words=words, table=table.name, column=column,
                                      profile=f'{line.type}, {line.stats()}', values=repr(line.examples),
                                      others=others)


def profile(table: QueryTable, config: AgentConfig, model: Model) -> SemanticProfile:
    output_type = semantics_type(config.profile_words)
    role = config.roles['profiler']
    settings = model_settings(config, 'profiler')
    prompts = {column: request(table, column, config.profile_words) for column in table.columns}
    paths = {column: Path(config.profile_cache_dir) / f'{_key(prompt, output_type, config, settings)}.json'
             for column, prompt in prompts.items()}
    cached = {column: json.loads(path.read_text()) for column, path in paths.items() if path.exists()}
    called = [column for column in table.columns if column not in cached]
    drawn = draw_each(profiler, [prompts[column] for column in called], model, settings, config.profiler_concurrency,
                      NativeOutput(output_type), config.sample_timeout, ATTEMPTS)

    answers, records = {}, {}
    for column, samples in zip(called, drawn):
        last = samples[-1]
        record = ColumnRecord(cached=False, attempts=len(samples),
                              input_tokens=sum(sample.input_tokens for sample in samples),
                              output_tokens=sum(sample.output_tokens for sample in samples),
                              thinking_tokens=sum(sample.thinking_tokens for sample in samples),
                              seconds=sum(sample.seconds for sample in samples),
                              thinking_cut=role.thinking and any(
                                  sample.thinking_tokens >= role.thinking_budget - 1 for sample in samples),
                              error=last.error,
                              null_strings=last.output._null_strings if last.output is not None else 0)
        records[column] = record
        if last.output is not None:
            answers[column] = last.output
            Path(config.profile_cache_dir).mkdir(parents=True, exist_ok=True)
            paths[column].write_text(to_json({'semantics': last.output, 'record': record}))
    for column, entry in cached.items():
        answers[column] = output_type.model_validate(entry['semantics'])
        records[column] = ColumnRecord(**entry['record'] | {'cached': True})

    columns = {}
    for column in table.columns:
        if column not in answers:
            continue
        answer = answers[column]
        records[column].nulls = {name: 'unsure' for name, value in answer if value is None}
        if answer.usage in NUMERIC_USAGES and table.profile[column].type != 'numeric':
            answer = answer.model_copy(update={'usage': None})
            records[column].nulls['usage'] = 'contradicted'
        columns[column] = answer
    return SemanticProfile(columns, {column: records[column] for column in table.columns})


def _key(prompt: str, output_type: type, config: AgentConfig, settings: dict) -> str:
    sent = [AutoDDG.SYSTEM_MESSAGE, prompt, output_type.model_json_schema(), config.roles['profiler'].model, settings]
    return hashlib.sha256(json.dumps(sent, sort_keys=True).encode()).hexdigest()
