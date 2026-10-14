import asyncio
import json
import random
import time
from contextlib import contextmanager
from dataclasses import dataclass, field, replace

from pydantic import BaseModel, ValidationError
from pydantic_ai import Agent, UnexpectedModelBehavior, capture_run_messages
from pydantic_ai.messages import (ModelResponse, PartDeltaEvent, PartStartEvent, TextPart, TextPartDelta, ThinkingPart,
                                  ThinkingPartDelta)
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.vllm import VLLMProvider
from pydantic_ai.result import AgentStream

from typing import Any, Callable, Hashable, Iterator, List, Literal, Optional, Tuple
from pydantic_ai.models import Model
from src.Agent.Config import AgentConfig

_RUNNER = asyncio.Runner()
Listener = Callable[[str, int, List[str], List[str]], None]
_listener: Optional[Listener] = None


@dataclass
class Sample:
    output: Any
    input_tokens: int = 0
    output_tokens: int = 0
    seconds: float = 0.0
    thinking_chars: int = 0
    error: Optional[str] = None
    error_locations: List[Tuple] = field(default_factory=list)
    raw: Any = None
    seed: Optional[int] = None
    thinking_tokens: int = 0
    phase: Optional[Literal['waiting', 'thinking', 'output']] = None
    partial_thinking: Optional[str] = None
    partial_text: Optional[str] = None
    idle_seconds: Optional[float] = None


def build_model(config: AgentConfig, role: str) -> Model:
    role_config = config.roles[role]
    return OpenAIChatModel(role_config.model,
                           provider=VLLMProvider(base_url=role_config.base_url, api_key=config.api_key))


def model_settings(config: AgentConfig, role: str) -> dict:
    role_config = config.roles[role]
    extra_body = {'chat_template_kwargs': {'enable_thinking': role_config.thinking}}
    if role_config.thinking:
        extra_body['thinking_token_budget'] = role_config.thinking_budget
    if role_config.thinking and role_config.reasoning_effort:
        extra_body['reasoning_effort'] = role_config.reasoning_effort
    return {'temperature': role_config.temperature, 'extra_body': extra_body, 'openai_continuous_usage_stats': True}


@contextmanager
def listening(listener: Listener) -> Iterator[None]:
    global _listener
    _listener = listener
    try:
        yield
    finally:
        _listener = None


async def _one(agent: Agent, prompt: str, model: Model, settings: dict, output_type: Any,
               timeout: Optional[int], instructions: Optional[str] = None, index: int = 0) -> Sample:
    start = time.perf_counter()
    stream, last = None, start
    thinking, text = [], []

    async def watch(_, events):
        nonlocal stream, last
        if isinstance(events, AgentStream):
            stream = events
        async for event in events:
            last = time.perf_counter()
            part = event.part if isinstance(event, PartStartEvent) \
                else event.delta if isinstance(event, PartDeltaEvent) else None
            if _listener is None or part is None:
                continue
            if isinstance(part, (ThinkingPart, ThinkingPartDelta)):
                thinking.append((part.content if isinstance(part, ThinkingPart) else part.content_delta) or '')
            elif isinstance(part, (TextPart, TextPartDelta)):
                text.append((part.content if isinstance(part, TextPart) else part.content_delta) or '')
            else:
                continue
            _listener(agent.name, index, thinking, text)

    try:
        async with asyncio.timeout(timeout):
            with capture_run_messages() as messages:
                result = await agent.run(prompt, model=model, model_settings=settings, output_type=output_type,
                                         instructions=instructions, event_stream_handler=watch)
    except TimeoutError:
        return _failed(_sent(stream), cut_off=True, seconds=timeout,
                       error=f'The request timed out after {timeout} s.', idle_seconds=time.perf_counter() - last)
    except UnexpectedModelBehavior as e:
        cause, causes = e, []
        while cause is not None:
            causes.append(cause)
            cause = cause.__cause__
        cause = next((cause for cause in causes if isinstance(cause, ValidationError)), causes[-1])
        locations = [tuple(error['loc']) for error in cause.errors()] if isinstance(cause, ValidationError) else []
        response = _sent(stream)
        return _failed(response, cut_off=response.finish_reason == 'length', seconds=time.perf_counter() - start,
                       error=str(cause), error_locations=locations, raw=_raw(messages))

    thinking = sum(len(part.content) for message in result.new_messages() for part in message.parts
                   if isinstance(part, ThinkingPart))
    return Sample(result.output, result.usage.input_tokens, result.usage.output_tokens, time.perf_counter() - start,
                  thinking, thinking_tokens=result.usage.details.get('reasoning_tokens', 0))


def _sent(stream: Optional[AgentStream]) -> ModelResponse:
    return stream.response if stream else ModelResponse(parts=[])


def _failed(response: ModelResponse, cut_off: bool, **fields) -> Sample:
    thinking = ''.join(part.content for part in response.parts if isinstance(part, ThinkingPart))
    usage = response.usage
    sample = Sample(None, usage.input_tokens, usage.output_tokens, thinking_chars=len(thinking),
                    thinking_tokens=usage.details.get('reasoning_tokens', 0), **fields)
    if not cut_off:
        return sample
    text = ''.join(part.content for part in response.parts if isinstance(part, TextPart))
    return replace(sample, phase='output' if text else 'thinking' if thinking else 'waiting',
                   partial_thinking=thinking, partial_text=text)


def _raw(messages: list) -> Any:
    text = ''.join(part.content for message in messages[-1:] if isinstance(message, ModelResponse)
                   for part in message.parts if isinstance(part, TextPart))
    try:
        return json.loads(text)
    except ValueError:
        return None


def draw(agent: Agent, prompt: str, model: Model, settings: dict, n: int, output_type: Any = None,
         timeout: Optional[int] = None, instructions: Optional[str] = None) -> List[Sample]:
    seeds = random.sample(range(2 ** 31), n)

    async def one(index: int, seed: int) -> Sample:
        return replace(await _one(agent, prompt, model, settings | {'seed': seed}, output_type, timeout, instructions,
                                  index), seed=seed)

    async def run_all():
        return await asyncio.gather(*[one(index, seed) for index, seed in enumerate(seeds)])

    return list(_RUNNER.run(run_all()))


def draw_each(agent: Agent, prompts: List[str], model: Model, settings: dict, cap: int, output_type: Any = None,
              timeout: Optional[int] = None, attempts: int = 1) -> List[List[Sample]]:
    limit = asyncio.Semaphore(cap)

    async def tries(index: int, prompt: str) -> List[Sample]:
        samples = []
        async with limit:
            while len(samples) < attempts and (not samples or samples[-1].output is None
                                               and samples[-1].idle_seconds is None):
                seed = random.randrange(2 ** 31)
                try:
                    samples.append(replace(await _one(agent, prompt, model, settings | {'seed': seed}, output_type,
                                                      timeout, index=index), seed=seed))
                except Exception as e:
                    samples.append(Sample(None, error=f'{type(e).__name__}: {e}', seed=seed))
                    break
        return samples

    async def run_all():
        return await asyncio.gather(*[tries(index, prompt) for index, prompt in enumerate(prompts)])

    return list(_RUNNER.run(run_all()))


def canonical(value: Any, free: Tuple[str, ...] = ('rationale',)) -> Hashable:
    if isinstance(value, BaseModel):
        value = value.model_dump()
    if isinstance(value, dict):
        return tuple(sorted((name, canonical(item, free)) for name, item in value.items() if name not in free))
    if isinstance(value, (list, tuple)):
        return tuple(sorted((canonical(item, free) for item in value), key=repr))
    return value


def groups(samples: List[Sample], key: Callable[[Any], Hashable]) -> List[List[int]]:
    grouped = {}
    for index, sample in enumerate(samples):
        if sample.output is not None:
            grouped.setdefault(key(sample.output), []).append(index)
    return sorted(grouped.values(), key=len, reverse=True)
