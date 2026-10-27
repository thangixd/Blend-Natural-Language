import configparser
from dataclasses import dataclass
from pathlib import Path

from typing import Callable, Dict, Optional, Tuple, Union

ROLES = ('router', 'decomposer', 'filler', 'edit', 'chat', 'verifier', 'profiler')
THRESHOLDS = ('any_disagreement', 'strict')
EFFORTS = ('low', 'medium', 'high', 'xhigh')
PLANNERS = ('staged', 'single_call', 'text')
DEFAULTS = {'clarification_rounds': '0', 'trigger_threshold': 'any_disagreement', 'fill_sides': 'false',
            'verify': 'false', 'semantic_profile': 'true', 'example_length': '50', 'profile_words': '4',
            'planner': 'staged', 'single_call_samples': '1'}


@dataclass(frozen=True)
class RoleConfig:
    model: str
    base_url: str
    thinking: bool
    thinking_budget: int
    temperature: float
    reasoning_effort: Optional[str] = None


@dataclass(frozen=True)
class AgentConfig:
    api_key: str
    roles: Dict[str, RoleConfig]

    samples: int
    k_default: int
    question_rounds: int
    clarification_rounds: int
    trigger_threshold: str
    fill_sides: bool
    verify: bool
    semantic_profile: bool
    planner: str
    single_call_samples: int
    open_at_once: int
    repair_rounds: int
    revision_rounds: int
    k_multiple: int
    nl_inner_cap: int
    exclusion_cap: int
    sample_timeout: int
    example_length: int
    profiler_concurrency: int
    profile_words: int
    prompt_token_limit: int

    record_dir: str
    profile_cache_dir: str

    @staticmethod
    def load(config_path: Union[str, Path]) -> 'AgentConfig':
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f'Config file not found at {config_path}')

        parser = configparser.ConfigParser()
        parser.read(config_path)
        if not parser.has_section('Agent'):
            raise KeyError(f'{config_path} has no [Agent] section')
        section = parser['Agent']

        def get(key: str, cast=str):
            if key not in section and key not in DEFAULTS:
                raise KeyError(f'[Agent] in {config_path} is missing the key {key!r}')
            try:
                return cast(section.get(key, DEFAULTS.get(key)))
            except ValueError as e:
                raise ValueError(f'[Agent] {key} in {config_path}: {e}') from None

        def boolean(value: str) -> bool:
            if value.lower() not in parser.BOOLEAN_STATES:
                raise ValueError(f'not a boolean: {value!r}')
            return parser.BOOLEAN_STATES[value.lower()]

        def one_of(allowed: Tuple[str, ...]) -> Callable[[str], str]:
            def cast(value: str) -> str:
                if value not in allowed:
                    raise ValueError(f'{value!r} is none of {", ".join(allowed)}')
                return value
            return cast

        def effort(role: str) -> Optional[str]:
            return get(f'{role}_reasoning_effort', one_of(EFFORTS)) if f'{role}_reasoning_effort' in section else None

        roles = {role: RoleConfig(model=get(f'{role}_model'), base_url=get(f'{role}_base_url'),
                                  thinking=get(f'{role}_thinking', boolean),
                                  thinking_budget=get(f'{role}_thinking_budget', int),
                                  temperature=get(f'{role}_temperature', float), reasoning_effort=effort(role))
                 for role in ROLES}

        casts = {int: int, bool: boolean, str: str}
        chosen = {'trigger_threshold': one_of(THRESHOLDS), 'planner': one_of(PLANNERS)}
        values = {name: get(name, chosen.get(name, casts[spec.type]))
                  for name, spec in AgentConfig.__dataclass_fields__.items() if name != 'roles'}
        return AgentConfig(roles=roles, **values)
