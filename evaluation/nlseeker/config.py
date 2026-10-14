from dataclasses import dataclass
from itertools import product
from pathlib import Path

import yaml

from evaluation.nlseeker.datasets import DATASETS, FAMILIES, lake_source

from typing import Dict, List, Optional, Set, Tuple


NLSEEKER_KEYS = frozenset({
    'llm_api_key', 'llm_model', 'llm_temperature', 'llm_max_new_tokens',
    'llm_context_length', 'llm_concurrency', 'llm_retry_attempts', 'llm_tokenizer',
    'embedding_api_key', 'embedding_model', 'embedding_max_tokens',
    'embedding_batch_size', 'embedding_tokenizer',
})

ENDPOINT_KEYS = frozenset({'llm_base_url', 'embedding_base_url'})

TOP_LEVEL_KEYS = frozenset({'run', 'datasets', 'families', 'retrieval', 'nlseeker', 'backends'})


@dataclass(frozen=True)
class EvalConfig:
    tag: str
    questions_limit: Optional[int]
    datasets: Tuple[str, ...]
    families: Tuple[str, ...]
    k: Tuple[int, ...]
    n: Tuple[int, ...]
    alpha: Tuple[float, ...]
    rerank: Tuple[bool, ...]
    nlseeker: Dict[str, str]
    backends: Dict[str, Dict[str, str]]
    attribution_k: Tuple[int, ...] = ()
    attribution_alpha: Tuple[float, ...] = ()
    subsets: Tuple[int, ...] = ()
    repetitions: int = 1

    def combos(self) -> List[Tuple[int, int, float, bool]]:
        return list(product(self.k, self.n, self.alpha, self.rerank))

    def shared_combos(self) -> List[Tuple[int, int, float, bool]]:
        return list(product(self.attribution_k, self.n, self.attribution_alpha, (False,)))

    def nlseeker_for(self, backend: str) -> Dict[str, str]:
        endpoints = self.backends[backend]
        return {**self.nlseeker, 'llm_base_url': endpoints['llm_base_url'],
                'embedding_base_url': endpoints['embedding_base_url']}


def _as_bool(value) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f'retrieval.rerank entries must be booleans (on/off), got {value!r}')
    return value


def _as_tuple(value, caster, key: str) -> tuple:
    items = value if isinstance(value, list) else [value]
    if not items:
        raise ValueError(f'retrieval.{key} must not be empty')
    return tuple(caster(item) for item in items)


def _require_keys(section: dict, expected: frozenset, name: str) -> None:
    unknown = set(section) - expected
    missing = expected - set(section)
    if unknown:
        raise ValueError(f'unknown {name} key(s): {sorted(unknown)}')
    if missing:
        raise ValueError(f'missing {name} key(s): {sorted(missing)}')


def load_config(path: Path) -> EvalConfig:
    return parse_config(Path(path).read_text())


def parse_config(text: str) -> EvalConfig:
    doc = yaml.safe_load(text)
    attribution = doc.pop('attribution', None) or {'k': [], 'alpha': []}
    throughput = doc.pop('throughput', None) or {'subsets': [], 'repetitions': 1}
    _require_keys(doc, TOP_LEVEL_KEYS, 'top-level')
    _require_keys(attribution, frozenset({'k', 'alpha'}), 'attribution')
    _require_keys(throughput, frozenset({'subsets', 'repetitions'}), 'throughput')
    repetitions = throughput['repetitions']
    if not isinstance(repetitions, int) or repetitions < 1:
        raise ValueError(f'throughput.repetitions must be a positive int, got {repetitions!r}')
    if throughput['subsets'] and len(doc['datasets']) != 1:
        raise ValueError('throughput subsets are taken from one dataset')
    if throughput['subsets'] and attribution['k']:
        raise ValueError('a throughput run takes no attribution arms')

    run = doc['run']
    _require_keys(run, frozenset({'tag', 'questions_limit'}), 'run')
    limit = run['questions_limit']
    if limit is not None and (not isinstance(limit, int) or limit < 1):
        raise ValueError(f'run.questions_limit must be a positive int or null, got {limit!r}')

    if not isinstance(doc['datasets'], list):
        raise ValueError('datasets must be a list')
    datasets = tuple(doc['datasets'])
    for dataset in datasets:
        if dataset not in DATASETS:
            raise ValueError(f'unknown dataset {dataset!r}; expected one of {sorted(DATASETS)}')

    if not isinstance(doc['families'], list):
        raise ValueError('families must be a list')
    families = tuple(doc['families'])
    for family in families:
        if family not in FAMILIES:
            raise ValueError(f'unknown family {family!r}; expected one of {FAMILIES}')

    retrieval = doc['retrieval']
    _require_keys(retrieval, frozenset({'k', 'n', 'alpha', 'rerank'}), 'retrieval')

    _require_keys(doc['nlseeker'], NLSEEKER_KEYS, 'nlseeker')
    nlseeker = {key: str(value) for key, value in doc['nlseeker'].items()}

    if not isinstance(doc['backends'], dict) or not doc['backends']:
        raise ValueError('backends must map each backend name to its endpoints')
    for name, endpoints in doc['backends'].items():
        _require_keys(endpoints, ENDPOINT_KEYS, f'backends.{name}')
    backends = {name: {key: str(value) for key, value in endpoints.items()}
                for name, endpoints in doc['backends'].items()}

    return EvalConfig(
        tag=str(run['tag']),
        questions_limit=limit,
        datasets=datasets,
        families=families,
        k=_as_tuple(retrieval['k'], int, 'k'),
        n=_as_tuple(retrieval['n'], int, 'n'),
        alpha=_as_tuple(retrieval['alpha'], float, 'alpha'),
        rerank=_as_tuple(retrieval['rerank'], _as_bool, 'rerank'),
        nlseeker=nlseeker,
        backends=backends,
        attribution_k=tuple(map(int, attribution['k'])),
        attribution_alpha=tuple(map(float, attribution['alpha'])),
        subsets=tuple(map(int, throughput['subsets'])),
        repetitions=repetitions,
    )


def for_lake(config_text: str, lake: str) -> str:
    doc = yaml.safe_load(config_text)
    dataset, size = (lake, None) if lake in DATASETS else lake_source(lake)
    if dataset not in doc['datasets']:
        raise ValueError(f'{lake!r} is not one of the config\'s datasets {doc["datasets"]}')
    subsets = (doc.get('throughput') or {}).get('subsets') or []
    if size is not None:
        if size not in subsets:
            raise ValueError(f'{lake!r} is not one of the config\'s throughput subsets {subsets}')
        doc['throughput']['subsets'] = [size]
    doc['datasets'] = [dataset]
    doc['run']['tag'] = f"{doc['run']['tag']}-{lake}"
    return yaml.safe_dump(doc, sort_keys=False)


def parse_layout(entry: str) -> Tuple[str, dict]:
    try:
        backend, cpus, gpus = entry.split(':')
        llm_gpu, embedding_gpu = gpus.split(',')
        cores = set()
        for part in cpus.split(','):
            first, _, last = part.partition('-')
            cores.update(range(int(first), int(last or first) + 1))
    except ValueError:
        raise ValueError(f'layout entry {entry!r} is not BACKEND:CPUS:LLM_GPU,EMBEDDING_GPU') from None
    if not backend or not llm_gpu or not embedding_gpu:
        raise ValueError(f'layout entry {entry!r} is not BACKEND:CPUS:LLM_GPU,EMBEDDING_GPU')
    return backend, {'cpus': sorted(cores), 'llm_gpu': llm_gpu, 'embedding_gpu': embedding_gpu}


def check_layout(cfg: EvalConfig, layout: Dict[str, dict], backend: str, affinity: Set[int]) -> None:
    if backend not in layout:
        raise ValueError(f'the layout does not place backend {backend!r}')
    if set(layout[backend]['cpus']) != set(affinity):
        raise ValueError(f'{backend} runs on cores {sorted(affinity)}, the layout gives it {layout[backend]["cpus"]}')
    resources = {name: {'endpoint URL': set(endpoints.values())} for name, endpoints in cfg.backends.items()}
    for name, place in layout.items():
        resources[name].update({'core': set(place['cpus']), 'GPU': {place['llm_gpu'], place['embedding_gpu']}})
    names = sorted(resources)
    for i, left in enumerate(names):
        for right in names[i + 1:]:
            for kind in ('core', 'GPU', 'endpoint URL'):
                shared = resources[left].get(kind, set()) & resources[right].get(kind, set())
                if shared:
                    raise ValueError(f'{left} and {right} share {kind} {sorted(shared)}')
