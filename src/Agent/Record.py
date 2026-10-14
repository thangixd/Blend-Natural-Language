import json
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from pydantic import BaseModel, TypeAdapter

from src.Agent.Text import Opaque


class Record:
    def __init__(self, directory: str) -> None:
        Path(directory).mkdir(parents=True, exist_ok=True)
        self.path = Path(directory) / f'{datetime.now():%Y%m%d-%H%M%S}-{uuid4().hex[:8]}.jsonl'

    def write(self, **entry) -> None:
        with self.path.open('a') as file:
            file.write(to_json({'time': datetime.now().isoformat(), **entry}) + '\n')

    def entries(self) -> list:
        return [json.loads(line) for line in self.path.read_text().splitlines()]


def to_json(value) -> str:
    return json.dumps(value, default=_plain)


_TEXT = TypeAdapter(Opaque)


def _plain(value):
    if isinstance(value, BaseModel):
        return value.model_dump(mode='json')
    if isinstance(value, Opaque):
        return _TEXT.dump_python(value, mode='json')
    return asdict(value) if is_dataclass(value) else str(value)
