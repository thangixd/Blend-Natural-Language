from pydantic_core import core_schema

from typing import Any, Sequence
from pydantic import GetCoreSchemaHandler


class Opaque:
    def for_prompt(self) -> str:
        return self._read()

    def _read(self) -> str:
        raise NotImplementedError

    def __str__(self) -> str:
        raise TypeError(f'{type(self).__name__} is read only through for_prompt()')

    def __format__(self, spec: str) -> str:
        raise TypeError(f'{type(self).__name__} is read only through for_prompt()')

    def __repr__(self) -> str:
        return f'{type(self).__name__}(...)'

    @classmethod
    def __get_pydantic_core_schema__(cls, source: Any, handler: GetCoreSchemaHandler) -> core_schema.CoreSchema:
        return core_schema.is_instance_schema(cls, serialization=_serialization())


def _serialization() -> core_schema.SerSchema:
    return core_schema.plain_serializer_function_ser_schema(lambda text: text._read(), when_used='json')


class _Text(Opaque):
    def __init__(self, text: str) -> None:
        if not isinstance(text, str):
            raise TypeError(f'{type(self).__name__} wraps a str, not {type(text).__name__}')
        self._text = text

    def _read(self) -> str:
        return self._text

    @classmethod
    def __get_pydantic_core_schema__(cls, source: Any, handler: GetCoreSchemaHandler) -> core_schema.CoreSchema:
        wrapped = core_schema.no_info_after_validator_function(cls, core_schema.str_schema())
        return core_schema.json_or_python_schema(
            json_schema=wrapped, python_schema=core_schema.union_schema([core_schema.is_instance_schema(cls), wrapped]),
            serialization=_serialization())


class UserText(_Text):
    pass
class ModelText(_Text):
    pass
class Line(Opaque):
    def __init__(self, template: str, *texts: Any) -> None:
        self._template, self._texts = template, texts

    def _read(self) -> str:
        return self._template.format(*(text._read() if isinstance(text, Opaque) else text for text in self._texts))


def joined(separator: str, texts: Sequence[Any], form: str = '{}') -> Line:
    return Line(separator.join([form] * len(texts)), *texts)
