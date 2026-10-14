import re
import Stemmer

from typing import Dict, List

STOPWORDS = frozenset({
    'a', 'an', 'and', 'are', 'as', 'at', 'be', 'but', 'by', 'for', 'if', 'in', 'into',
    'is', 'it', 'no', 'not', 'of', 'on', 'or', 'such', 'that', 'the', 'their', 'then',
    'there', 'these', 'they', 'this', 'to', 'was', 'will', 'with',
})

TOKEN_PATTERN = re.compile(r'(?u)\b\w\w+\b')

_stemmer = Stemmer.Stemmer('english')


def tokenize(text: str) -> List[str]:
    tokens = [token for token in TOKEN_PATTERN.findall(text.lower()) if token not in STOPWORDS]
    return _stemmer.stemWords(tokens)


def term_frequencies(text: str) -> Dict[str, int]:
    frequencies = {}
    for token in tokenize(text):
        frequencies[token] = frequencies.get(token, 0) + 1
    return frequencies
