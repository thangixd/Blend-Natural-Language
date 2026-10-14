import statistics

from typing import Collection, Dict, Iterable, List, Optional, Tuple


def distinct(table_ids: Iterable[int]) -> List[int]:
    return list(dict.fromkeys(table_ids))


def answer_found(row: dict, tables: Iterable[int]) -> bool:
    return not set(row['answers']).isdisjoint(tables)


def measured_basis(medians: Dict[str, float]) -> Dict[str, int]:
    cheapest = min(medians.values())
    return {name: max(1, round(median / cheapest)) for name, median in medians.items()}


def rule_order_breaks(medians: Dict[str, float], order: List[str]) -> List[Tuple[str, str]]:
    present = [name for name in order if name in medians]
    return [(cheap, dear) for i, cheap in enumerate(present) for dear in present[i + 1:]
            if medians[cheap] > medians[dear]]


def regret(seconds: Dict[str, float]) -> Dict[str, float]:
    best = min(seconds.values())
    return {order: (value - best) / best for order, value in seconds.items()}


def fit(predicted: List[float], measured: List[float]) -> Dict[str, Optional[float]]:
    mean = statistics.mean(measured)
    total = sum((m - mean) ** 2 for m in measured)
    residual = sum((p - m) ** 2 for p, m in zip(predicted, measured))
    return {'r2': 1 - residual / total if total else None,
            'median_pct_error': statistics.median(abs(p - m) / m * 100 for p, m in zip(predicted, measured)),
            'mean_ms_error': statistics.mean(abs(p - m) for p, m in zip(predicted, measured)) * 1000}


def retention(answer: Iterable[int], reference: Collection[int]) -> float:
    if not reference:
        return 1.0
    return len(set(answer) & set(reference)) / len(set(reference))


def filtered_prefix(ranking: Iterable[int], allowed: Collection[int], k: int) -> List[int]:
    return [table_id for table_id in ranking if table_id in allowed][:k]


def normalization_costs_answers(restricted: int, filtered: int, questions: int, floor: float) -> bool:
    return (filtered - restricted) / questions > floor and 100 * (filtered - restricted) >= 3 * questions


def intersect_without_pushdown(first: List[int], second: List[int], k: int) -> List[int]:
    return filtered_prefix(distinct(first), set(second), k)


def divergence(last: List[int], first: List[int], last_on_common: List[int], nl_inner: List[int],
               sc_inner_rows: List[int], k: int) -> Dict[str, bool]:
    on_common_nl = filtered_prefix(nl_inner, set(sc_inner_rows), k)
    on_common_sc = filtered_prefix(sc_inner_rows, set(nl_inner), k)
    return {'diverged': set(last) != set(first),
            'inner cut': set(last) != set(last_on_common) or set(on_common_sc) != set(first),
            'normalization': set(last_on_common) != set(on_common_nl),
            'final cut': set(on_common_nl) != set(on_common_sc)}
