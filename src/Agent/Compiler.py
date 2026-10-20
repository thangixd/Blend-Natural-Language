import warnings
from dataclasses import dataclass, field

from src.Agent.Plans import BlendNode, Combiner, Part, ranked_by, repeating_seekers, unranked, walk
from src.Agent.Sampling import canonical
from src.DBHandler import DBHandler
from src.Operators.Combiners import Counter, Difference, Intersection, Union as BlendUnion
from src.Operators.Seekers import Correlation, Keyword, MultiColumnOverlap, NaturalLanguage, SingleColumnOverlap
from src.Plan import Plan

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from src.Agent.Config import AgentConfig
from src.Agent.Plans import SearchPlan
from src.Agent.Profile import QueryTable
from src.Operators import Operator


@dataclass
class Result:
    tables: List[Tuple[int, str]]
    nodes: List[BlendNode]
    ranking: Dict[str, str] = field(default_factory=dict)
    counts: Dict[str, int] = field(default_factory=dict)
    empty_means: int = 0


EMPTY_MEAN_WARNINGS = ('Mean of empty slice', 'invalid value encountered in scalar divide')
COMBINERS = {'intersection': Intersection, 'union': BlendUnion, 'counter': Counter, 'difference': Difference}
SEEKERS = {'keyword': Keyword, 'single_column': SingleColumnOverlap, 'multi_column': MultiColumnOverlap,
           'correlation': Correlation, 'natural_language': NaturalLanguage}


def run(plan: SearchPlan, table: QueryTable, lake_config: Union[str, Path]) -> Result:
    blend, operators, nodes, ranking = build(plan, table, lake_config)
    ids = list(dict.fromkeys(int(table_id) for table_id in blend.run()))
    if unranked(plan.root, ranking):
        ids.sort()
    ids = ids[:plan.k.value]
    counted = [node.inputs[1] for node in walk(plan.root) if isinstance(node, Combiner) and node.kind == 'difference']
    if not ids and isinstance(plan.root, Combiner):
        counted += plan.parts()
        counted += [plan.root.inputs[0]] if plan.root.kind == 'difference' else []
    counts, empty_means = {}, _empty_means(operators)
    for node in {node.id: node for node in counted}.values():
        if not _contains_natural_language(node):
            counts[node.id], empty = _count(node, table, lake_config)
            empty_means += empty
    if not ids and isinstance(plan.root, Part):
        counts[plan.root.id] = 0
    return Result(list(zip(ids, filenames(blend.DB, ids))), nodes, ranking, counts, empty_means)


def _count(node: Union[Part, Combiner], table: QueryTable, lake_config: Union[str, Path]) -> Tuple[int, int]:
    blend, operators = Plan(), {}
    blend.DB.load_config(lake_config)
    _add(blend, node, node.k, table, operators, {})
    sql = operators[node.id].create_sql_query(blend.DB)
    [(found,)] = blend.DB.execute_and_fetchall(f'SELECT COUNT(DISTINCT TableId) FROM ({sql}) AS counted')
    return found, _empty_means(operators)


def _empty_means(operators: Dict[str, Operator]) -> int:
    return sum(operator.empty_means for operator in operators.values() if isinstance(operator, Correlation))


def _contains_natural_language(node: Union[Part, Combiner]) -> bool:
    return any(isinstance(inner, Part) and inner.fill.seeker == 'natural_language' for inner in walk(node))


def build(plan: SearchPlan, table: QueryTable,
          lake_config: Union[str, Path]) -> Tuple[Plan, Dict[str, Operator], List[BlendNode], Dict[str, str]]:
    blend = Plan()
    blend.DB.load_config(lake_config)
    operators, described = {}, {}
    _add(blend, plan.root, plan.k.value, table, operators, described)
    ranking = rankings(plan.root, table, blend.DB)
    root = plan.root
    return blend, operators, _nodes(root, root.k or plan.k.value, '', operators, described, ranking), ranking


def _add(blend: Plan, node: Union[Part, Combiner], k: int, table: QueryTable, operators: Dict[str, Operator],
         described: Dict[str, str]) -> None:
    if isinstance(node, Part):
        operators[node.id], described[node.id] = seeker(node, node.k or k, table)
        blend.add(node.id, operators[node.id])
        return
    inputs = _ordered(node)
    for child in inputs:
        _add(blend, child, k, table, operators, described)
    operators[node.id], described[node.id] = COMBINERS[node.kind](k=node.k or k), node.removal or ''
    blend.add(node.id, operators[node.id], [child.id for child in inputs])


def _nodes(node: Union[Part, Combiner], k: int, role: str, operators: Dict[str, Operator],
           described: Dict[str, str], ranking: Dict[str, str]) -> List[BlendNode]:
    description = ', '.join(text for text in (described[node.id], role) if text)
    if isinstance(node, Part):
        return [BlendNode(node.id, type(operators[node.id]).__name__, [], k, description)]
    inputs = _ordered(node)
    roles = {child.id: '' for child in inputs}
    first = ranked_by(node, ranking) if node.kind in ('intersection', 'difference') else None
    if node.kind == 'intersection':
        ranks = 'passes on all its tables, unranked' if unranked(first, ranking) else 'ranks the result'
        roles = {child.id: ranks if child is first else 'only filters' for child in inputs}
    elif node.kind == 'difference':
        roles = dict(zip(roles, ('kept', 'removed')))
    nodes = [inner for child in inputs
             for inner in _nodes(child, k if child is first else child.k, roles[child.id], operators, described,
                                 ranking)]
    return nodes + [BlendNode(node.id, type(operators[node.id]).__name__, list(roles), k, description)]


def _ordered(node: Combiner) -> List[Union[Part, Combiner]]:
    return sorted(node.inputs, key=_canonical) if node.kind == 'intersection' else list(node.inputs)


def _canonical(node: Union[Part, Combiner]) -> str:
    if isinstance(node, Part):
        return repr(canonical((node.fill, node.k), ('status', 'alternatives')))
    inputs = [_canonical(child) for child in node.inputs]
    return repr((node.kind, node.removal, node.k, inputs if node.kind == 'difference' else sorted(inputs)))


def rankings(root: Union[Part, Combiner], table: QueryTable, db: DBHandler) -> Dict[str, str]:
    operators = {}
    for node in walk(root):
        if isinstance(node, Part):
            operators[node.id] = seeker(node, 1, table)[0]
            operators[node.id].DB = db
        else:
            operators[node.id] = COMBINERS[node.kind]()
            operators[node.id].set_inputs([operators[child.id] for child in _ordered(node)])
    return {node.id: ranking_input(_ordered(node), operators, db) for node in walk(root)
            if isinstance(node, Combiner) and node.kind == 'intersection'}


def ranking_input(inputs: List[Union[Part, Combiner]], operators: Dict[str, Operator], db: DBHandler) -> str:
    return sorted(inputs, key=lambda node: (operators[node.id].cost(), operators[node.id].ml_cost(db)))[-1].id


def k_rule(root: Union[Part, Combiner], k: int, config: AgentConfig, table: QueryTable,
           db: DBHandler) -> Union[Part, Combiner]:
    ranking = rankings(root, table, db)
    ruled = root if isinstance(root, Part) else _inner_k(root, k, config, ranking)
    if isinstance(ruled, Combiner) and (ruled.kind in ('union', 'counter') or unranked(ruled, ranking)):
        return ruled
    return ruled.model_copy(update={'k': config.k_multiple * k if repeating_seekers(ruled, ranking) else None})


def _inner_k(node: Union[Part, Combiner], k: int, config: AgentConfig, ranking: Dict[str, str],
             parent: Optional[Combiner] = None, index: int = 0, produces_answer: bool = False,
             excluded: bool = False) -> Union[Part, Combiner]:
    produces_answer = produces_answer or parent is not None and parent.removal == 'comparison' and index == 1
    excluded = parent is not None and (parent.removal == 'exclusion' and index == 1
                                       or parent.kind == 'union' and excluded)
    multiple = k if produces_answer else config.k_multiple * k
    if isinstance(node, Part):
        if excluded and node.fill.seeker == 'keyword':
            inner = config.exclusion_cap
        elif node.fill.seeker == 'natural_language':
            inner = min(multiple, config.nl_inner_cap)
        else:
            inner = multiple
        return node.model_copy(update={'k': inner})
    answering = set()
    if produces_answer and node.kind == 'union':
        answering = {child.id for child in node.inputs}
    elif produces_answer and node.kind != 'counter':
        answering = {ranked_by(node, ranking).id}
    node = node.model_copy(update={'inputs': tuple(_inner_k(child, k, config, ranking, node, index,
                                                            child.id in answering, excluded)
                                                   for index, child in enumerate(node.inputs))})
    if produces_answer and not unranked(node, ranking):
        return node.model_copy(update={'k': k})
    if node.kind in ('union', 'counter'):
        return node.model_copy(update={'k': sum(child.k for child in node.inputs)})
    first = ranked_by(node, ranking)
    return node.model_copy(update={'k': first.k if unranked(first, ranking) else config.k_multiple * k})


def seeker(part: Part, k: int, table: QueryTable) -> Tuple[Operator, str]:
    operator = SEEKERS[part.fill.seeker](*part.fill.inputs(table), k)
    return _counting_empty_means(operator) if isinstance(operator, Correlation) else operator, part.fill.describe()


def _counting_empty_means(operator: Correlation) -> Correlation:
    operator.empty_means = 0
    create_sql_query = operator.create_sql_query

    def counted(db: DBHandler, additionals: str = '') -> str:
        with warnings.catch_warnings():
            if not len(operator.input_target):
                operator.empty_means += 1
                for message in EMPTY_MEAN_WARNINGS:
                    warnings.filterwarnings('ignore', message, RuntimeWarning)
            return create_sql_query(db, additionals)

    operator.create_sql_query = counted
    return operator


def missing(parts: List[Part], table: QueryTable, db: DBHandler) -> List[Tuple[str, str, str]]:
    checks = [(part.id, slot, name, tokens) for part in parts for slot, name, tokens in part.fill.checks(table)]
    return [(part_id, slot, name) for part_id, slot, name, tokens in checks if not _occurs(db, tokens, table.name)]


def _occurs(db: DBHandler, tokens: List[str], table_name: str) -> bool:
    tokens = [token for token in tokens if token]
    return bool(tokens) and bool(db.execute_and_fetchall(
        f'SELECT 1 FROM AllTables WHERE CellValue IN ({db.create_sql_list_str(tokens)}) AND TableId NOT IN '
        f'(SELECT TableId FROM AllTables_tables WHERE filename = ?) LIMIT 1', [table_name]))


def filenames(db: DBHandler, ids: List[int]) -> List[str]:
    if not ids:
        return []
    rows = dict(db.execute_and_fetchall(f'SELECT TableId, filename FROM AllTables_tables '
                                        f'WHERE TableId IN ({db.create_sql_list_numeric(ids)})'))
    return [rows[table_id] for table_id in ids]
