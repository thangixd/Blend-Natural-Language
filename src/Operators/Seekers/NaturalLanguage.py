from src.Operators.OperatorBase import Operator
from src.Operators.Seekers.SeekerBase import Seeker

# Typing imports
from src.DBHandler import DBHandler
from typing import List, Optional

_retrievers = {}


class NaturalLanguage(Seeker):
    """Answers a natural-language question against the NL index."""

    def __init__(self, query: str, k: int = 10, n: Optional[int] = None,
                 alpha: Optional[float] = None, index_name: Optional[str] = None,
                 rerank: bool = True) -> None:
        # No regression model, see ml_cost.
        Operator.__init__(self, k)
        self.query = str(query)
        self.n = n
        self.alpha = alpha
        self.index_name = index_name
        self.rerank = rerank

    def create_sql_query(self, db: DBHandler, additionals: str = "") -> str:
        # The predicate is spliced into the retrieval query rather than the SQL below, so the
        # top-k is taken inside the allow-list instead of after the cut.
        table_ids = self._retrieve(db, additionals)
        if len(table_ids) == 0:
            return "SELECT TableId FROM AllTables WHERE 1=0"

        # Retrieval rank is the point of this seeker, so the ids must keep their order.
        return db.ranked_table_ids_to_sql(table_ids, self.k)

    def cost(self) -> float:
        return self.DB.seeker_cost('NaturalLanguage_judge' if self.rerank else 'NaturalLanguage_nojudge', 11)

    def ml_cost(self, db: DBHandler) -> float:
        """Seconds: retrieval plus, with judge, one call per pooled document (at most k * n).
        Without measured parameters: rerank * k * n, enough to order NL seekers among themselves."""
        if not db.USE_ML_OPTIMIZER:
            return 1
        n = self.n
        if n is None:
            from src.NLSeeker.Config import NLSeekerConfig
            n = NLSeekerConfig.load(db.config_path).n

        costs = db.seeker_costs or {}
        if 'NaturalLanguage_t_retrieval' not in costs or 'NaturalLanguage_t_doc' not in costs:
            return self.rerank * self.k * n
        return costs['NaturalLanguage_t_retrieval'] + self.rerank * self.k * n * costs['NaturalLanguage_t_doc']

    def _retrieve(self, db: DBHandler, additionals: str) -> List[int]:
        from dataclasses import replace
        from src.NLSeeker.Config import NLSeekerConfig

        config = NLSeekerConfig.load(db.config_path)
        if self.index_name is not None:
            config = replace(config, index_name=self.index_name)

        retriever = _load_retriever(db.config_path, config)
        return retriever.retrieve(db, self.query, k=self.k,
                                  n=config.n if self.n is None else self.n,
                                  alpha=config.alpha if self.alpha is None else self.alpha,
                                  additionals=additionals,
                                  rerank=self.rerank)


def _load_retriever(config_path, config):
    """Keeps one Retriever per index so a plan does not rebuild the inference clients per operator."""
    from src.NLSeeker.Clients import EmbeddingClient, LLMClient
    from src.NLSeeker.Retriever import Retriever

    key = (str(config_path), config.index_name)
    if key not in _retrievers:
        _retrievers[key] = Retriever(config, LLMClient(config), EmbeddingClient(config))
    return _retrievers[key]
