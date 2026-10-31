import argparse
import random
import time
from pathlib import Path

import pandas as pd
from xgboost import XGBRegressor

from src.DBHandler import DBHandler

from typing import Callable, List, Tuple

# NaturalLanguage has no model: its ml_cost is a formula with parameters from scripts/derive_costs.py.
TOKEN_SEEKERS = ['Keyword', 'SingleColumnOverlap', 'MultiColumnOverlap', 'Correlation']


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True,
                        help='ini file with [Database]')
    parser.add_argument('--samples', type=int, default=1000)
    parser.add_argument('--seekers', nargs='+', default=TOKEN_SEEKERS, choices=TOKEN_SEEKERS)
    parser.add_argument('--out', type=Path, default=Path('src/Operators/Seekers'))
    parser.add_argument('--seed', type=int, default=0)
    return parser.parse_args()


def measure(db: DBHandler, build: Callable, attempts: int) -> Tuple[List[list], List[float]]:
    X, y = [], []
    for _ in range(attempts):
        seeker = build()
        if seeker is None:
            continue
        features = seeker._features(db)
        start = time.perf_counter()
        seeker.run()
        y.append(time.perf_counter() - start)
        X.append(features)
    return X, y


class TableSampler:
    def __init__(self, db: DBHandler, rng: random.Random) -> None:
        self.db = db
        self.rng = rng
        self._tables = {}
        self.table_ids = [row[0] for row in db.execute_and_fetchall('SELECT DISTINCT TableId FROM AllTables ORDER BY TableId')]

    def table(self, table_id: int) -> pd.DataFrame:
        if table_id not in self._tables:
            self._tables[table_id] = self.db.get_table_from_index(table_id)
        return self._tables[table_id]

    def random_table(self) -> pd.DataFrame:
        return self.table(self.rng.choice(self.table_ids))


def build_keyword(db: DBHandler, rng: random.Random):
    from src.Operators.Seekers.Keyword import Keyword
    count = rng.randint(1, 5)
    rows = db.execute_and_fetchall(
        f"SELECT CellValue FROM AllTables WHERE CellValue <> '' USING SAMPLE {count} ROWS")
    if len(rows) == 0:
        return None
    return Keyword([row[0] for row in rows])


def build_sc(sampler: TableSampler):
    from src.Operators.Seekers.SingleColumnOverlap import SingleColumnOverlap
    df = sampler.random_table()
    column = sampler.rng.choice(list(df.columns))
    values = [v for v in df[column].dropna().tolist() if v != ''][:100]
    if len(values) == 0:
        return None
    return SingleColumnOverlap(values)


def build_mc(sampler: TableSampler):
    from src.Operators.Seekers.MultiColumnOverlap import MultiColumnOverlap
    df = sampler.random_table()
    if len(df.columns) < 2:
        return None
    columns = sampler.rng.sample(list(df.columns), sampler.rng.randint(2, min(3, len(df.columns))))
    sub = df[columns].replace('', pd.NA).dropna()
    if len(sub) < 2:
        return None
    sub = sub.sample(min(30, len(sub)), random_state=sampler.rng.randint(0, 2 ** 31))
    return MultiColumnOverlap(sub)


def build_correlation(sampler: TableSampler, numeric_pairs: list):
    from src.Operators.Seekers.Correlation import Correlation
    table_id, numeric_column = sampler.rng.choice(numeric_pairs)
    df = sampler.table(table_id)
    numeric_column = str(numeric_column)
    categorical = [c for c in df.columns if c != numeric_column]
    if numeric_column not in df.columns or len(categorical) == 0:
        return None
    column = sampler.rng.choice(categorical)
    frame = pd.DataFrame({
        'source': df[column],
        'target': pd.to_numeric(df[numeric_column], errors='coerce'),
    }).replace('', pd.NA).dropna()
    if len(frame) < 2:
        return None
    return Correlation(frame['source'].tolist(), frame['target'].tolist())


def collect_token_seeker(name: str, db: DBHandler, rng: random.Random, samples: int):
    sampler = TableSampler(db, rng)
    if name == 'Keyword':
        build = lambda: build_keyword(db, rng)
    elif name == 'SingleColumnOverlap':
        build = lambda: build_sc(sampler)
    elif name == 'MultiColumnOverlap':
        build = lambda: build_mc(sampler)
    else:
        numeric_pairs = db.execute_and_fetchall(
            'SELECT DISTINCT TableId, ColumnId FROM AllTables WHERE Quadrant IS NOT NULL')
        if len(numeric_pairs) == 0:
            raise RuntimeError('No numeric columns in the index - cannot train Correlation')
        build = lambda: build_correlation(sampler, numeric_pairs)
    return measure(db, build, samples)


def main() -> None:
    args = parse_args()

    DBHandler.USE_ML_OPTIMIZER = False
    from src.Operators.OperatorBase import Operator
    Operator.DB.load_config(args.config)
    db = Operator.DB
    rng = random.Random(args.seed)

    for name in args.seekers:
        X, y = collect_token_seeker(name, db, rng, args.samples)
        if len(y) < 10:
            raise RuntimeError(f'{name}: only {len(y)} usable samples - not enough to train')
        print(f'{name}: {len(y)} samples, fitting...')
        model = XGBRegressor()
        model.fit(X, y)
        model.save_model(args.out / f'{name}_model.json')
        print(f'{name}: saved {args.out / f"{name}_model.json"}')


if __name__ == '__main__':
    main()
