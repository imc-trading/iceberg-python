# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""End-to-end benchmark: merge() vs create_match_filter + overwrite().

Compares the full write path (filter construction + file I/O + snapshot commit)
between the new merge() implementation and the previous approach.

Usage:
    poetry run pytest tests/benchmark/test_merge_filter.py -v -s -m benchmark
"""

import gc
import itertools
import timeit
import tracemalloc
from collections.abc import Callable
from pathlib import PosixPath
from typing import Any

import pyarrow as pa
import pytest

from pyiceberg.catalog import Catalog
from pyiceberg.exceptions import NoSuchTableError
from pyiceberg.schema import Schema
from pyiceberg.table import Table
from pyiceberg.table.upsert_util import create_match_filter
from pyiceberg.types import IntegerType, NestedField, StringType
from tests.catalog.test_base import InMemoryCatalog


def _make_schema(col_cardinalities: dict[str, int]) -> Schema:
    fields = []
    for i, col in enumerate(col_cardinalities):
        field_type = IntegerType() if col == "date_id" else StringType()
        fields.append(NestedField(i + 1, col, field_type, required=True))
    fields.append(NestedField(len(col_cardinalities) + 1, "v", IntegerType(), required=True))
    return Schema(*fields)


def _build_table(col_cardinalities: dict[str, int]) -> tuple[pa.Table, list[str], Schema]:
    from pyiceberg.io.pyarrow import schema_to_pyarrow

    schema = _make_schema(col_cardinalities)
    arrow_schema = schema_to_pyarrow(schema)

    vals: list[list[Any]] = []
    for col, card in col_cardinalities.items():
        if col == "date_id":
            vals.append(list(range(20260101, 20260101 + card)))
        else:
            vals.append([f"{col}_{i}" for i in range(card)])
    combos = list(itertools.product(*vals))
    data = {col: [c[i] for c in combos] for i, col in enumerate(col_cardinalities)}
    data["v"] = list(range(len(combos)))
    return pa.table(data, schema=arrow_schema), list(col_cardinalities.keys()), schema


def _fresh_table(catalog: Catalog, name: str, schema: Schema, data: pa.Table) -> Table:
    ident = f"default.{name}"
    try:
        catalog.drop_table(ident)
    except NoSuchTableError:
        pass
    tbl = catalog.create_table(ident, schema=schema)
    tbl.append(data)
    return tbl


def _measure(fn: Callable[[], Any], runs: int = 3) -> tuple[float, int]:
    """Returns (avg_seconds, peak_memory_bytes)."""
    times = []
    peak = 0
    for _ in range(runs):
        gc.collect()
        tracemalloc.start()
        t0 = timeit.default_timer()
        fn()
        times.append(timeit.default_timer() - t0)
        _, p = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        peak = max(peak, p)
    return sum(times) / len(times), peak


def _fmt(secs: float, mem_bytes: int) -> str:
    mem = f"{mem_bytes / 1024:.0f} KB" if mem_bytes < 1048576 else f"{mem_bytes / 1048576:.1f} MB"
    return f"{secs * 1000:.0f} ms, peak {mem}"


COLS = {"date_id": 252, "account": 100}  # 25,200 target rows


@pytest.mark.benchmark
@pytest.mark.parametrize("n_source", [100, 5000])
def test_e2e_merge(n_source: int, tmp_path: PosixPath) -> None:
    """End-to-end merge(): per-column In + anti-join + single OVERWRITE snapshot."""
    target_data, join_cols, schema = _build_table(COLS)

    catalog = InMemoryCatalog("bench", warehouse=str(tmp_path))
    catalog.create_namespace("default")
    tbl = _fresh_table(catalog, f"merge_{n_source}", schema, target_data)

    source_dict = {col: target_data.column(col).to_pylist()[:n_source] for col in target_data.column_names}
    source_dict["v"] = [x + 9000 for x in source_dict["v"]]
    source = pa.table(source_dict, schema=target_data.schema)

    avg, peak = _measure(lambda: tbl.merge(source, join_cols=join_cols))
    print(f"\n  merge(): {target_data.num_rows:,} target, {n_source:,} source -> {_fmt(avg, peak)}")


@pytest.mark.benchmark
def test_e2e_overwrite_100src(tmp_path: PosixPath) -> None:
    """End-to-end overwrite() with 100 source rows."""
    target_data, join_cols, schema = _build_table(COLS)

    catalog = InMemoryCatalog("bench", warehouse=str(tmp_path))
    catalog.create_namespace("default")
    tbl = _fresh_table(catalog, "overwrite_100", schema, target_data)

    source_dict = {col: target_data.column(col).to_pylist()[:100] for col in target_data.column_names}
    source_dict["v"] = [x + 9000 for x in source_dict["v"]]
    source = pa.table(source_dict, schema=target_data.schema)

    avg, peak = _measure(lambda: (tbl.overwrite(source, overwrite_filter=create_match_filter(source, join_cols))))
    print(f"\n  overwrite(): {target_data.num_rows:,} target, 100 source -> {_fmt(avg, peak)}")


@pytest.mark.benchmark
def test_e2e_overwrite_5ksrc_filter_only(tmp_path: PosixPath) -> None:
    """At 5,000 source rows, just constructing the filter takes seconds.

    We only measure filter construction here because the full overwrite()
    with a 20,000-node expression tree causes process termination during
    manifest evaluation.
    """
    target_data, join_cols, schema = _build_table(COLS)

    source_dict = {col: target_data.column(col).to_pylist()[:5000] for col in target_data.column_names}
    source_dict["v"] = [x + 9000 for x in source_dict["v"]]
    source = pa.table(source_dict, schema=target_data.schema)

    avg, peak = _measure(lambda: create_match_filter(source, join_cols), runs=1)
    print(f"\n  create_match_filter only (no overwrite): 5,000 source -> {_fmt(avg, peak)}")
