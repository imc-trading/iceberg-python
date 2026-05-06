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
import warnings
from pathlib import PosixPath

import pyarrow as pa
import pytest

from pyiceberg.catalog import Catalog
from pyiceberg.exceptions import NoSuchTableError
from pyiceberg.schema import Schema
from pyiceberg.table import MergeResult
from pyiceberg.table.snapshots import Operation
from pyiceberg.types import IntegerType, NestedField, StringType
from tests.catalog.test_base import InMemoryCatalog


@pytest.fixture
def catalog(tmp_path: PosixPath) -> InMemoryCatalog:
    catalog = InMemoryCatalog("test.in_memory.catalog", warehouse=tmp_path.absolute().as_posix())
    catalog.create_namespace("default")
    return catalog


def _drop(catalog: Catalog, ident: str) -> None:
    try:
        catalog.drop_table(ident)
    except NoSuchTableError:
        pass


SCHEMA = Schema(
    NestedField(1, "user_id", IntegerType(), required=True),
    NestedField(2, "name", StringType(), required=True),
    NestedField(3, "score", IntegerType(), required=True),
)

ARROW = pa.schema(
    [
        pa.field("user_id", pa.int32(), nullable=False),
        pa.field("name", pa.string(), nullable=False),
        pa.field("score", pa.int32(), nullable=False),
    ]
)


# ==================== BASIC MERGE BEHAVIOR ====================


def test_merge_replaces_matching_rows(catalog: Catalog) -> None:
    ident = "default.merge_replace"
    _drop(catalog, ident)
    tbl = catalog.create_table(ident, schema=SCHEMA)

    tbl.append(
        pa.Table.from_pylist(
            [
                {"user_id": 1, "name": "Alice", "score": 100},
                {"user_id": 2, "name": "Bob", "score": 200},
                {"user_id": 3, "name": "Charlie", "score": 300},
            ],
            schema=ARROW,
        )
    )

    tbl.merge(
        pa.Table.from_pylist(
            [
                {"user_id": 1, "name": "Alice", "score": 150},
                {"user_id": 2, "name": "Bob", "score": 250},
            ],
            schema=ARROW,
        ),
        join_cols=["user_id"],
    )

    result = tbl.scan().to_arrow().sort_by("user_id")
    assert result.num_rows == 3
    assert result.column("score").to_pylist() == [150, 250, 300]


def test_merge_inserts_new_rows(catalog: Catalog) -> None:
    ident = "default.merge_insert"
    _drop(catalog, ident)
    tbl = catalog.create_table(ident, schema=SCHEMA)

    tbl.append(
        pa.Table.from_pylist(
            [
                {"user_id": 1, "name": "Alice", "score": 100},
            ],
            schema=ARROW,
        )
    )

    tbl.merge(
        pa.Table.from_pylist(
            [
                {"user_id": 1, "name": "Alice", "score": 150},
                {"user_id": 2, "name": "Bob", "score": 200},
            ],
            schema=ARROW,
        ),
        join_cols=["user_id"],
    )

    result = tbl.scan().to_arrow().sort_by("user_id")
    assert result.num_rows == 2
    assert result.column("score").to_pylist() == [150, 200]


def test_merge_all_new_rows(catalog: Catalog) -> None:
    """Source has no overlap with target - all rows inserted."""
    ident = "default.merge_all_new"
    _drop(catalog, ident)
    tbl = catalog.create_table(ident, schema=SCHEMA)

    tbl.append(
        pa.Table.from_pylist(
            [
                {"user_id": 1, "name": "Alice", "score": 100},
            ],
            schema=ARROW,
        )
    )

    tbl.merge(
        pa.Table.from_pylist(
            [
                {"user_id": 99, "name": "New", "score": 999},
            ],
            schema=ARROW,
        ),
        join_cols=["user_id"],
    )

    result = tbl.scan().to_arrow().sort_by("user_id")
    assert result.num_rows == 2
    assert result.column("user_id").to_pylist() == [1, 99]


# ==================== UNIQUENESS: THE KEY DIFFERENCE FROM UPSERT ====================


def test_merge_allows_duplicate_source_keys(catalog: Catalog) -> None:
    """upsert() would reject this - merge() allows it."""
    ident = "default.merge_dup_src"
    _drop(catalog, ident)
    tbl = catalog.create_table(ident, schema=SCHEMA)

    tbl.append(
        pa.Table.from_pylist(
            [
                {"user_id": 1, "name": "Alice", "score": 100},
            ],
            schema=ARROW,
        )
    )

    # Two rows with same key - upsert raises ValueError
    tbl.merge(
        pa.Table.from_pylist(
            [
                {"user_id": 1, "name": "Alice V1", "score": 150},
                {"user_id": 1, "name": "Alice V2", "score": 200},
            ],
            schema=ARROW,
        ),
        join_cols=["user_id"],
    )

    result = tbl.scan().to_arrow()
    assert result.num_rows == 2
    assert set(result.column("score").to_pylist()) == {150, 200}


def test_merge_allows_duplicate_target_keys(catalog: Catalog) -> None:
    """upsert() would reject this - merge() allows it."""
    ident = "default.merge_dup_tgt"
    _drop(catalog, ident)
    tbl = catalog.create_table(ident, schema=SCHEMA)

    # Target has duplicates
    tbl.append(
        pa.Table.from_pylist(
            [
                {"user_id": 1, "name": "Alice", "score": 100},
                {"user_id": 1, "name": "Alice V2", "score": 110},
                {"user_id": 2, "name": "Bob", "score": 200},
            ],
            schema=ARROW,
        )
    )

    tbl.merge(
        pa.Table.from_pylist(
            [
                {"user_id": 1, "name": "Alice", "score": 999},
            ],
            schema=ARROW,
        ),
        join_cols=["user_id"],
    )

    result = tbl.scan().to_arrow().sort_by("user_id")
    assert result.num_rows == 2
    assert result.column("score").to_pylist() == [999, 200]


# ==================== COMPOSITE KEYS ====================


def test_merge_composite_join_cols(catalog: Catalog) -> None:
    ident = "default.merge_composite"
    _drop(catalog, ident)
    tbl = catalog.create_table(ident, schema=SCHEMA)

    tbl.append(
        pa.Table.from_pylist(
            [
                {"user_id": 1, "name": "Alice", "score": 100},
                {"user_id": 1, "name": "Bob", "score": 200},
            ],
            schema=ARROW,
        )
    )

    # Only replace (user_id=1, name="Alice")
    tbl.merge(
        pa.Table.from_pylist(
            [
                {"user_id": 1, "name": "Alice", "score": 999},
            ],
            schema=ARROW,
        ),
        join_cols=["user_id", "name"],
    )

    result = tbl.scan().to_arrow().sort_by("name")
    assert result.num_rows == 2
    assert result.column("score").to_pylist() == [999, 200]


def test_merge_three_join_cols(catalog: Catalog) -> None:
    """Three-column composite key (realistic ETL scenario)."""
    schema = Schema(
        NestedField(1, "date_id", IntegerType(), required=True),
        NestedField(2, "account", StringType(), required=True),
        NestedField(3, "security", StringType(), required=True),
        NestedField(4, "value", IntegerType(), required=True),
    )
    arrow = pa.schema(
        [
            pa.field("date_id", pa.int32(), nullable=False),
            pa.field("account", pa.string(), nullable=False),
            pa.field("security", pa.string(), nullable=False),
            pa.field("value", pa.int32(), nullable=False),
        ]
    )
    ident = "default.merge_3col"
    _drop(catalog, ident)
    tbl = catalog.create_table(ident, schema=schema)

    tbl.append(
        pa.Table.from_pylist(
            [
                {"date_id": 20260101, "account": "A", "security": "S1", "value": 100},
                {"date_id": 20260101, "account": "A", "security": "S2", "value": 200},
                {"date_id": 20260101, "account": "B", "security": "S1", "value": 300},
            ],
            schema=arrow,
        )
    )

    # Replace only (20260101, A, S1)
    tbl.merge(
        pa.Table.from_pylist(
            [
                {"date_id": 20260101, "account": "A", "security": "S1", "value": 999},
            ],
            schema=arrow,
        ),
        join_cols=["date_id", "account", "security"],
    )

    result = tbl.scan().to_arrow().sort_by([("account", "ascending"), ("security", "ascending")])
    assert result.num_rows == 3
    assert result.column("value").to_pylist() == [999, 200, 300]


# ==================== EDGE CASES ====================


def test_merge_into_empty_table(catalog: Catalog) -> None:
    ident = "default.merge_empty_tgt"
    _drop(catalog, ident)
    tbl = catalog.create_table(ident, schema=SCHEMA)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        tbl.merge(
            pa.Table.from_pylist(
                [
                    {"user_id": 1, "name": "Alice", "score": 100},
                    {"user_id": 2, "name": "Bob", "score": 200},
                ],
                schema=ARROW,
            ),
            join_cols=["user_id"],
        )

    result = tbl.scan().to_arrow().sort_by("user_id")
    assert result.num_rows == 2
    assert result.column("score").to_pylist() == [100, 200]


def test_merge_empty_source_is_noop(catalog: Catalog) -> None:
    ident = "default.merge_empty_src"
    _drop(catalog, ident)
    tbl = catalog.create_table(ident, schema=SCHEMA)

    tbl.append(
        pa.Table.from_pylist(
            [
                {"user_id": 1, "name": "Alice", "score": 100},
            ],
            schema=ARROW,
        )
    )

    tbl.merge(pa.Table.from_pylist([], schema=ARROW), join_cols=["user_id"])

    result = tbl.scan().to_arrow()
    assert result.num_rows == 1
    assert result.column("score").to_pylist() == [100]


def test_merge_idempotent(catalog: Catalog) -> None:
    """Running merge twice with the same data produces the same result."""
    ident = "default.merge_idempotent"
    _drop(catalog, ident)
    tbl = catalog.create_table(ident, schema=SCHEMA)

    tbl.append(
        pa.Table.from_pylist(
            [
                {"user_id": 1, "name": "Alice", "score": 100},
                {"user_id": 2, "name": "Bob", "score": 200},
            ],
            schema=ARROW,
        )
    )

    source = pa.Table.from_pylist(
        [
            {"user_id": 1, "name": "Alice", "score": 150},
        ],
        schema=ARROW,
    )

    tbl.merge(source, join_cols=["user_id"])
    tbl.merge(source, join_cols=["user_id"])

    result = tbl.scan().to_arrow().sort_by("user_id")
    assert result.num_rows == 2
    assert result.column("score").to_pylist() == [150, 200]


# ==================== SNAPSHOT BEHAVIOR ====================


def test_merge_produces_overwrite_snapshot(catalog: Catalog) -> None:
    """merge() should produce a single OVERWRITE snapshot, not DELETE + APPEND."""
    ident = "default.merge_snapshot"
    _drop(catalog, ident)
    tbl = catalog.create_table(ident, schema=SCHEMA)

    tbl.append(
        pa.Table.from_pylist(
            [
                {"user_id": 1, "name": "Alice", "score": 100},
            ],
            schema=ARROW,
        )
    )

    snapshot_count_before = len(tbl.snapshots())

    tbl.merge(
        pa.Table.from_pylist(
            [
                {"user_id": 1, "name": "Alice", "score": 150},
            ],
            schema=ARROW,
        ),
        join_cols=["user_id"],
    )

    snapshots_after = tbl.snapshots()
    # Should add exactly ONE snapshot (OVERWRITE), not two (DELETE + APPEND)
    assert len(snapshots_after) == snapshot_count_before + 1
    assert snapshots_after[-1].summary is not None
    assert snapshots_after[-1].summary.operation == Operation.OVERWRITE


# ==================== VALIDATION ====================


def test_merge_rejects_empty_join_cols(catalog: Catalog) -> None:
    ident = "default.merge_no_cols"
    _drop(catalog, ident)
    tbl = catalog.create_table(ident, schema=SCHEMA)

    with pytest.raises(ValueError, match="non-empty"):
        tbl.merge(
            pa.Table.from_pylist(
                [
                    {"user_id": 1, "name": "Alice", "score": 100},
                ],
                schema=ARROW,
            ),
            join_cols=[],
        )


def test_merge_rejects_missing_join_cols(catalog: Catalog) -> None:
    ident = "default.merge_bad_col"
    _drop(catalog, ident)
    tbl = catalog.create_table(ident, schema=SCHEMA)

    with pytest.raises(ValueError, match="not found in source"):
        tbl.merge(
            pa.Table.from_pylist(
                [
                    {"user_id": 1, "name": "Alice", "score": 100},
                ],
                schema=ARROW,
            ),
            join_cols=["nonexistent"],
        )


def test_merge_rejects_non_arrow_input(catalog: Catalog) -> None:
    ident = "default.merge_bad_type"
    _drop(catalog, ident)
    tbl = catalog.create_table(ident, schema=SCHEMA)

    with pytest.raises(ValueError, match="Expected PyArrow"):
        tbl.merge("not a table", join_cols=["user_id"])


# ==================== check_duplicate_keys ====================


def test_merge_check_duplicate_keys_raises(catalog: Catalog) -> None:
    """check_duplicate_keys=True raises on duplicate source keys."""
    ident = "default.merge_dup_check"
    _drop(catalog, ident)
    tbl = catalog.create_table(ident, schema=SCHEMA)

    tbl.append(
        pa.Table.from_pylist(
            [
                {"user_id": 1, "name": "Alice", "score": 100},
            ],
            schema=ARROW,
        )
    )

    with pytest.raises(ValueError, match="Duplicate rows"):
        tbl.merge(
            pa.Table.from_pylist(
                [
                    {"user_id": 1, "name": "Alice V1", "score": 150},
                    {"user_id": 1, "name": "Alice V2", "score": 200},
                ],
                schema=ARROW,
            ),
            join_cols=["user_id"],
            check_duplicate_keys=True,
        )


def test_merge_check_duplicate_keys_allows_unique(catalog: Catalog) -> None:
    """check_duplicate_keys=True passes when source keys are unique."""
    ident = "default.merge_dup_check_ok"
    _drop(catalog, ident)
    tbl = catalog.create_table(ident, schema=SCHEMA)

    tbl.append(
        pa.Table.from_pylist(
            [
                {"user_id": 1, "name": "Alice", "score": 100},
            ],
            schema=ARROW,
        )
    )

    tbl.merge(
        pa.Table.from_pylist(
            [
                {"user_id": 1, "name": "Alice", "score": 150},
                {"user_id": 2, "name": "Bob", "score": 200},
            ],
            schema=ARROW,
        ),
        join_cols=["user_id"],
        check_duplicate_keys=True,
    )

    result = tbl.scan().to_arrow().sort_by("user_id")
    assert result.num_rows == 2
    assert result.column("score").to_pylist() == [150, 200]


def test_merge_default_allows_duplicate_keys(catalog: Catalog) -> None:
    """Default (check_duplicate_keys=False) allows duplicates."""
    ident = "default.merge_default_dup"
    _drop(catalog, ident)
    tbl = catalog.create_table(ident, schema=SCHEMA)

    tbl.append(
        pa.Table.from_pylist(
            [
                {"user_id": 1, "name": "Alice", "score": 100},
            ],
            schema=ARROW,
        )
    )

    tbl.merge(
        pa.Table.from_pylist(
            [
                {"user_id": 1, "name": "Alice V1", "score": 150},
                {"user_id": 1, "name": "Alice V2", "score": 200},
            ],
            schema=ARROW,
        ),
        join_cols=["user_id"],
    )

    result = tbl.scan().to_arrow()
    assert result.num_rows == 2


# ==================== CORRECTNESS: OVER-APPROXIMATION + ANTI-JOIN ====================


def test_merge_preserves_non_matching_rows_in_same_file(catalog: Catalog) -> None:
    """The per-column In filter over-approximates: In(a,[1,2]) AND In(b,[x,y])
    matches (1,y) and (2,x) even if they're not in the source.

    The anti-join must correctly keep those rows.
    """
    schema = Schema(
        NestedField(1, "a", IntegerType(), required=True),
        NestedField(2, "b", StringType(), required=True),
        NestedField(3, "val", IntegerType(), required=True),
    )
    arrow = pa.schema(
        [
            pa.field("a", pa.int32(), nullable=False),
            pa.field("b", pa.string(), nullable=False),
            pa.field("val", pa.int32(), nullable=False),
        ]
    )
    ident = "default.merge_overapprox"
    _drop(catalog, ident)
    tbl = catalog.create_table(ident, schema=schema)

    tbl.append(
        pa.Table.from_pylist(
            [
                {"a": 1, "b": "x", "val": 100},
                {"a": 1, "b": "y", "val": 200},  # over-approx match, NOT in source
                {"a": 2, "b": "x", "val": 300},  # over-approx match, NOT in source
                {"a": 2, "b": "y", "val": 400},
                {"a": 3, "b": "z", "val": 500},  # completely unrelated
            ],
            schema=arrow,
        )
    )

    # Source only has (1,x) and (2,y) - not (1,y) or (2,x)
    tbl.merge(
        pa.Table.from_pylist(
            [
                {"a": 1, "b": "x", "val": 999},
                {"a": 2, "b": "y", "val": 888},
            ],
            schema=arrow,
        ),
        join_cols=["a", "b"],
    )

    result = tbl.scan().to_arrow().sort_by([("a", "ascending"), ("b", "ascending")])
    assert result.num_rows == 5
    assert result.column("val").to_pylist() == [999, 200, 300, 888, 500]


def test_merge_preserves_unrelated_rows_in_same_file(catalog: Catalog) -> None:
    """Rows in the same data file that don't match any join key must survive."""
    ident = "default.merge_unrelated"
    _drop(catalog, ident)
    tbl = catalog.create_table(ident, schema=SCHEMA)

    # All in one file (single append)
    tbl.append(
        pa.Table.from_pylist(
            [
                {"user_id": 1, "name": "Alice", "score": 100},
                {"user_id": 2, "name": "Bob", "score": 200},
                {"user_id": 3, "name": "Charlie", "score": 300},
                {"user_id": 4, "name": "Diana", "score": 400},
                {"user_id": 5, "name": "Eve", "score": 500},
            ],
            schema=ARROW,
        )
    )

    # Only replace user_id=2
    tbl.merge(
        pa.Table.from_pylist(
            [
                {"user_id": 2, "name": "Bob", "score": 999},
            ],
            schema=ARROW,
        ),
        join_cols=["user_id"],
    )

    result = tbl.scan().to_arrow().sort_by("user_id")
    assert result.num_rows == 5
    assert result.column("score").to_pylist() == [100, 999, 300, 400, 500]


def test_merge_schema_preserved_after_anti_join(catalog: Catalog) -> None:
    """Anti-join can strip nullability and field metadata.
    The output schema must match the Iceberg table schema exactly."""
    ident = "default.merge_schema"
    _drop(catalog, ident)
    tbl = catalog.create_table(ident, schema=SCHEMA)

    tbl.append(
        pa.Table.from_pylist(
            [
                {"user_id": 1, "name": "Alice", "score": 100},
                {"user_id": 2, "name": "Bob", "score": 200},
            ],
            schema=ARROW,
        )
    )

    tbl.merge(
        pa.Table.from_pylist(
            [
                {"user_id": 1, "name": "Alice", "score": 150},
            ],
            schema=ARROW,
        ),
        join_cols=["user_id"],
    )

    # Read back and verify data is correct
    result = tbl.scan().to_arrow().sort_by("user_id")
    assert result.num_rows == 2
    assert result.column("score").to_pylist() == [150, 200]

    # Verify we can merge again (schema must be valid for next write)
    tbl.merge(
        pa.Table.from_pylist(
            [
                {"user_id": 2, "name": "Bob", "score": 250},
            ],
            schema=ARROW,
        ),
        join_cols=["user_id"],
    )

    result = tbl.scan().to_arrow().sort_by("user_id")
    assert result.num_rows == 2
    assert result.column("score").to_pylist() == [150, 250]


# ==================== MERGE RESULT ====================


def test_merge_result_empty_df(catalog: Catalog) -> None:
    """Empty source returns zero counts and is a no-op."""
    ident = "default.merge_result_empty"
    _drop(catalog, ident)
    tbl = catalog.create_table(ident, schema=SCHEMA)

    tbl.append(
        pa.Table.from_pylist(
            [
                {"user_id": 1, "name": "Alice", "score": 100},
                {"user_id": 2, "name": "Bob", "score": 200},
                {"user_id": 3, "name": "Charlie", "score": 300},
            ],
            schema=ARROW,
        )
    )

    result = tbl.merge(pa.Table.from_pylist([], schema=ARROW), join_cols=["user_id"])

    assert result == MergeResult(rows_deleted=0, rows_inserted=0)
    assert tbl.scan().to_arrow().num_rows == 3


def test_merge_result_no_overlap(catalog: Catalog) -> None:
    """Source keys don't match any target rows - takes append fast-path."""
    ident = "default.merge_result_no_overlap"
    _drop(catalog, ident)
    tbl = catalog.create_table(ident, schema=SCHEMA)

    tbl.append(
        pa.Table.from_pylist(
            [
                {"user_id": 1, "name": "Alice", "score": 100},
                {"user_id": 2, "name": "Bob", "score": 200},
                {"user_id": 3, "name": "Charlie", "score": 300},
            ],
            schema=ARROW,
        )
    )

    result = tbl.merge(
        pa.Table.from_pylist(
            [
                {"user_id": 4, "name": "Dave", "score": 400},
                {"user_id": 5, "name": "Eve", "score": 500},
            ],
            schema=ARROW,
        ),
        join_cols=["user_id"],
    )

    assert result == MergeResult(rows_deleted=0, rows_inserted=2)
    assert tbl.scan().to_arrow().num_rows == 5


def test_merge_result_full_overlap(catalog: Catalog) -> None:
    """Every source key matches a target row - all replaced."""
    ident = "default.merge_result_full_overlap"
    _drop(catalog, ident)
    tbl = catalog.create_table(ident, schema=SCHEMA)

    tbl.append(
        pa.Table.from_pylist(
            [
                {"user_id": 1, "name": "Alice", "score": 100},
                {"user_id": 2, "name": "Bob", "score": 200},
                {"user_id": 3, "name": "Charlie", "score": 300},
            ],
            schema=ARROW,
        )
    )

    result = tbl.merge(
        pa.Table.from_pylist(
            [
                {"user_id": 1, "name": "Alice", "score": 150},
                {"user_id": 2, "name": "Bob", "score": 250},
                {"user_id": 3, "name": "Charlie", "score": 350},
            ],
            schema=ARROW,
        ),
        join_cols=["user_id"],
    )

    assert result == MergeResult(rows_deleted=3, rows_inserted=3)
    assert tbl.scan().to_arrow().num_rows == 3


def test_merge_result_partial_overlap(catalog: Catalog) -> None:
    """Some source keys match, some don't."""
    ident = "default.merge_result_partial"
    _drop(catalog, ident)
    tbl = catalog.create_table(ident, schema=SCHEMA)

    tbl.append(
        pa.Table.from_pylist(
            [
                {"user_id": 1, "name": "Alice", "score": 100},
                {"user_id": 2, "name": "Bob", "score": 200},
                {"user_id": 3, "name": "Charlie", "score": 300},
            ],
            schema=ARROW,
        )
    )

    result = tbl.merge(
        pa.Table.from_pylist(
            [
                {"user_id": 2, "name": "Bob", "score": 250},
                {"user_id": 3, "name": "Charlie", "score": 350},
                {"user_id": 4, "name": "Dave", "score": 400},
            ],
            schema=ARROW,
        ),
        join_cols=["user_id"],
    )

    assert result == MergeResult(rows_deleted=2, rows_inserted=3)
    assert tbl.scan().to_arrow().num_rows == 4


def test_merge_result_target_duplicates(catalog: Catalog) -> None:
    """Target has multiple rows per matching key - delete-insert is N:M, not 1:1."""
    ident = "default.merge_result_target_dups"
    _drop(catalog, ident)
    tbl = catalog.create_table(ident, schema=SCHEMA)

    tbl.append(
        pa.Table.from_pylist(
            [
                {"user_id": 1, "name": "Alice", "score": 100},
                {"user_id": 1, "name": "Alice", "score": 110},
                {"user_id": 1, "name": "Alice", "score": 120},
            ],
            schema=ARROW,
        )
    )

    result = tbl.merge(
        pa.Table.from_pylist(
            [
                {"user_id": 1, "name": "Alice", "score": 999},
            ],
            schema=ARROW,
        ),
        join_cols=["user_id"],
    )

    assert result == MergeResult(rows_deleted=3, rows_inserted=1)
    assert tbl.scan().to_arrow().num_rows == 1


def test_merge_result_source_duplicates(catalog: Catalog) -> None:
    """Source has duplicate keys - rows_inserted counts source verbatim."""
    ident = "default.merge_result_source_dups"
    _drop(catalog, ident)
    tbl = catalog.create_table(ident, schema=SCHEMA)

    tbl.append(
        pa.Table.from_pylist(
            [
                {"user_id": 1, "name": "Alice", "score": 100},
            ],
            schema=ARROW,
        )
    )

    result = tbl.merge(
        pa.Table.from_pylist(
            [
                {"user_id": 1, "name": "Alice", "score": 150},
                {"user_id": 1, "name": "Alice", "score": 160},
                {"user_id": 1, "name": "Alice", "score": 170},
            ],
            schema=ARROW,
        ),
        join_cols=["user_id"],
    )

    assert result == MergeResult(rows_deleted=1, rows_inserted=3)
    assert tbl.scan().to_arrow().num_rows == 3
