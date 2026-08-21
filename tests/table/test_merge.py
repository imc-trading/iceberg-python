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
from pyiceberg.table import Table
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


def _current_snapshot_id(tbl: Table) -> int:
    snap = tbl.current_snapshot()
    assert snap is not None, "expected a snapshot"
    return snap.snapshot_id


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


# ==================== ARROW TYPE-WIDTH DRIFT (string vs large_string) ====================


def test_merge_handles_large_string_kept_rows_with_string_source(catalog: Catalog) -> None:
    """Regression: kept_rows from ArrowScan can come back as ``large_string``
    while a pandas->arrow source df typically produces ``string``.

    The two encodings are semantically the same Iceberg text type but are
    physically distinct PyArrow types. ``pa.concat_tables(..., promote_options="default")``
    refuses to bridge them and merge() blows up with::

        ArrowTypeError: Unable to merge: Field <name> has incompatible types:
                       string vs large_string

    This is the exact failure observed against real Iceberg parquet tables
    (where ArrowScan returns ``large_*``) when consolidating multiple feeds
    into one table via composite merge keys.

    The fix in ``Transaction.merge()`` is to use ``promote_options="permissive"``
    (matching what ``pyiceberg/io/pyarrow.py`` already does for the same
    reason). This test pins that behavior.
    """
    ident = "default.merge_large_string_drift"
    _drop(catalog, ident)
    tbl = catalog.create_table(ident, schema=SCHEMA)

    # Write the initial rows with ``large_string`` to simulate what
    # ``ArrowScan.to_table()`` returns from a real Iceberg parquet read.
    large_string_arrow = pa.schema(
        [
            pa.field("user_id", pa.int32(), nullable=False),
            pa.field("name", pa.large_string(), nullable=False),
            pa.field("score", pa.int32(), nullable=False),
        ]
    )
    tbl.append(
        pa.Table.from_pylist(
            [
                {"user_id": 1, "name": "Alice", "score": 100},
                {"user_id": 2, "name": "Bob", "score": 200},
            ],
            schema=large_string_arrow,
        )
    )

    # New batch using plain ``string`` (the pandas->arrow default for
    # ``pd.StringDtype()`` columns).
    assert ARROW.field("name").type == pa.string()
    tbl.merge(
        pa.Table.from_pylist(
            [
                {"user_id": 1, "name": "Alice", "score": 150},
                {"user_id": 3, "name": "Charlie", "score": 300},
            ],
            schema=ARROW,
        ),
        join_cols=["user_id"],
    )

    result = tbl.scan().to_arrow().sort_by("user_id")
    assert result.num_rows == 3
    assert result.column("score").to_pylist() == [150, 200, 300]
    assert result.column("name").to_pylist() == ["Alice", "Bob", "Charlie"]


def test_merge_handles_string_kept_rows_with_large_string_source(catalog: Catalog) -> None:
    """Mirror of the previous test - target is ``string``, source is
    ``large_string``. Both directions must be tolerated since the
    drift can flip either way depending on PyArrow version, file
    metadata, and the ``PYARROW_USE_LARGE_TYPES_ON_READ`` setting.
    """
    ident = "default.merge_large_string_reverse"
    _drop(catalog, ident)
    tbl = catalog.create_table(ident, schema=SCHEMA)

    tbl.append(
        pa.Table.from_pylist(
            [
                {"user_id": 1, "name": "Alice", "score": 100},
                {"user_id": 2, "name": "Bob", "score": 200},
            ],
            schema=ARROW,  # plain string
        )
    )

    large_string_arrow = pa.schema(
        [
            pa.field("user_id", pa.int32(), nullable=False),
            pa.field("name", pa.large_string(), nullable=False),
            pa.field("score", pa.int32(), nullable=False),
        ]
    )
    tbl.merge(
        pa.Table.from_pylist(
            [
                {"user_id": 1, "name": "Alice", "score": 150},
            ],
            schema=large_string_arrow,
        ),
        join_cols=["user_id"],
    )

    result = tbl.scan().to_arrow().sort_by("user_id")
    assert result.num_rows == 2
    assert result.column("score").to_pylist() == [150, 200]


def test_merge_handles_large_binary_kept_rows_with_binary_source(catalog: Catalog) -> None:
    """Same drift family for binary columns: PyIceberg's ``ArrowScan`` can
    return ``large_binary`` while user-provided sources typically have
    ``binary``. ``"permissive"`` promote bridges these too.
    """
    from pyiceberg.types import BinaryType

    binary_schema = Schema(
        NestedField(1, "user_id", IntegerType(), required=True),
        NestedField(2, "payload", BinaryType(), required=True),
    )
    ident = "default.merge_large_binary_drift"
    _drop(catalog, ident)
    tbl = catalog.create_table(ident, schema=binary_schema)

    large_binary_arrow = pa.schema(
        [
            pa.field("user_id", pa.int32(), nullable=False),
            pa.field("payload", pa.large_binary(), nullable=False),
        ]
    )
    tbl.append(
        pa.Table.from_pylist(
            [
                {"user_id": 1, "payload": b"alpha"},
                {"user_id": 2, "payload": b"bravo"},
            ],
            schema=large_binary_arrow,
        )
    )

    binary_arrow = pa.schema(
        [
            pa.field("user_id", pa.int32(), nullable=False),
            pa.field("payload", pa.binary(), nullable=False),
        ]
    )
    tbl.merge(
        pa.Table.from_pylist(
            [
                {"user_id": 1, "payload": b"alpha-updated"},
                {"user_id": 3, "payload": b"charlie"},
            ],
            schema=binary_arrow,
        ),
        join_cols=["user_id"],
    )

    result = tbl.scan().to_arrow().sort_by("user_id")
    assert result.num_rows == 3
    assert result.column("payload").to_pylist() == [b"alpha-updated", b"bravo", b"charlie"]


# ==================== JOIN-COL TYPE-WIDTH DRIFT (EXHAUSTIVE) ====================
#
# pyarrow's ``Table.join`` is strict on physical encoding pairs that are
# semantically the same Iceberg type:
#
#     string         vs  large_string       (Iceberg StringType)
#     binary         vs  large_binary       (Iceberg BinaryType)
#
# ``Transaction.merge()`` reads target via ``ArrowScan`` (which can return
# either variant depending on parquet metadata and PyArrow version) and joins
# against a user-provided df (which can also be either variant depending on
# how it was built — pandas->arrow defaults to the small variants, polars and
# explicit large_* schemas use the large variants). Without normalization the
# join blows up with::
#
#     ArrowInvalid: Incompatible data types for corresponding join field keys:
#         FieldRef.Name(<col>) of type string
#         FieldRef.Name(<col>) of type large_string
#
# The earlier ``promote_options="permissive"`` fix on ``pa.concat_tables`` did
# NOT cover this — concat runs AFTER the join, so the join failure short-
# circuits before the permissive concat ever runs. That gap exists for all
# combinations of:
#
#   * direction      (target-large/source-small AND target-small/source-large)
#   * type family    (string / binary)
#   * key shape      (single col / composite / mixed types)
#   * match pattern  (all/none/partial/empty)
#   * edge values    (empty string, unicode, long strings, embedded NULs)
#   * call site      (Table.merge / Transaction.merge)
#
# The tests below pin every combination so a future change to the merge
# implementation that re-introduces type strictness will fail loudly, not
# silently regress for whichever combination ships in production.


def _str_schema(*, required: bool = True) -> Schema:
    return Schema(
        NestedField(1, "key", StringType(), required=required),
        NestedField(2, "value", IntegerType(), required=True),
    )


def _bin_schema(*, required: bool = True) -> Schema:
    from pyiceberg.types import BinaryType

    return Schema(
        NestedField(1, "key", BinaryType(), required=required),
        NestedField(2, "value", IntegerType(), required=True),
    )


def _arrow_schema_with_key(key_type: pa.DataType) -> pa.Schema:
    return pa.schema(
        [
            pa.field("key", key_type, nullable=False),
            pa.field("value", pa.int32(), nullable=False),
        ]
    )


# ---- (1) Single-col drift, all 4 directions, both type families ------------


@pytest.mark.parametrize(
    "target_type, source_type",
    [
        (pa.string(), pa.large_string()),
        (pa.large_string(), pa.string()),
        (pa.binary(), pa.large_binary()),
        (pa.large_binary(), pa.binary()),
    ],
    ids=[
        "string_vs_large_string",
        "large_string_vs_string",
        "binary_vs_large_binary",
        "large_binary_vs_binary",
    ],
)
def test_merge_join_col_drift_single(catalog: Catalog, target_type: pa.DataType, source_type: pa.DataType) -> None:
    """Drift on a single string/binary join column, every direction.

    Pinned values exercise:
      - one matching key (replaced)
      - one non-matching target key (kept)
      - one new source key (inserted)
    """
    is_binary = pa.types.is_binary(target_type) or pa.types.is_large_binary(target_type)
    schema = _bin_schema() if is_binary else _str_schema()
    a, b, c = (b"a", b"b", b"c") if is_binary else ("a", "b", "c")
    ident = f"default.merge_join_drift_single_{target_type}_{source_type}"
    _drop(catalog, ident)
    tbl = catalog.create_table(ident, schema=schema)

    tbl.append(
        pa.Table.from_pylist(
            [{"key": a, "value": 1}, {"key": b, "value": 2}],
            schema=_arrow_schema_with_key(target_type),
        )
    )
    result = tbl.merge(
        pa.Table.from_pylist(
            [{"key": a, "value": 11}, {"key": c, "value": 3}],
            schema=_arrow_schema_with_key(source_type),
        ),
        join_cols=["key"],
    )

    out = tbl.scan().to_arrow().sort_by("key")
    assert out.num_rows == 3
    assert out.column("key").to_pylist() == [a, b, c]
    assert out.column("value").to_pylist() == [11, 2, 3]
    assert (result.rows_deleted, result.rows_inserted) == (1, 2)


# ---- (2) Composite keys: drift on one col, multiple cols, mixed types ------


@pytest.mark.parametrize(
    "drift_target, drift_source",
    [
        (pa.string(), pa.large_string()),
        (pa.large_string(), pa.string()),
        (pa.binary(), pa.large_binary()),
        (pa.large_binary(), pa.binary()),
    ],
    ids=["str_drift_LtoS", "str_drift_StoL", "bin_drift_LtoS", "bin_drift_StoL"],
)
def test_merge_composite_key_drift_on_one_of_two(catalog: Catalog, drift_target: pa.DataType, drift_source: pa.DataType) -> None:
    """Composite (string-or-binary, int) join key. Drift on the text/binary
    half, int half is stable. Both halves must participate in the join.
    """
    from pyiceberg.types import BinaryType

    is_binary = pa.types.is_binary(drift_target) or pa.types.is_large_binary(drift_target)
    iceberg_inner = BinaryType() if is_binary else StringType()
    schema = Schema(
        NestedField(1, "key_a", iceberg_inner, required=True),
        NestedField(2, "key_b", IntegerType(), required=True),
        NestedField(3, "value", IntegerType(), required=True),
    )
    a, b = (b"a", b"b") if is_binary else ("a", "b")
    ident = f"default.merge_composite_drift_one_{drift_target}_{drift_source}"
    _drop(catalog, ident)
    tbl = catalog.create_table(ident, schema=schema)

    target_arrow = pa.schema(
        [
            pa.field("key_a", drift_target, nullable=False),
            pa.field("key_b", pa.int32(), nullable=False),
            pa.field("value", pa.int32(), nullable=False),
        ]
    )
    source_arrow = pa.schema(
        [
            pa.field("key_a", drift_source, nullable=False),
            pa.field("key_b", pa.int32(), nullable=False),
            pa.field("value", pa.int32(), nullable=False),
        ]
    )

    tbl.append(
        pa.Table.from_pylist(
            [
                {"key_a": a, "key_b": 1, "value": 100},
                {"key_a": a, "key_b": 2, "value": 200},  # same key_a, different key_b -> kept
                {"key_a": b, "key_b": 1, "value": 300},
            ],
            schema=target_arrow,
        )
    )
    tbl.merge(
        pa.Table.from_pylist(
            [
                {"key_a": a, "key_b": 1, "value": 999},  # exact (a,1) -> replace
                {"key_a": b, "key_b": 9, "value": 800},  # new combo -> insert
            ],
            schema=source_arrow,
        ),
        join_cols=["key_a", "key_b"],
    )

    out = tbl.scan().to_arrow().sort_by([("key_a", "ascending"), ("key_b", "ascending")])
    assert out.num_rows == 4
    assert out.column("value").to_pylist() == [999, 200, 300, 800]


@pytest.mark.parametrize(
    "drift_target, drift_source",
    [
        (pa.string(), pa.large_string()),
        (pa.large_string(), pa.string()),
    ],
    ids=["str_drift_LtoS", "str_drift_StoL"],
)
def test_merge_composite_key_drift_on_both_string_cols(
    catalog: Catalog, drift_target: pa.DataType, drift_source: pa.DataType
) -> None:
    """Both cols of a composite key drift simultaneously."""
    schema = Schema(
        NestedField(1, "key_a", StringType(), required=True),
        NestedField(2, "key_b", StringType(), required=True),
        NestedField(3, "value", IntegerType(), required=True),
    )
    ident = f"default.merge_composite_drift_both_{drift_target}_{drift_source}"
    _drop(catalog, ident)
    tbl = catalog.create_table(ident, schema=schema)

    target_arrow = pa.schema(
        [
            pa.field("key_a", drift_target, nullable=False),
            pa.field("key_b", drift_target, nullable=False),
            pa.field("value", pa.int32(), nullable=False),
        ]
    )
    source_arrow = pa.schema(
        [
            pa.field("key_a", drift_source, nullable=False),
            pa.field("key_b", drift_source, nullable=False),
            pa.field("value", pa.int32(), nullable=False),
        ]
    )

    tbl.append(
        pa.Table.from_pylist(
            [
                {"key_a": "x", "key_b": "1", "value": 1},
                {"key_a": "y", "key_b": "1", "value": 2},
            ],
            schema=target_arrow,
        )
    )
    tbl.merge(
        pa.Table.from_pylist(
            [
                {"key_a": "x", "key_b": "1", "value": 99},
                {"key_a": "z", "key_b": "9", "value": 100},
            ],
            schema=source_arrow,
        ),
        join_cols=["key_a", "key_b"],
    )

    out = tbl.scan().to_arrow().sort_by([("key_a", "ascending"), ("key_b", "ascending")])
    assert out.num_rows == 3
    assert out.column("value").to_pylist() == [99, 2, 100]


def test_merge_composite_key_mixed_string_and_binary_drift(catalog: Catalog) -> None:
    """Composite key spans both type families with drift on each."""
    from pyiceberg.types import BinaryType

    schema = Schema(
        NestedField(1, "txt_key", StringType(), required=True),
        NestedField(2, "bin_key", BinaryType(), required=True),
        NestedField(3, "value", IntegerType(), required=True),
    )
    ident = "default.merge_composite_mixed_drift"
    _drop(catalog, ident)
    tbl = catalog.create_table(ident, schema=schema)

    # Target: large_string + large_binary
    target_arrow = pa.schema(
        [
            pa.field("txt_key", pa.large_string(), nullable=False),
            pa.field("bin_key", pa.large_binary(), nullable=False),
            pa.field("value", pa.int32(), nullable=False),
        ]
    )
    # Source: string + binary (opposite drift on both)
    source_arrow = pa.schema(
        [
            pa.field("txt_key", pa.string(), nullable=False),
            pa.field("bin_key", pa.binary(), nullable=False),
            pa.field("value", pa.int32(), nullable=False),
        ]
    )

    tbl.append(
        pa.Table.from_pylist(
            [
                {"txt_key": "alpha", "bin_key": b"\x01", "value": 1},
                {"txt_key": "beta", "bin_key": b"\x02", "value": 2},
            ],
            schema=target_arrow,
        )
    )
    tbl.merge(
        pa.Table.from_pylist(
            [
                {"txt_key": "alpha", "bin_key": b"\x01", "value": 99},  # replace
                {"txt_key": "gamma", "bin_key": b"\x03", "value": 3},  # insert
            ],
            schema=source_arrow,
        ),
        join_cols=["txt_key", "bin_key"],
    )

    out = tbl.scan().to_arrow().sort_by("txt_key")
    assert out.num_rows == 3
    assert out.column("value").to_pylist() == [99, 2, 3]
    assert out.column("txt_key").to_pylist() == ["alpha", "beta", "gamma"]
    assert out.column("bin_key").to_pylist() == [b"\x01", b"\x02", b"\x03"]


# ---- (3) Drift on join AND non-join column simultaneously ------------------


def test_merge_drift_on_join_and_non_join_cols_simultaneously(catalog: Catalog) -> None:
    """Both the join column AND a non-join column drift in the SAME merge.
    Exercises the join-step fix AND the concat-step permissive promotion
    in the same call. Either fix alone is insufficient.
    """
    schema = Schema(
        NestedField(1, "key", StringType(), required=True),
        NestedField(2, "name", StringType(), required=True),
        NestedField(3, "score", IntegerType(), required=True),
    )
    ident = "default.merge_drift_both_join_and_nonjoin"
    _drop(catalog, ident)
    tbl = catalog.create_table(ident, schema=schema)

    # Target: large_string for both string cols
    target_arrow = pa.schema(
        [
            pa.field("key", pa.large_string(), nullable=False),
            pa.field("name", pa.large_string(), nullable=False),
            pa.field("score", pa.int32(), nullable=False),
        ]
    )
    # Source: string for both string cols
    source_arrow = pa.schema(
        [
            pa.field("key", pa.string(), nullable=False),
            pa.field("name", pa.string(), nullable=False),
            pa.field("score", pa.int32(), nullable=False),
        ]
    )

    tbl.append(
        pa.Table.from_pylist(
            [
                {"key": "k1", "name": "Alice", "score": 10},
                {"key": "k2", "name": "Bob", "score": 20},
            ],
            schema=target_arrow,
        )
    )
    tbl.merge(
        pa.Table.from_pylist(
            [
                {"key": "k1", "name": "Alice2", "score": 99},
                {"key": "k3", "name": "Charlie", "score": 30},
            ],
            schema=source_arrow,
        ),
        join_cols=["key"],
    )

    out = tbl.scan().to_arrow().sort_by("key")
    assert out.num_rows == 3
    assert out.column("name").to_pylist() == ["Alice2", "Bob", "Charlie"]
    assert out.column("score").to_pylist() == [99, 20, 30]


def test_merge_drift_on_join_and_non_join_in_opposite_directions(catalog: Catalog) -> None:
    """Pathological: drift goes one direction on the join col, the OTHER
    direction on the non-join col. Both fixes still need to apply.
    """
    schema = Schema(
        NestedField(1, "key", StringType(), required=True),
        NestedField(2, "name", StringType(), required=True),
        NestedField(3, "score", IntegerType(), required=True),
    )
    ident = "default.merge_drift_opposite_directions"
    _drop(catalog, ident)
    tbl = catalog.create_table(ident, schema=schema)

    # Target: key=string, name=large_string
    target_arrow = pa.schema(
        [
            pa.field("key", pa.string(), nullable=False),
            pa.field("name", pa.large_string(), nullable=False),
            pa.field("score", pa.int32(), nullable=False),
        ]
    )
    # Source: key=large_string, name=string  (opposite drift)
    source_arrow = pa.schema(
        [
            pa.field("key", pa.large_string(), nullable=False),
            pa.field("name", pa.string(), nullable=False),
            pa.field("score", pa.int32(), nullable=False),
        ]
    )

    tbl.append(
        pa.Table.from_pylist(
            [{"key": "k1", "name": "A", "score": 1}, {"key": "k2", "name": "B", "score": 2}],
            schema=target_arrow,
        )
    )
    tbl.merge(
        pa.Table.from_pylist(
            [{"key": "k1", "name": "A2", "score": 99}, {"key": "k3", "name": "C", "score": 3}],
            schema=source_arrow,
        ),
        join_cols=["key"],
    )

    out = tbl.scan().to_arrow().sort_by("key")
    assert out.num_rows == 3
    assert out.column("name").to_pylist() == ["A2", "B", "C"]


# ---- (4) Match patterns with drift -----------------------------------------


@pytest.mark.parametrize(
    "target_type, source_type",
    [(pa.string(), pa.large_string()), (pa.large_string(), pa.string())],
    ids=["LtoS", "StoL"],
)
def test_merge_drift_all_source_keys_match(catalog: Catalog, target_type: pa.DataType, source_type: pa.DataType) -> None:
    """Every source key matches a target key. All target rows replaced."""
    schema = _str_schema()
    ident = f"default.merge_drift_all_match_{target_type}"
    _drop(catalog, ident)
    tbl = catalog.create_table(ident, schema=schema)

    tbl.append(
        pa.Table.from_pylist(
            [{"key": "a", "value": 1}, {"key": "b", "value": 2}],
            schema=_arrow_schema_with_key(target_type),
        )
    )
    result = tbl.merge(
        pa.Table.from_pylist(
            [{"key": "a", "value": 11}, {"key": "b", "value": 22}],
            schema=_arrow_schema_with_key(source_type),
        ),
        join_cols=["key"],
    )

    out = tbl.scan().to_arrow().sort_by("key")
    assert out.num_rows == 2
    assert out.column("value").to_pylist() == [11, 22]
    assert (result.rows_deleted, result.rows_inserted) == (2, 2)


@pytest.mark.parametrize(
    "target_type, source_type",
    [(pa.string(), pa.large_string()), (pa.large_string(), pa.string())],
    ids=["LtoS", "StoL"],
)
def test_merge_drift_no_source_keys_match(catalog: Catalog, target_type: pa.DataType, source_type: pa.DataType) -> None:
    """No source keys match target. Hits the early-return ``no candidate
    files`` path before the join. Drift fix should not interfere - this
    must still work correctly.
    """
    schema = _str_schema()
    ident = f"default.merge_drift_no_match_{target_type}"
    _drop(catalog, ident)
    tbl = catalog.create_table(ident, schema=schema)

    tbl.append(
        pa.Table.from_pylist(
            [{"key": "a", "value": 1}, {"key": "b", "value": 2}],
            schema=_arrow_schema_with_key(target_type),
        )
    )
    result = tbl.merge(
        pa.Table.from_pylist(
            [{"key": "x", "value": 10}, {"key": "y", "value": 20}],
            schema=_arrow_schema_with_key(source_type),
        ),
        join_cols=["key"],
    )

    out = tbl.scan().to_arrow().sort_by("key")
    assert out.num_rows == 4
    assert sorted(out.column("key").to_pylist()) == ["a", "b", "x", "y"]
    assert (result.rows_deleted, result.rows_inserted) == (0, 2)


# ---- (5) Edge values in join key (with drift) ------------------------------


@pytest.mark.parametrize(
    "target_type, source_type",
    [(pa.string(), pa.large_string()), (pa.large_string(), pa.string())],
    ids=["LtoS", "StoL"],
)
def test_merge_drift_with_empty_string_key(catalog: Catalog, target_type: pa.DataType, source_type: pa.DataType) -> None:
    """Empty-string join key is a real value (distinct from null) and must
    round-trip through the drift-fix cast cleanly.
    """
    schema = _str_schema()
    ident = f"default.merge_drift_empty_str_{target_type}"
    _drop(catalog, ident)
    tbl = catalog.create_table(ident, schema=schema)

    tbl.append(
        pa.Table.from_pylist(
            [{"key": "", "value": 1}, {"key": "x", "value": 2}],
            schema=_arrow_schema_with_key(target_type),
        )
    )
    tbl.merge(
        pa.Table.from_pylist(
            [{"key": "", "value": 99}],
            schema=_arrow_schema_with_key(source_type),
        ),
        join_cols=["key"],
    )

    out = tbl.scan().to_arrow().sort_by("key")
    assert out.num_rows == 2
    assert out.column("key").to_pylist() == ["", "x"]
    assert out.column("value").to_pylist() == [99, 2]


@pytest.mark.parametrize(
    "target_type, source_type",
    [(pa.string(), pa.large_string()), (pa.large_string(), pa.string())],
    ids=["LtoS", "StoL"],
)
def test_merge_drift_with_unicode_key(catalog: Catalog, target_type: pa.DataType, source_type: pa.DataType) -> None:
    """Multi-byte unicode keys: ensure the drift cast preserves UTF-8
    semantics, not just the bytes-modulo-encoding shape.
    """
    schema = _str_schema()
    ident = f"default.merge_drift_unicode_{target_type}"
    _drop(catalog, ident)
    tbl = catalog.create_table(ident, schema=schema)

    tbl.append(
        pa.Table.from_pylist(
            [
                {"key": "naïve", "value": 1},
                {"key": "日本語", "value": 2},
                {"key": "emoji-🚀", "value": 3},
            ],
            schema=_arrow_schema_with_key(target_type),
        )
    )
    tbl.merge(
        pa.Table.from_pylist(
            [{"key": "日本語", "value": 99}, {"key": "café", "value": 4}],
            schema=_arrow_schema_with_key(source_type),
        ),
        join_cols=["key"],
    )

    out = tbl.scan().to_arrow().sort_by("key")
    assert out.num_rows == 4
    assert sorted(out.column("key").to_pylist()) == sorted(["naïve", "日本語", "emoji-🚀", "café"])
    rows_by_key = dict(zip(out.column("key").to_pylist(), out.column("value").to_pylist(), strict=True))
    assert rows_by_key["日本語"] == 99
    assert rows_by_key["naïve"] == 1


@pytest.mark.parametrize(
    "target_type, source_type",
    [(pa.string(), pa.large_string()), (pa.large_string(), pa.string())],
    ids=["LtoS", "StoL"],
)
def test_merge_drift_with_long_string_key(catalog: Catalog, target_type: pa.DataType, source_type: pa.DataType) -> None:
    """Long strings (10K chars) - well below either offset width's limit,
    but enough to confirm large->small cast doesn't truncate.
    """
    schema = _str_schema()
    ident = f"default.merge_drift_long_str_{target_type}"
    _drop(catalog, ident)
    tbl = catalog.create_table(ident, schema=schema)

    long_key = "x" * 10_000
    other = "y" * 10_000
    tbl.append(
        pa.Table.from_pylist(
            [{"key": long_key, "value": 1}, {"key": other, "value": 2}],
            schema=_arrow_schema_with_key(target_type),
        )
    )
    tbl.merge(
        pa.Table.from_pylist(
            [{"key": long_key, "value": 99}],
            schema=_arrow_schema_with_key(source_type),
        ),
        join_cols=["key"],
    )

    out = tbl.scan().to_arrow().sort_by("key")
    assert out.num_rows == 2
    rows_by_key = dict(zip(out.column("key").to_pylist(), out.column("value").to_pylist(), strict=True))
    assert rows_by_key[long_key] == 99
    assert rows_by_key[other] == 2


@pytest.mark.parametrize(
    "target_type, source_type",
    [(pa.binary(), pa.large_binary()), (pa.large_binary(), pa.binary())],
    ids=["LtoS", "StoL"],
)
def test_merge_drift_with_empty_and_high_bit_binary_key(
    catalog: Catalog, target_type: pa.DataType, source_type: pa.DataType
) -> None:
    """Binary join key edge values: empty bytes, high-bit (>=0x80) bytes
    that aren't valid UTF-8, and embedded NULs.
    """
    schema = _bin_schema()
    ident = f"default.merge_drift_bin_edge_{target_type}"
    _drop(catalog, ident)
    tbl = catalog.create_table(ident, schema=schema)

    keys = [b"", b"\x00\x01\x02", b"\xff\xfe\xfd"]
    tbl.append(
        pa.Table.from_pylist(
            [{"key": k, "value": i} for i, k in enumerate(keys)],
            schema=_arrow_schema_with_key(target_type),
        )
    )
    tbl.merge(
        pa.Table.from_pylist(
            [{"key": b"\x00\x01\x02", "value": 999}, {"key": b"\x80\x81", "value": 7}],
            schema=_arrow_schema_with_key(source_type),
        ),
        join_cols=["key"],
    )

    out = tbl.scan().to_arrow()
    rows_by_key = dict(zip(out.column("key").to_pylist(), out.column("value").to_pylist(), strict=True))
    assert rows_by_key[b""] == 0
    assert rows_by_key[b"\x00\x01\x02"] == 999
    assert rows_by_key[b"\xff\xfe\xfd"] == 2
    assert rows_by_key[b"\x80\x81"] == 7


# ---- (6) MergeResult correctness with drift --------------------------------


@pytest.mark.parametrize(
    "target_type, source_type",
    [(pa.string(), pa.large_string()), (pa.large_string(), pa.string())],
    ids=["LtoS", "StoL"],
)
def test_merge_result_counts_correct_with_drift(catalog: Catalog, target_type: pa.DataType, source_type: pa.DataType) -> None:
    """``MergeResult(rows_deleted, rows_inserted)`` must be accurate even
    when the drift cast runs - rows_deleted is computed from the join
    output and could be wrong if the cast accidentally changed cardinality.
    """
    schema = _str_schema()
    ident = f"default.merge_drift_result_counts_{target_type}"
    _drop(catalog, ident)
    tbl = catalog.create_table(ident, schema=schema)

    tbl.append(
        pa.Table.from_pylist(
            [{"key": k, "value": i} for i, k in enumerate(["a", "b", "c", "d"])],
            schema=_arrow_schema_with_key(target_type),
        )
    )
    result = tbl.merge(
        pa.Table.from_pylist(
            [{"key": "b", "value": 99}, {"key": "c", "value": 88}, {"key": "z", "value": 77}],
            schema=_arrow_schema_with_key(source_type),
        ),
        join_cols=["key"],
    )

    assert (result.rows_deleted, result.rows_inserted) == (2, 3)
    assert tbl.scan().to_arrow().num_rows == 5


# ---- (7) Re-merge / idempotency / alternating directions -------------------


def test_merge_drift_idempotent_when_reapplied(catalog: Catalog) -> None:
    """Applying the same merge twice (with drift) is a no-op the second
    time: same source keys already match what's on disk, so nothing
    changes. Pins that the drift cast doesn't accidentally produce
    pseudo-different keys that would phantom-replace rows.
    """
    schema = _str_schema()
    ident = "default.merge_drift_idempotent"
    _drop(catalog, ident)
    tbl = catalog.create_table(ident, schema=schema)

    tbl.append(
        pa.Table.from_pylist(
            [{"key": "a", "value": 1}, {"key": "b", "value": 2}],
            schema=_arrow_schema_with_key(pa.large_string()),
        )
    )
    src = pa.Table.from_pylist(
        [{"key": "a", "value": 99}, {"key": "c", "value": 3}],
        schema=_arrow_schema_with_key(pa.string()),
    )
    r1 = tbl.merge(src, join_cols=["key"])
    r2 = tbl.merge(src, join_cols=["key"])

    assert (r1.rows_deleted, r1.rows_inserted) == (1, 2)
    assert (r2.rows_deleted, r2.rows_inserted) == (2, 2)  # second pass replaces both rows it inserted
    out = tbl.scan().to_arrow().sort_by("key")
    assert out.num_rows == 3
    assert out.column("value").to_pylist() == [99, 2, 3]


def test_merge_drift_alternating_directions(catalog: Catalog) -> None:
    """Successive merges flipping drift direction every time. After each
    merge, on-disk schema is whatever the iceberg canonicalization says
    - the next merge may see a different target encoding, and the fix
    must work either way.
    """
    schema = _str_schema()
    ident = "default.merge_drift_alternating"
    _drop(catalog, ident)
    tbl = catalog.create_table(ident, schema=schema)

    tbl.append(
        pa.Table.from_pylist(
            [{"key": "a", "value": 1}],
            schema=_arrow_schema_with_key(pa.large_string()),
        )
    )
    # Each merge alternates source encoding.
    encodings = [pa.string(), pa.large_string(), pa.string(), pa.large_string()]
    for i, enc in enumerate(encodings):
        tbl.merge(
            pa.Table.from_pylist(
                [{"key": "a", "value": 100 + i}, {"key": f"k{i}", "value": 200 + i}],
                schema=_arrow_schema_with_key(enc),
            ),
            join_cols=["key"],
        )

    out = tbl.scan().to_arrow().sort_by("key")
    rows_by_key = dict(zip(out.column("key").to_pylist(), out.column("value").to_pylist(), strict=True))
    assert rows_by_key["a"] == 103  # last merge wins
    assert all(f"k{i}" in rows_by_key for i in range(4))


# ---- (8) Transaction.merge() with drift -----------------------------------


def test_transaction_merge_with_drift(catalog: Catalog) -> None:
    """Same fix must apply when the caller goes through a transaction
    rather than the table-level convenience method.
    """
    schema = _str_schema()
    ident = "default.merge_drift_transaction"
    _drop(catalog, ident)
    tbl = catalog.create_table(ident, schema=schema)

    tbl.append(
        pa.Table.from_pylist(
            [{"key": "a", "value": 1}, {"key": "b", "value": 2}],
            schema=_arrow_schema_with_key(pa.large_string()),
        )
    )
    with tbl.transaction() as tx:
        tx.merge(
            pa.Table.from_pylist(
                [{"key": "a", "value": 11}, {"key": "c", "value": 3}],
                schema=_arrow_schema_with_key(pa.string()),
            ),
            join_cols=["key"],
        )

    out = tbl.scan().to_arrow().sort_by("key")
    assert out.num_rows == 3
    assert out.column("value").to_pylist() == [11, 2, 3]


# ---- (9) Iceberg metadata stability (drift fix doesn't change schema) -----


def test_merge_with_drift_preserves_iceberg_schema(catalog: Catalog) -> None:
    """After a drift-handled merge, the Iceberg-level field type must
    remain ``StringType`` (the casts are PyArrow physical-encoding only -
    they must not leak into the on-disk Iceberg schema).
    """
    schema = _str_schema()
    ident = "default.merge_drift_schema_stable"
    _drop(catalog, ident)
    tbl = catalog.create_table(ident, schema=schema)

    tbl.append(
        pa.Table.from_pylist(
            [{"key": "a", "value": 1}],
            schema=_arrow_schema_with_key(pa.large_string()),
        )
    )
    tbl.merge(
        pa.Table.from_pylist(
            [{"key": "a", "value": 99}],
            schema=_arrow_schema_with_key(pa.string()),
        ),
        join_cols=["key"],
    )

    field = tbl.schema().find_field("key")
    assert isinstance(field.field_type, StringType)


# ---- (10) Pandas-built source (real caller path) --------------------------


def test_merge_drift_with_pandas_source(catalog: Catalog) -> None:
    """The original failure originated from a pandas DataFrame. Pin that
    path explicitly. Pandas's default arrow conversion produces
    ``string`` (small variant), so this drives target-large/source-small.
    """
    pd = pytest.importorskip("pandas")
    schema = _str_schema()
    ident = "default.merge_drift_pandas"
    _drop(catalog, ident)
    tbl = catalog.create_table(ident, schema=schema)

    tbl.append(
        pa.Table.from_pylist(
            [{"key": "a", "value": 1}, {"key": "b", "value": 2}],
            schema=_arrow_schema_with_key(pa.large_string()),
        )
    )

    df = pd.DataFrame({"key": ["a", "c"], "value": [99, 3]})
    df["key"] = df["key"].astype("string")  # pandas StringDtype -> arrow string()
    df["value"] = df["value"].astype("int32")
    # Build with required-fields schema to satisfy iceberg's compatibility check.
    arrow_from_pandas = pa.Table.from_pandas(df, preserve_index=False, schema=_arrow_schema_with_key(pa.string()))
    # Sanity: confirm pandas produced the small variant for our drift test.
    assert pa.types.is_string(arrow_from_pandas.schema.field("key").type)

    tbl.merge(arrow_from_pandas, join_cols=["key"])

    out = tbl.scan().to_arrow().sort_by("key")
    assert out.num_rows == 3
    assert out.column("value").to_pylist() == [99, 2, 3]


# ---- (11) Negative tests: drift fix must not mask real errors -------------


def test_merge_drift_does_not_mask_missing_join_col(catalog: Catalog) -> None:
    """Source missing a join column must still raise the standard error,
    not be silently widened by the drift cast.
    """
    schema = _str_schema()
    ident = "default.merge_drift_missing_join"
    _drop(catalog, ident)
    tbl = catalog.create_table(ident, schema=schema)
    tbl.append(
        pa.Table.from_pylist(
            [{"key": "a", "value": 1}],
            schema=_arrow_schema_with_key(pa.large_string()),
        )
    )

    bad_source = pa.Table.from_pylist([{"value": 99}], schema=pa.schema([pa.field("value", pa.int32(), nullable=False)]))
    with pytest.raises(ValueError, match="join_cols not found"):
        tbl.merge(bad_source, join_cols=["key"])


def test_merge_drift_with_single_row_target_and_source(catalog: Catalog) -> None:
    """Boundary: 1-row target, 1-row matching source. The smallest possible
    join still exercises the cast.
    """
    schema = _str_schema()
    ident = "default.merge_drift_single_row"
    _drop(catalog, ident)
    tbl = catalog.create_table(ident, schema=schema)

    tbl.append(pa.Table.from_pylist([{"key": "a", "value": 1}], schema=_arrow_schema_with_key(pa.large_string())))
    result = tbl.merge(
        pa.Table.from_pylist([{"key": "a", "value": 99}], schema=_arrow_schema_with_key(pa.string())),
        join_cols=["key"],
    )

    assert (result.rows_deleted, result.rows_inserted) == (1, 1)
    out = tbl.scan().to_arrow()
    assert out.num_rows == 1
    assert out.column("value").to_pylist() == [99]


# ---- (12) Multi-file target: ArrowScan reads multiple files with drift ----


@pytest.mark.parametrize(
    "target_type, source_type",
    [(pa.string(), pa.large_string()), (pa.large_string(), pa.string())],
    ids=["LtoS", "StoL"],
)
def test_merge_drift_with_multi_file_target(catalog: Catalog, target_type: pa.DataType, source_type: pa.DataType) -> None:
    """Real Iceberg tables are usually multi-file. ArrowScan reads from
    several parquet files at once and concatenates the result. Pin that
    the drift cast works on the concatenated multi-file scan output.
    """
    schema = _str_schema()
    ident = f"default.merge_drift_multi_file_{target_type}"
    _drop(catalog, ident)
    tbl = catalog.create_table(ident, schema=schema)

    # Three separate appends -> three parquet files in the table.
    for i in range(3):
        tbl.append(
            pa.Table.from_pylist(
                [{"key": f"k{i}_a", "value": i * 10 + 1}, {"key": f"k{i}_b", "value": i * 10 + 2}],
                schema=_arrow_schema_with_key(target_type),
            )
        )

    tbl.merge(
        pa.Table.from_pylist(
            [
                {"key": "k0_a", "value": 999},
                {"key": "k1_b", "value": 998},
                {"key": "k2_a", "value": 997},
                {"key": "new", "value": 100},
            ],
            schema=_arrow_schema_with_key(source_type),
        ),
        join_cols=["key"],
    )

    out = tbl.scan().to_arrow()
    rows_by_key = dict(zip(out.column("key").to_pylist(), out.column("value").to_pylist(), strict=True))
    assert rows_by_key["k0_a"] == 999
    assert rows_by_key["k1_b"] == 998
    assert rows_by_key["k2_a"] == 997
    assert rows_by_key["new"] == 100
    assert rows_by_key["k0_b"] == 2
    assert rows_by_key["k1_a"] == 11
    assert rows_by_key["k2_b"] == 22
    assert out.num_rows == 7


# ---- (13) NULL handling in join keys with drift ----------------------------


# ---- NULL join keys are rejected on both sides ----------------------------


def test_merge_rejects_null_in_single_source_join_col(catalog: Catalog) -> None:
    """Source NULL in the join column is rejected by pyiceberg's ``In``
    predicate constructor (which forbids NULL literals).  We rely on this
    native rejection instead of an explicit check so behavior matches
    upsert() and the spec.  NULL has undefined equality semantics
    (SQL three-valued logic) so it can never be a valid join key.
    """
    schema = Schema(
        NestedField(1, "key", StringType(), required=False),
        NestedField(2, "value", IntegerType(), required=True),
    )
    ident = "default.merge_reject_source_null_single"
    _drop(catalog, ident)
    tbl = catalog.create_table(ident, schema=schema)
    tbl.append(
        pa.Table.from_pylist(
            [{"key": "a", "value": 1}],
            schema=pa.schema([pa.field("key", pa.large_string(), nullable=True), pa.field("value", pa.int32(), nullable=False)]),
        )
    )

    src = pa.Table.from_pylist(
        [{"key": None, "value": 99}, {"key": "a", "value": 88}],
        schema=pa.schema([pa.field("key", pa.string(), nullable=True), pa.field("value", pa.int32(), nullable=False)]),
    )
    with pytest.raises((TypeError, ValueError)):
        tbl.merge(src, join_cols=["key"])

    # Target unchanged.
    assert tbl.scan().to_arrow().num_rows == 1


def test_merge_rejects_null_in_one_of_composite_source_join_cols(catalog: Catalog) -> None:
    """Composite key with NULL in one component must also be rejected.
    The error message names the offending column.
    """
    schema = Schema(
        NestedField(1, "key_a", StringType(), required=False),
        NestedField(2, "key_b", IntegerType(), required=True),
        NestedField(3, "value", IntegerType(), required=True),
    )
    ident = "default.merge_reject_composite_source_null"
    _drop(catalog, ident)
    tbl = catalog.create_table(ident, schema=schema)
    target_arrow = pa.schema(
        [
            pa.field("key_a", pa.large_string(), nullable=True),
            pa.field("key_b", pa.int32(), nullable=False),
            pa.field("value", pa.int32(), nullable=False),
        ]
    )
    tbl.append(
        pa.Table.from_pylist(
            [{"key_a": "x", "key_b": 1, "value": 10}],
            schema=target_arrow,
        )
    )

    source_arrow = pa.schema(
        [
            pa.field("key_a", pa.string(), nullable=True),
            pa.field("key_b", pa.int32(), nullable=False),
            pa.field("value", pa.int32(), nullable=False),
        ]
    )
    src = pa.Table.from_pylist(
        [
            {"key_a": "x", "key_b": 1, "value": 99},
            {"key_a": None, "key_b": 2, "value": 88},  # NULL in key_a -> reject
        ],
        schema=source_arrow,
    )
    with pytest.raises((TypeError, ValueError)):
        tbl.merge(src, join_cols=["key_a", "key_b"])


def test_merge_rejects_when_all_source_keys_null(catalog: Catalog) -> None:
    """All source rows have NULL in the join col -> rejected (not
    silently degraded to append). Symmetric with the partial-NULL case.
    """
    schema = Schema(
        NestedField(1, "key", StringType(), required=False),
        NestedField(2, "value", IntegerType(), required=True),
    )
    ident = "default.merge_reject_all_source_null"
    _drop(catalog, ident)
    tbl = catalog.create_table(ident, schema=schema)
    tbl.append(
        pa.Table.from_pylist(
            [{"key": "a", "value": 1}],
            schema=pa.schema([pa.field("key", pa.large_string(), nullable=True), pa.field("value", pa.int32(), nullable=False)]),
        )
    )

    src = pa.Table.from_pylist(
        [{"key": None, "value": 10}, {"key": None, "value": 20}],
        schema=pa.schema([pa.field("key", pa.string(), nullable=True), pa.field("value", pa.int32(), nullable=False)]),
    )
    with pytest.raises((TypeError, ValueError)):
        tbl.merge(src, join_cols=["key"])


def test_merge_preserves_null_in_target_join_col(catalog: Catalog) -> None:
    """Target has pre-existing NULL keys.  Source does not.  The anti-join
    correctly preserves the target NULL row (its key is not in the source
    key set), and the source rows are inserted alongside.  This matches the
    spec-compliant "merge by join columns" semantics: rows whose keys are
    not in source are left alone.
    """
    schema = Schema(
        NestedField(1, "key", StringType(), required=False),
        NestedField(2, "value", IntegerType(), required=True),
    )
    ident = "default.merge_preserves_target_null"
    _drop(catalog, ident)
    tbl = catalog.create_table(ident, schema=schema)

    target_arrow = pa.schema([pa.field("key", pa.large_string(), nullable=True), pa.field("value", pa.int32(), nullable=False)])
    tbl.append(
        pa.Table.from_pylist(
            [{"key": None, "value": 1}, {"key": "a", "value": 2}],
            schema=target_arrow,
        )
    )

    source_arrow = pa.schema([pa.field("key", pa.string(), nullable=True), pa.field("value", pa.int32(), nullable=False)])
    src = pa.Table.from_pylist(
        [{"key": "a", "value": 99}],  # all source non-null; target has NULL
        schema=source_arrow,
    )
    tbl.merge(src, join_cols=["key"])

    out = tbl.scan().to_arrow()
    rows = list(zip(out.column("key").to_pylist(), out.column("value").to_pylist(), strict=True))
    # NULL row preserved, "a" row replaced.
    assert (None, 1) in rows
    assert ("a", 99) in rows
    assert len(rows) == 2


def test_merge_target_null_check_only_runs_when_files_are_candidates(catalog: Catalog) -> None:
    """Target-NULL validation should only fire if the file containing
    NULL rows is a candidate for the merge. If source's keys don't
    overlap any file with target NULLs, the no-tasks early-return path
    is taken and target NULLs are never read - merge succeeds.

    This avoids penalizing callers whose target has NULLs in OTHER
    partitions/files that aren't touched by this merge.
    """
    schema = Schema(
        NestedField(1, "key", StringType(), required=False),
        NestedField(2, "value", IntegerType(), required=True),
    )
    ident = "default.merge_target_null_unrelated"
    _drop(catalog, ident)
    tbl = catalog.create_table(ident, schema=schema)

    target_arrow = pa.schema([pa.field("key", pa.large_string(), nullable=True), pa.field("value", pa.int32(), nullable=False)])
    # File 1: contains a NULL row that should NOT be touched by this merge.
    tbl.append(pa.Table.from_pylist([{"key": None, "value": 1}], schema=target_arrow))
    # File 2: contains "a", which the source will match.
    tbl.append(pa.Table.from_pylist([{"key": "a", "value": 2}], schema=target_arrow))

    source_arrow = pa.schema([pa.field("key", pa.string(), nullable=True), pa.field("value", pa.int32(), nullable=False)])
    src = pa.Table.from_pylist([{"key": "a", "value": 99}], schema=source_arrow)
    # Should succeed: only File 2 is a candidate; File 1 (with NULL) is
    # not read at all so the target-NULL check doesn't even see it.
    tbl.merge(src, join_cols=["key"])

    out = tbl.scan().to_arrow()
    rows = list(zip(out.column("key").to_pylist(), out.column("value").to_pylist(), strict=True))
    assert (None, 1) in rows  # File 1 untouched
    assert ("a", 99) in rows  # File 2 replaced
    assert len(rows) == 2


# ---- (14) Schema invariants: cast preserves nullable flag ------------------


def test_merge_drift_preserves_nullable_flag_after_cast(catalog: Catalog) -> None:
    """Nullable flags must be preserved by the drift cast. Optional
    iceberg fields produce nullable arrow fields; required produce
    non-nullable. Cast across these must not corrupt the merge.
    """
    schema = Schema(
        NestedField(1, "key", StringType(), required=True),
        NestedField(2, "tag", StringType(), required=False),
        NestedField(3, "value", IntegerType(), required=True),
    )
    ident = "default.merge_drift_nullable_flag"
    _drop(catalog, ident)
    tbl = catalog.create_table(ident, schema=schema)

    target_arrow = pa.schema(
        [
            pa.field("key", pa.large_string(), nullable=False),
            pa.field("tag", pa.large_string(), nullable=True),
            pa.field("value", pa.int32(), nullable=False),
        ]
    )
    source_arrow = pa.schema(
        [
            pa.field("key", pa.string(), nullable=False),
            pa.field("tag", pa.string(), nullable=True),
            pa.field("value", pa.int32(), nullable=False),
        ]
    )
    tbl.append(
        pa.Table.from_pylist(
            [{"key": "a", "tag": None, "value": 1}, {"key": "b", "tag": "x", "value": 2}],
            schema=target_arrow,
        )
    )
    tbl.merge(
        pa.Table.from_pylist(
            [{"key": "a", "tag": "updated", "value": 99}, {"key": "c", "tag": None, "value": 3}],
            schema=source_arrow,
        ),
        join_cols=["key"],
    )

    out = tbl.scan().to_arrow().sort_by("key")
    assert out.num_rows == 3
    rows = list(zip(out.column("key").to_pylist(), out.column("tag").to_pylist(), out.column("value").to_pylist(), strict=True))
    assert ("a", "updated", 99) in rows
    assert ("b", "x", 2) in rows
    assert ("c", None, 3) in rows


# ---- (15) Symmetric widen guarantee: source large value never narrowed ----


def test_merge_drift_symmetric_widen_does_not_truncate(catalog: Catalog) -> None:
    """Critical invariant: when target=string (small) and source=large_string
    contains a long value, the fix MUST widen target to large_string
    rather than narrow source to string. A naive one-sided cast (source
    -> target's small variant) would either truncate or raise; the
    symmetric unify-and-widen approach used here always succeeds.

    We use a 100K-char string (well under 2GB but plenty to confirm the
    cast doesn't truncate). The point isn't the size - it's that the
    fix is symmetric, so any source value that fits in `large_string`
    survives regardless of target's encoding.
    """
    schema = _str_schema()
    ident = "default.merge_drift_widen_no_truncate"
    _drop(catalog, ident)
    tbl = catalog.create_table(ident, schema=schema)

    tbl.append(
        pa.Table.from_pylist(
            [{"key": "small", "value": 1}],
            schema=_arrow_schema_with_key(pa.string()),
        )
    )

    long_value = "z" * 100_000
    src = pa.Table.from_pylist(
        [{"key": long_value, "value": 99}, {"key": "small", "value": 11}],
        schema=_arrow_schema_with_key(pa.large_string()),
    )
    assert pa.types.is_large_string(src.schema.field("key").type)

    tbl.merge(src, join_cols=["key"])

    out = tbl.scan().to_arrow()
    rows = dict(zip(out.column("key").to_pylist(), out.column("value").to_pylist(), strict=True))
    assert rows["small"] == 11
    assert rows[long_value] == 99
    assert len([k for k in rows if k == long_value][0]) == 100_000  # not truncated


# ---- (16) Source built from a RecordBatch (chunked layout) ----------------


def test_merge_drift_source_built_from_record_batch(catalog: Catalog) -> None:
    """A user may construct source via ``pa.Table.from_batches`` (e.g.,
    streaming producer). The chunked layout must still drift-cast cleanly.
    """
    schema = _str_schema()
    ident = "default.merge_drift_source_record_batch"
    _drop(catalog, ident)
    tbl = catalog.create_table(ident, schema=schema)

    tbl.append(
        pa.Table.from_pylist(
            [{"key": "a", "value": 1}, {"key": "b", "value": 2}],
            schema=_arrow_schema_with_key(pa.large_string()),
        )
    )

    src_schema = _arrow_schema_with_key(pa.string())
    batch1 = pa.RecordBatch.from_pylist([{"key": "a", "value": 99}], schema=src_schema)
    batch2 = pa.RecordBatch.from_pylist([{"key": "c", "value": 3}], schema=src_schema)
    src = pa.Table.from_batches([batch1, batch2])
    assert src.num_rows == 2

    tbl.merge(src, join_cols=["key"])

    out = tbl.scan().to_arrow().sort_by("key")
    assert out.num_rows == 3
    assert out.column("value").to_pylist() == [99, 2, 3]


# ---- (17) Atomicity on validation failure ---------------------------------


def test_merge_source_null_rejection_leaves_target_unchanged(catalog: Catalog) -> None:
    """When merge() rejects source NULL keys, target must be untouched -
    no partial commit, no orphaned files, no snapshot. Pin atomicity.
    """
    schema = Schema(
        NestedField(1, "key", StringType(), required=False),
        NestedField(2, "value", IntegerType(), required=True),
    )
    ident = "default.merge_atomic_source_null"
    _drop(catalog, ident)
    tbl = catalog.create_table(ident, schema=schema)

    initial = pa.Table.from_pylist(
        [{"key": "a", "value": 1}, {"key": "b", "value": 2}],
        schema=pa.schema([pa.field("key", pa.large_string(), nullable=True), pa.field("value", pa.int32(), nullable=False)]),
    )
    tbl.append(initial)
    snapshot_id_before = _current_snapshot_id(tbl)

    src = pa.Table.from_pylist(
        [{"key": None, "value": 99}, {"key": "a", "value": 88}],
        schema=pa.schema([pa.field("key", pa.string(), nullable=True), pa.field("value", pa.int32(), nullable=False)]),
    )
    with pytest.raises((TypeError, ValueError)):
        tbl.merge(src, join_cols=["key"])

    # No new snapshot created.
    tbl.refresh()
    assert _current_snapshot_id(tbl) == snapshot_id_before
    # Target data identical.
    out = tbl.scan().to_arrow().sort_by("key")
    assert out.num_rows == 2
    assert out.column("value").to_pylist() == [1, 2]


# ---- (18) Branch parameter with drift -------------------------------------


def test_merge_with_drift_to_non_main_branch(catalog: Catalog) -> None:
    """Merge with type drift to a non-main branch must work the same
    way as a main-branch merge - the drift fix and validation run
    regardless of branch.
    """
    schema = _str_schema()
    ident = "default.merge_drift_branch"
    _drop(catalog, ident)
    tbl = catalog.create_table(ident, schema=schema)

    tbl.append(
        pa.Table.from_pylist(
            [{"key": "a", "value": 1}, {"key": "b", "value": 2}],
            schema=_arrow_schema_with_key(pa.large_string()),
        )
    )
    # Create a new branch off main.
    with tbl.manage_snapshots() as ms:
        ms.create_branch(_current_snapshot_id(tbl), "feature_branch")

    # Merge with drift onto the new branch.
    tbl.merge(
        pa.Table.from_pylist(
            [{"key": "a", "value": 99}, {"key": "c", "value": 3}],
            schema=_arrow_schema_with_key(pa.string()),
        ),
        join_cols=["key"],
        branch="feature_branch",
    )

    # Branch has merged result.
    branch_out = tbl.scan(snapshot_id=tbl.refs()["feature_branch"].snapshot_id).to_arrow().sort_by("key")
    assert branch_out.num_rows == 3
    assert branch_out.column("value").to_pylist() == [99, 2, 3]
    # Main untouched.
    main_out = tbl.scan().to_arrow().sort_by("key")
    assert main_out.num_rows == 2
    assert main_out.column("value").to_pylist() == [1, 2]


# ---- (19) Source column order independence --------------------------------


def test_merge_drift_with_source_columns_in_different_order(catalog: Catalog) -> None:
    """Source df has the schema's columns in a DIFFERENT physical order
    than target. The drift cast must align by name, not by position.
    """
    schema = Schema(
        NestedField(1, "key", StringType(), required=True),
        NestedField(2, "name", StringType(), required=True),
        NestedField(3, "value", IntegerType(), required=True),
    )
    ident = "default.merge_drift_col_order"
    _drop(catalog, ident)
    tbl = catalog.create_table(ident, schema=schema)

    target_arrow = pa.schema(
        [
            pa.field("key", pa.large_string(), nullable=False),
            pa.field("name", pa.large_string(), nullable=False),
            pa.field("value", pa.int32(), nullable=False),
        ]
    )
    tbl.append(
        pa.Table.from_pylist(
            [{"key": "a", "name": "Alice", "value": 1}, {"key": "b", "name": "Bob", "value": 2}],
            schema=target_arrow,
        )
    )

    # Source: same columns, DIFFERENT order, drift on `key` and `name`.
    source_arrow = pa.schema(
        [
            pa.field("value", pa.int32(), nullable=False),  # value first
            pa.field("name", pa.string(), nullable=False),
            pa.field("key", pa.string(), nullable=False),  # key last
        ]
    )
    tbl.merge(
        pa.Table.from_pylist(
            [{"value": 99, "name": "Alice2", "key": "a"}, {"value": 3, "name": "Charlie", "key": "c"}],
            schema=source_arrow,
        ),
        join_cols=["key"],
    )

    out = tbl.scan().to_arrow().sort_by("key")
    rows = list(zip(out.column("key").to_pylist(), out.column("name").to_pylist(), out.column("value").to_pylist(), strict=True))
    assert ("a", "Alice2", 99) in rows
    assert ("b", "Bob", 2) in rows
    assert ("c", "Charlie", 3) in rows


# ---- (20) Long composite key (4+ cols, drift on subset) -------------------


def test_merge_drift_long_composite_key_partial_drift(catalog: Catalog) -> None:
    """Composite key with 4 columns. Drift on cols 1 and 3 (string),
    cols 2 and 4 stable (int + binary-which-also-drifts). Pins that the
    unify_schemas approach handles arbitrary-arity composite keys with
    interleaved drifting/non-drifting cols.
    """
    from pyiceberg.types import BinaryType

    schema = Schema(
        NestedField(1, "k1", StringType(), required=True),
        NestedField(2, "k2", IntegerType(), required=True),
        NestedField(3, "k3", StringType(), required=True),
        NestedField(4, "k4", BinaryType(), required=True),
        NestedField(5, "value", IntegerType(), required=True),
    )
    ident = "default.merge_drift_4col_key"
    _drop(catalog, ident)
    tbl = catalog.create_table(ident, schema=schema)

    target_arrow = pa.schema(
        [
            pa.field("k1", pa.large_string(), nullable=False),
            pa.field("k2", pa.int32(), nullable=False),
            pa.field("k3", pa.large_string(), nullable=False),
            pa.field("k4", pa.large_binary(), nullable=False),
            pa.field("value", pa.int32(), nullable=False),
        ]
    )
    source_arrow = pa.schema(
        [
            pa.field("k1", pa.string(), nullable=False),
            pa.field("k2", pa.int32(), nullable=False),
            pa.field("k3", pa.string(), nullable=False),
            pa.field("k4", pa.binary(), nullable=False),
            pa.field("value", pa.int32(), nullable=False),
        ]
    )
    tbl.append(
        pa.Table.from_pylist(
            [
                {"k1": "x", "k2": 1, "k3": "p", "k4": b"\x01", "value": 100},
                {"k1": "y", "k2": 2, "k3": "q", "k4": b"\x02", "value": 200},
            ],
            schema=target_arrow,
        )
    )
    tbl.merge(
        pa.Table.from_pylist(
            [
                {"k1": "x", "k2": 1, "k3": "p", "k4": b"\x01", "value": 999},  # exact match -> replace
                {"k1": "z", "k2": 3, "k3": "r", "k4": b"\x03", "value": 300},  # new
            ],
            schema=source_arrow,
        ),
        join_cols=["k1", "k2", "k3", "k4"],
    )

    out = tbl.scan().to_arrow().sort_by("k1")
    assert out.num_rows == 3
    rows_by_value = sorted(out.column("value").to_pylist())
    assert rows_by_value == [200, 300, 999]


# ---- (21) Whitespace and case sensitivity in string keys ------------------


def test_merge_drift_treats_whitespace_and_case_as_significant(catalog: Catalog) -> None:
    """String join keys are byte-for-byte equal: leading/trailing
    whitespace and letter case must NOT be normalized by the drift
    cast. ``" a"`` ≠ ``"a "`` ≠ ``"a"`` ≠ ``"A"``.
    """
    schema = _str_schema()
    ident = "default.merge_drift_whitespace_case"
    _drop(catalog, ident)
    tbl = catalog.create_table(ident, schema=schema)

    tbl.append(
        pa.Table.from_pylist(
            [
                {"key": "a", "value": 1},
                {"key": " a", "value": 2},  # leading space
                {"key": "a ", "value": 3},  # trailing space
                {"key": "A", "value": 4},  # different case
            ],
            schema=_arrow_schema_with_key(pa.large_string()),
        )
    )
    tbl.merge(
        pa.Table.from_pylist(
            [{"key": "a", "value": 99}],  # only the exact "a" matches
            schema=_arrow_schema_with_key(pa.string()),
        ),
        join_cols=["key"],
    )

    out = tbl.scan().to_arrow()
    rows = dict(zip(out.column("key").to_pylist(), out.column("value").to_pylist(), strict=True))
    assert rows["a"] == 99  # replaced
    assert rows[" a"] == 2  # untouched
    assert rows["a "] == 3  # untouched
    assert rows["A"] == 4  # untouched
    assert len(rows) == 4


# ---- (22) Float join key with NaN handling --------------------------------


def test_merge_rejects_nan_in_source_float_join_key(catalog: Catalog) -> None:
    """Source NaN in a float join column is rejected by pyiceberg's ``In``
    predicate constructor (which forbids NaN literals).  ``NaN == NaN`` is
    false per IEEE 754, so NaN can never be a valid join key.  We rely on
    pyiceberg-native rejection here instead of an explicit check.
    """
    from pyiceberg.types import DoubleType

    schema = Schema(
        NestedField(1, "key", DoubleType(), required=True),
        NestedField(2, "value", IntegerType(), required=True),
    )
    ident = "default.merge_reject_source_nan"
    _drop(catalog, ident)
    tbl = catalog.create_table(ident, schema=schema)

    target_arrow = pa.schema([pa.field("key", pa.float64(), nullable=False), pa.field("value", pa.int32(), nullable=False)])
    tbl.append(
        pa.Table.from_pylist(
            [{"key": 1.5, "value": 1}, {"key": 2.5, "value": 2}],
            schema=target_arrow,
        )
    )

    with pytest.raises((TypeError, ValueError)):
        tbl.merge(
            pa.Table.from_pylist(
                [{"key": 1.5, "value": 99}, {"key": float("nan"), "value": 88}],
                schema=target_arrow,
            ),
            join_cols=["key"],
        )

    # Target unchanged.
    assert tbl.scan().to_arrow().num_rows == 2


def test_merge_preserves_nan_in_target_float_join_key(catalog: Catalog) -> None:
    """Target has a pre-existing NaN row.  Source does not.  The anti-join
    correctly preserves the target NaN row (its key cannot match anything
    in the source key set), and the source rows are inserted alongside.
    """
    from pyiceberg.types import DoubleType

    schema = Schema(
        NestedField(1, "key", DoubleType(), required=True),
        NestedField(2, "value", IntegerType(), required=True),
    )
    ident = "default.merge_preserves_target_nan"
    _drop(catalog, ident)
    tbl = catalog.create_table(ident, schema=schema)

    target_arrow = pa.schema([pa.field("key", pa.float64(), nullable=False), pa.field("value", pa.int32(), nullable=False)])
    # Target file has 1.5, NaN, 2.5; source touches 1.5 -> file is candidate.
    tbl.append(
        pa.Table.from_pylist(
            [{"key": 1.5, "value": 1}, {"key": float("nan"), "value": 2}, {"key": 2.5, "value": 3}],
            schema=target_arrow,
        )
    )

    tbl.merge(
        pa.Table.from_pylist(
            [{"key": 1.5, "value": 99}],  # source clean
            schema=target_arrow,
        ),
        join_cols=["key"],
    )

    out = tbl.scan().to_arrow()
    vals_by_key = {
        (k if k == k else "nan"): v for k, v in zip(out.column("key").to_pylist(), out.column("value").to_pylist(), strict=True)
    }
    assert vals_by_key[1.5] == 99  # replaced
    assert vals_by_key[2.5] == 3  # preserved (not in source key set)
    assert vals_by_key["nan"] == 2  # preserved
    assert out.num_rows == 3


def test_merge_with_clean_float_join_key(catalog: Catalog) -> None:
    """Float join keys without NaN/NULL work normally. Pins that the
    rejection logic doesn't false-positive on regular floats.
    """
    from pyiceberg.types import DoubleType

    schema = Schema(
        NestedField(1, "key", DoubleType(), required=True),
        NestedField(2, "value", IntegerType(), required=True),
    )
    ident = "default.merge_clean_float_key"
    _drop(catalog, ident)
    tbl = catalog.create_table(ident, schema=schema)

    target_arrow = pa.schema([pa.field("key", pa.float64(), nullable=False), pa.field("value", pa.int32(), nullable=False)])
    tbl.append(
        pa.Table.from_pylist(
            [{"key": 1.5, "value": 1}, {"key": 2.5, "value": 2}],
            schema=target_arrow,
        )
    )
    tbl.merge(
        pa.Table.from_pylist(
            [{"key": 1.5, "value": 99}, {"key": 3.5, "value": 3}],
            schema=target_arrow,
        ),
        join_cols=["key"],
    )

    out = tbl.scan().to_arrow().sort_by("key")
    assert out.num_rows == 3
    assert out.column("value").to_pylist() == [99, 2, 3]


# ---- (23) Source df from concat_tables (re-chunked layout) ----------------


def test_merge_drift_with_source_from_concat_tables(catalog: Catalog) -> None:
    """Source built via ``pa.concat_tables`` has potentially-different
    chunking than a single from_pylist. Pin that the drift cast
    survives this layout, since real callers commonly merge
    multi-source data via concat.
    """
    schema = _str_schema()
    ident = "default.merge_drift_source_concat"
    _drop(catalog, ident)
    tbl = catalog.create_table(ident, schema=schema)

    tbl.append(
        pa.Table.from_pylist(
            [{"key": "a", "value": 1}, {"key": "b", "value": 2}],
            schema=_arrow_schema_with_key(pa.large_string()),
        )
    )

    src_schema = _arrow_schema_with_key(pa.string())
    part1 = pa.Table.from_pylist([{"key": "a", "value": 99}], schema=src_schema)
    part2 = pa.Table.from_pylist([{"key": "c", "value": 3}, {"key": "d", "value": 4}], schema=src_schema)
    src = pa.concat_tables([part1, part2])
    assert src.num_rows == 3

    tbl.merge(src, join_cols=["key"])

    out = tbl.scan().to_arrow().sort_by("key")
    assert out.num_rows == 4
    assert out.column("value").to_pylist() == [99, 2, 3, 4]


# ---- (24) Merge run twice in same transaction (drift + drift) -------------


def test_merge_drift_in_same_transaction_twice(catalog: Catalog) -> None:
    """Two merges in a single transaction, both with type drift. Pins
    that the drift fix isn't accidentally per-table-state - it must
    work regardless of how many drift-handled merges precede it within
    a transaction.
    """
    schema = _str_schema()
    ident = "default.merge_drift_two_in_tx"
    _drop(catalog, ident)
    tbl = catalog.create_table(ident, schema=schema)

    tbl.append(
        pa.Table.from_pylist(
            [{"key": "a", "value": 1}, {"key": "b", "value": 2}],
            schema=_arrow_schema_with_key(pa.large_string()),
        )
    )

    with tbl.transaction() as tx:
        tx.merge(
            pa.Table.from_pylist(
                [{"key": "a", "value": 99}, {"key": "c", "value": 3}],
                schema=_arrow_schema_with_key(pa.string()),
            ),
            join_cols=["key"],
        )
        tx.merge(
            pa.Table.from_pylist(
                [{"key": "b", "value": 88}, {"key": "d", "value": 4}],
                schema=_arrow_schema_with_key(pa.string()),
            ),
            join_cols=["key"],
        )

    out = tbl.scan().to_arrow().sort_by("key")
    assert out.num_rows == 4
    assert out.column("value").to_pylist() == [99, 88, 3, 4]


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

    assert (result.rows_deleted, result.rows_inserted) == (0, 0)
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

    assert (result.rows_deleted, result.rows_inserted) == (0, 2)
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

    assert (result.rows_deleted, result.rows_inserted) == (3, 3)
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

    assert (result.rows_deleted, result.rows_inserted) == (2, 3)
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

    assert (result.rows_deleted, result.rows_inserted) == (3, 1)
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

    assert (result.rows_deleted, result.rows_inserted) == (1, 3)
    assert tbl.scan().to_arrow().num_rows == 3
