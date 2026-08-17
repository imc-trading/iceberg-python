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
from pathlib import PosixPath
from typing import Any, List, Set

import pyarrow as pa
import pytest

from pyiceberg.exceptions import CommitFailedException, CommitStateUnknownException
from pyiceberg.expressions import EqualTo
from pyiceberg.io.pyarrow import _dataframe_to_data_files
from pyiceberg.schema import Schema
from pyiceberg.table import Table, Transaction
from pyiceberg.table.metadata import COMMIT_NUM_RETRIES
from pyiceberg.types import IntegerType, NestedField, StringType
from tests.catalog.test_base import InMemoryCatalog

SCHEMA = Schema(
    NestedField(1, "id", IntegerType(), required=True),
    NestedField(2, "name", StringType(), required=True),
)

ARROW_SCHEMA = pa.schema([pa.field("id", pa.int32(), nullable=False), pa.field("name", pa.string(), nullable=False)])


@pytest.fixture
def warehouse(tmp_path: PosixPath) -> PosixPath:
    return tmp_path


@pytest.fixture
def table(warehouse: PosixPath) -> Table:
    catalog = InMemoryCatalog("test.in_memory.catalog", warehouse=warehouse.absolute().as_posix())
    catalog.create_namespace("default")
    # A single attempt per commit, so a test asserting on one failed attempt is not asserting on five.
    return catalog.create_table("default.cleanup", schema=SCHEMA, properties={COMMIT_NUM_RETRIES: "0"})


def _rows(*ids: int) -> pa.Table:
    return pa.table({"id": list(ids), "name": [f"row-{i}" for i in ids]}, schema=ARROW_SCHEMA)


def _parquet_files(warehouse: PosixPath) -> Set[str]:
    return {path.name for path in warehouse.rglob("*.parquet")}


def _fail_commit_with(monkeypatch: pytest.MonkeyPatch, error: Exception) -> None:
    def _do_commit(self: Table, **kwargs: Any) -> None:
        raise error

    monkeypatch.setattr(Table, "_do_commit", _do_commit)


def test_rejected_append_deletes_the_data_file_it_wrote(
    table: Table, warehouse: PosixPath, monkeypatch: pytest.MonkeyPatch
) -> None:
    _fail_commit_with(monkeypatch, CommitFailedException("rejected"))

    with pytest.raises(CommitFailedException):
        table.append(_rows(1, 2))

    assert _parquet_files(warehouse) == set()


def test_rejected_copy_on_write_delete_deletes_the_rewritten_file(
    table: Table, warehouse: PosixPath, monkeypatch: pytest.MonkeyPatch
) -> None:
    table.append(_rows(1, 2))
    committed = _parquet_files(warehouse)
    _fail_commit_with(monkeypatch, CommitFailedException("rejected"))

    with pytest.raises(CommitFailedException):
        table.delete(EqualTo("id", 1))

    assert _parquet_files(warehouse) == committed, (
        "the rewritten survivors of a rejected delete are referenced by no snapshot and must not be left behind"
    )


def test_unknown_commit_state_keeps_the_data_file(table: Table, warehouse: PosixPath, monkeypatch: pytest.MonkeyPatch) -> None:
    _fail_commit_with(monkeypatch, CommitStateUnknownException("no idea"))

    with pytest.raises(CommitStateUnknownException):
        table.append(_rows(1, 2))

    assert _parquet_files(warehouse), (
        "the commit may have landed, so the file may be live table data — leave it for orphan cleanup to age out"
    )


def test_rejected_commit_keeps_data_files_the_caller_supplied(
    table: Table, warehouse: PosixPath, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Writers that write once and re-append across commit attempts own their files; the transaction
    must not delete them out from under the next attempt.
    """
    data_files: List[Any] = list(_dataframe_to_data_files(table_metadata=table.metadata, df=_rows(1, 2), io=table.io))
    supplied = _parquet_files(warehouse)
    _fail_commit_with(monkeypatch, CommitFailedException("rejected"))

    transaction = Transaction(table, autocommit=False)
    with transaction.update_snapshot().fast_append() as append_files:
        for data_file in data_files:
            append_files.append_data_file(data_file)
    with pytest.raises(CommitFailedException):
        transaction.commit_transaction()

    assert _parquet_files(warehouse) == supplied


def test_abandoned_transaction_deletes_the_data_file_it_wrote(table: Table, warehouse: PosixPath) -> None:
    with pytest.raises(RuntimeError):
        with table.transaction() as transaction:
            transaction.append(_rows(1, 2))
            raise RuntimeError("caller gave up")

    assert _parquet_files(warehouse) == set()
