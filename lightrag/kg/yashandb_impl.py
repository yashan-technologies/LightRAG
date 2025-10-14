import array
import asyncio
import configparser
import datetime
import json
import os
import re
from dataclasses import dataclass
from datetime import timezone
from typing import (
    Any,
    Collection,
    Literal,
    Optional,
    TypeVar,
    Union,
    cast,
    final,
    overload,
)

import numpy as np
import yasdb
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)
from yasdb.cursor import YasdbCursor

from ..base import (
    BaseGraphStorage,
    BaseKVStorage,
    BaseVectorStorage,
    DocProcessingStatus,
    DocStatus,
    DocStatusStorage,
)
from ..constants import GRAPH_FIELD_SEP
from ..namespace import NameSpace, is_namespace
from ..types import KnowledgeGraph, KnowledgeGraphEdge, KnowledgeGraphNode
from ..utils import logger
from .shared_storage import get_data_init_lock, get_storage_lock, get_graph_db_lock

T = TypeVar("T")

_SingleQuery = dict[str, Any]


@dataclass
class YashanDBConfig:
    user: str
    password: str
    host: str = "localhost"
    port: int = 1688

    # NOTE: This is for sharing the same table with multiple lightrag instances
    workspace: str = "default"

    # vdb related configs
    vector_idx_type: str = "hnsw"
    hnsw_m: int = 16
    hnsw_ef: int = 200


@dataclass
class YashanDB:
    config: YashanDBConfig
    _conn: Optional[yasdb.YasdbConnection] = None

    @property
    def workspace(self):
        return self.config.workspace

    @property
    def vector_idx_type(self):
        return self.config.vector_idx_type

    @property
    def hnsw_m(self):
        return self.config.hnsw_m

    @property
    def hnsw_ef(self):
        return self.config.hnsw_ef

    def initdb(self):
        if self._conn:
            self._conn.close()

        self._conn = yasdb.connect(
            user=self.config.user,
            password=self.config.password,
            host=self.config.host,
            port=self.config.port,
            autocommit=True,
        )

    def close(self):
        if self._conn:
            self._conn.close()

    def cursor(self):
        assert self._conn
        return self._conn.cursor()

    @overload
    async def query(
        self,
        query: str,
        params: Optional[dict[str, Any]] = None,
        multirows: Literal[False] = False,
    ) -> Optional[_SingleQuery]: ...

    @overload
    async def query(
        self,
        query: str,
        params: Optional[dict[str, Any]] = None,
        multirows: Literal[True] = True,
    ) -> list[_SingleQuery]: ...

    async def query(
        self,
        query: str,
        params: Optional[dict[str, Any]] = None,
        multirows: bool = False,
        # with_age: bool = False,
        # graph_name: str | None = None,
    ):
        def get_columns(c: YasdbCursor) -> list[str]:
            assert c.description is not None
            return [cast(str, d[0]).lower() for d in c.description]

        try:
            with self.cursor() as c:
                c.execute(query, params or {})
                rows: list[Any] = c.fetchall()
                if multirows:
                    if not rows:
                        return []

                    assert c.description is not None
                    columns = get_columns(c)
                    return [dict(zip(columns, row)) for row in rows]

                if not rows:
                    return None

                columns = get_columns(c)
                return dict(zip(columns, rows[0]))
        except Exception:
            # TODO: log error
            raise

    async def execute(
        self,
        sql: str,
        data: Optional[dict[str, Any]] = None,
        # upsert: bool = False,
        ignore_if_exists: bool = False,
    ):
        try:
            with self.cursor() as c:
                c.execute(sql, data or {})
        except yasdb.DatabaseError as e:
            pat = re.compile(r"YAS-\d{5}")
            if not ignore_if_exists or not pat.search(str(e)):
                raise

    async def check_tables(self):
        # First create all tables
        for k, v in TABLES.items():
            try:
                query = "select 1 from user_tables where table_name = :table"
                rows = await self.query(query, {"table": k.upper()})
                if not rows:
                    # table not found
                    logger.info(f"YashanDB, Try Creating table {k}")
                    await self.execute(v["ddl"])
                    logger.info(f"YashanDB, Creation success table {k}")
            except Exception as e:
                logger.error(
                    f"YashanDB, Failed to verify or create table {k}. "
                    f"Please verify the connection with YashanDB, Got: {e}"
                )
                raise

        # Then create vector indexes (HNSW)
        try:
            await self._create_hnsw_vector_indexes()
        except Exception as e:
            logger.error(
                f"YashanDB, Failed to create vector index, type: HNSW, Got: {e}"
            )

    async def _create_hnsw_vector_indexes(self):
        vdb_tables = [
            "LIGHTRAG_VDB_CHUNKS",
            "LIGHTRAG_VDB_ENTITY",
            "LIGHTRAG_VDB_RELATION",
        ]

        for k in vdb_tables:
            vector_index_name = f"idx_{k.lower()}_hnsw_cosine"
            check_vector_index_sql = """
                SELECT 1 FROM user_indexes
                WHERE index_name = :idx_name
                AND table_name = :tbl_name
                """
            try:
                vector_index_exists = await self.query(
                    check_vector_index_sql,
                    {"idx_name": vector_index_name.upper(), "tbl_name": k.upper()},
                )

                if vector_index_exists is None:
                    create_vector_index_sql = f"""
                            CREATE VECTOR INDEX {vector_index_name}
                            ON {k}(content_vector)
                            ORGANIZATION NEIGHBOR GRAPH WITH DISTANCE COSINE
                            PARAMETERS(TYPE HNSW, M {self.hnsw_m}, EFCONSTRUCTION {self.hnsw_ef})
                        """
                    logger.info(f"Creating hnsw index {vector_index_name} on table {k}")
                    data = {"m": self.hnsw_m, "ef": self.hnsw_ef}
                    data = None
                    await self.execute(create_vector_index_sql, data)
                    logger.info(
                        f"Successfully created vector index {vector_index_name} on table {k}"
                    )
                else:
                    logger.info(
                        f"HNSW vector index {vector_index_name} already exists on table {k}"
                    )
            except Exception as e:
                logger.error(f"Failed to create vector index on table {k}, Got: {e}")


class ClientManager:
    _lock = asyncio.Lock()
    _db: Optional[YashanDB] = None
    _ref: int = 0

    @classmethod
    def get_config(cls):
        config = configparser.ConfigParser()
        config.read("config.ini", "utf-8")
        env = os.environ.get

        def env_int(env_name: str, default: int) -> int:
            try:
                value = env(env_name, None)
                assert value is not None  # go to except branch
                return int(value)
            except Exception:
                return default

        return YashanDBConfig(
            host=env("YASHANDB_HOST", "localhost"),
            port=env_int("YASHANDB_PORT", 1688),
            user=env("YASHANDB_USER", "yashan"),
            password=env("YASHANDB_PASSWORD", "yashan"),
            workspace=env("YASHANDB_WORKSPACE", "default"),
            # vdb
            vector_idx_type=env("YASHANDB_VECTOR_INDEX_TYPE", "hnsw"),
            hnsw_m=env_int("YASHANDB_HNSW_M", 16),
            hnsw_ef=env_int("YASHANDB_HNSW_EF", 200),
        )

    @classmethod
    async def get_client(cls) -> YashanDB:
        async with cls._lock:
            if cls._db is None:
                cfg = cls.get_config()
                db = YashanDB(cfg)
                db.initdb()
                await db.check_tables()
                cls._db = db
                cls._ref = 0
            cls._ref += 1
            return cls._db

    @classmethod
    async def release_client(cls, db: YashanDB):
        async with cls._lock:
            assert db is not None
            if db is cls._db:
                cls._ref -= 1
                if cls._ref == 0:
                    db.close()
                    # log db closing
                    cls._db = None
            else:
                db.close()


# Note: Order matters! More specific namespaces (e.g., "full_entities") must come before
# more general ones (e.g., "entities") because is_namespace() uses endswith() matching
NAMESPACE_TABLE_MAP = {
    NameSpace.KV_STORE_FULL_DOCS: "LIGHTRAG_DOC_FULL",
    NameSpace.KV_STORE_TEXT_CHUNKS: "LIGHTRAG_DOC_CHUNKS",
    NameSpace.KV_STORE_FULL_ENTITIES: "LIGHTRAG_FULL_ENTITIES",
    NameSpace.KV_STORE_FULL_RELATIONS: "LIGHTRAG_FULL_RELATIONS",
    NameSpace.KV_STORE_LLM_RESPONSE_CACHE: "LIGHTRAG_LLM_CACHE",
    NameSpace.VECTOR_STORE_CHUNKS: "LIGHTRAG_VDB_CHUNKS",
    NameSpace.VECTOR_STORE_ENTITIES: "LIGHTRAG_VDB_ENTITY",
    NameSpace.VECTOR_STORE_RELATIONSHIPS: "LIGHTRAG_VDB_RELATION",
    NameSpace.DOC_STATUS: "LIGHTRAG_DOC_STATUS",
}


def namespace_to_table_name(namespace: str) -> str:
    for k, v in NAMESPACE_TABLE_MAP.items():
        if is_namespace(namespace, k):
            return v
    assert False


def ids_format_query(ids: Collection[str]) -> str:
    """Format ids and return a string Q that fits the SQL where clause: `WHERE id = {Q}`"""
    match len(ids):
        case 0:
            return "NULL"  # "WHERE id = NULL" is never true.
        case 1:
            return f"'{list(ids)[0]}'"  # WHERE id = id1
        case _:
            return f"ANY{tuple(ids)}"  # WHERE id = ANY(id1, id2, id3)


EMBEDDING_DIM = os.environ.get("EMBEDDING_DIM", "1024")

TABLES = {
    "LIGHTRAG_DOC_FULL": {
        "ddl": """CREATE TABLE LIGHTRAG_DOC_FULL (
                    id VARCHAR(255 CHAR),
                    workspace VARCHAR(255 CHAR),
                    doc_name VARCHAR(1024 CHAR),
                    content CLOB,
                    meta JSON,
                    create_time TIMESTAMP(0) DEFAULT CURRENT_TIMESTAMP,
                    update_time TIMESTAMP(0) DEFAULT CURRENT_TIMESTAMP,
                    CONSTRAINT LIGHTRAG_DOC_FULL_PK PRIMARY KEY (workspace, id)
                    )"""
    },
    "LIGHTRAG_DOC_CHUNKS": {
        "ddl": """CREATE TABLE LIGHTRAG_DOC_CHUNKS (
                    id VARCHAR(255 CHAR),
                    workspace VARCHAR(255 CHAR),
                    full_doc_id VARCHAR(256 CHAR),
                    chunk_order_index INTEGER,
                    tokens INTEGER,
                    content CLOB,
                    file_path VARCHAR(65534) NULL,
                    llm_cache_list JSON NULL DEFAULT '[]',
                    create_time TIMESTAMP(0) DEFAULT CURRENT_TIMESTAMP,
                    update_time TIMESTAMP(0) DEFAULT CURRENT_TIMESTAMP,
                    CONSTRAINT LIGHTRAG_DOC_CHUNKS_PK PRIMARY KEY (workspace, id)
                    )"""
    },
    "LIGHTRAG_VDB_CHUNKS": {
        "ddl": f"""CREATE TABLE LIGHTRAG_VDB_CHUNKS (
                    id VARCHAR(255 CHAR),
                    workspace VARCHAR(255 CHAR),
                    full_doc_id VARCHAR(256 CHAR),
                    chunk_order_index INTEGER,
                    tokens INTEGER,
                    content CLOB,
                    content_vector VECTOR({EMBEDDING_DIM}),
                    file_path VARCHAR(65534) NULL,
                    create_time TIMESTAMP(0) DEFAULT CURRENT_TIMESTAMP,
                    update_time TIMESTAMP(0) DEFAULT CURRENT_TIMESTAMP,
                    CONSTRAINT LIGHTRAG_VDB_CHUNKS_PK PRIMARY KEY (workspace, id)
                    )"""
    },
    "LIGHTRAG_VDB_ENTITY": {
        "ddl": f"""CREATE TABLE LIGHTRAG_VDB_ENTITY (
                    id VARCHAR(255 CHAR),
                    workspace VARCHAR(255 CHAR),
                    entity_name VARCHAR(512 CHAR),
                    content CLOB,
                    content_vector VECTOR({EMBEDDING_DIM}),
                    create_time TIMESTAMP(0) DEFAULT CURRENT_TIMESTAMP,
                    update_time TIMESTAMP(0) DEFAULT CURRENT_TIMESTAMP,
                    chunk_ids JSON NULL,
                    file_path VARCHAR(65534) NULL,
                    CONSTRAINT LIGHTRAG_VDB_ENTITY_PK PRIMARY KEY (workspace, id)
                    )"""
    },
    "LIGHTRAG_VDB_RELATION": {
        "ddl": f"""CREATE TABLE LIGHTRAG_VDB_RELATION (
                    id VARCHAR(255 CHAR),
                    workspace VARCHAR(255 CHAR),
                    source_id VARCHAR(512 CHAR),
                    target_id VARCHAR(512 CHAR),
                    content CLOB,
                    content_vector VECTOR({EMBEDDING_DIM}),
                    create_time TIMESTAMP(0) DEFAULT CURRENT_TIMESTAMP,
                    update_time TIMESTAMP(0) DEFAULT CURRENT_TIMESTAMP,
                    chunk_ids JSON NULL,
                    file_path VARCHAR(65534) NULL,
                    CONSTRAINT LIGHTRAG_VDB_RELATION_PK PRIMARY KEY (workspace, id)
                    )"""
    },
    "LIGHTRAG_LLM_CACHE": {
        "ddl": """CREATE TABLE LIGHTRAG_LLM_CACHE (
                    workspace varchar(255 CHAR) NOT NULL,
                    id varchar(255 CHAR) NOT NULL,
                    original_prompt CLOB,
                    return_value CLOB,
                    chunk_id VARCHAR(255 CHAR) NULL,
                    cache_type VARCHAR(32 CHAR),
                    queryparam JSON NULL,
                    create_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    update_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    CONSTRAINT LIGHTRAG_LLM_CACHE_PK PRIMARY KEY (workspace, id)
                    )"""
    },
    "LIGHTRAG_DOC_STATUS": {
        "ddl": """CREATE TABLE LIGHTRAG_DOC_STATUS (
                workspace varchar(255 CHAR) NOT NULL,
                id varchar(255 CHAR) NOT NULL,
                content_summary varchar(255 CHAR) NULL,
                content_length INTEGER NULL,
                chunks_count INTEGER NULL,
                status varchar(64 CHAR) NULL,
                file_path VARCHAR(65534) NULL,
                chunks_list JSON NULL DEFAULT '[]',
                track_id varchar(255 CHAR) NULL,
                metadata JSON NULL DEFAULT '{}',
                error_msg CLOB NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                CONSTRAINT LIGHTRAG_DOC_STATUS_PK PRIMARY KEY (workspace, id)
                )"""
    },
    "LIGHTRAG_FULL_ENTITIES": {
        "ddl": """CREATE TABLE LIGHTRAG_FULL_ENTITIES (
                    id VARCHAR(255 CHAR),
                    workspace VARCHAR(255 CHAR),
                    entity_names JSON,
                    count INTEGER,
                    create_time TIMESTAMP(0) DEFAULT CURRENT_TIMESTAMP,
                    update_time TIMESTAMP(0) DEFAULT CURRENT_TIMESTAMP,
                    CONSTRAINT LIGHTRAG_FULL_ENTITIES_PK PRIMARY KEY (workspace, id)
                    )"""
    },
    "LIGHTRAG_FULL_RELATIONS": {
        "ddl": """CREATE TABLE LIGHTRAG_FULL_RELATIONS (
                    id VARCHAR(255 CHAR),
                    workspace VARCHAR(255 CHAR),
                    relation_pairs JSON,
                    count INTEGER,
                    create_time TIMESTAMP(0) DEFAULT CURRENT_TIMESTAMP,
                    update_time TIMESTAMP(0) DEFAULT CURRENT_TIMESTAMP,
                    CONSTRAINT LIGHTRAG_FULL_RELATIONS_PK PRIMARY KEY (workspace, id)
                    )"""
    },
}


SQL_COMMON_TEMPLATES = {
    "filter_keys": "SELECT id FROM {table_name} WHERE workspace=:workspace AND id IN ({ids})",
    "drop_specific_table_workspace": """
        DELETE FROM {table_name} WHERE workspace=:workspace
       """,
}

#######################################################################################################################
# YashanDB as KV storage
#######################################################################################################################


def extract_epoch_seconds_sql(timestamp_column: str) -> str:
    return f"""(
        with t as (select ({timestamp_column})-(timestamp '1970-01-01') AS diff from dual)
        select cast(
            extract(day from diff) * 86400 +
            extract(hour from diff) * 3600 +
            extract(minute from diff) * 60 +
            extract(second from diff)
        as bigint) from t
    )"""


SQL_KV_TEMPLATES = {
    "get_by_id_full_docs": """SELECT id, content
                                FROM LIGHTRAG_DOC_FULL WHERE workspace=:workspace AND id=:id
                            """,
    "get_by_id_text_chunks": f"""SELECT id, tokens, content,
                                chunk_order_index, full_doc_id, file_path,
                                COALESCE(llm_cache_list, JSON('[]')) as llm_cache_list,
                                {extract_epoch_seconds_sql("create_time")} as create_time,
                                {extract_epoch_seconds_sql("update_time")} as update_time
                                FROM LIGHTRAG_DOC_CHUNKS WHERE workspace=:workspace AND id=:id
                            """,
    "get_by_id_llm_response_cache": f"""SELECT id, original_prompt, return_value, chunk_id, cache_type, queryparam,
                                {extract_epoch_seconds_sql("create_time")} as create_time,
                                {extract_epoch_seconds_sql("update_time")} as update_time
                                FROM LIGHTRAG_LLM_CACHE WHERE workspace=:workspace AND id=:id
                            """,
    "get_by_ids_full_docs": """SELECT id, content
                                FROM LIGHTRAG_DOC_FULL WHERE workspace=:workspace AND id IN ({ids})
                            """,
    "get_by_ids_text_chunks": f"""SELECT id, tokens, content,
                                chunk_order_index, full_doc_id, file_path,
                                COALESCE(llm_cache_list, JSON('[]')) as llm_cache_list,
                                {extract_epoch_seconds_sql("create_time")} as create_time,
                                {extract_epoch_seconds_sql("update_time")} as update_time
                                FROM LIGHTRAG_DOC_CHUNKS WHERE workspace=:workspace AND id IN ({{ids}})
                                """,
    "get_by_ids_llm_response_cache": f"""SELECT id, original_prompt, return_value, chunk_id, cache_type, queryparam,
                                {extract_epoch_seconds_sql("create_time")} as create_time,
                                {extract_epoch_seconds_sql("update_time")} as update_time
                                FROM LIGHTRAG_LLM_CACHE WHERE workspace=:workspace AND id IN ({{ids}})
                                """,
    "get_by_id_full_entities": f"""SELECT id, entity_names, count,
                                {extract_epoch_seconds_sql("create_time")} as create_time,
                                {extract_epoch_seconds_sql("update_time")} as update_time
                                FROM LIGHTRAG_FULL_ENTITIES WHERE workspace=:workspace AND id=:id
                            """,
    "get_by_id_full_relations": f"""SELECT id, relation_pairs, count,
                                {extract_epoch_seconds_sql("create_time")} as create_time,
                                {extract_epoch_seconds_sql("update_time")} as update_time
                                FROM LIGHTRAG_FULL_RELATIONS WHERE workspace=:workspace AND id=:id
                            """,
    "get_by_ids_full_entities": f"""SELECT id, entity_names, count,
                                {extract_epoch_seconds_sql("create_time")} as create_time,
                                {extract_epoch_seconds_sql("update_time")} as update_time
                                FROM LIGHTRAG_FULL_ENTITIES WHERE workspace=:workspace AND id IN ({{ids}})
                                """,
    "get_by_ids_full_relations": f"""SELECT id, relation_pairs, count,
                                {extract_epoch_seconds_sql("create_time")} as create_time,
                                {extract_epoch_seconds_sql("update_time")} as update_time
                                FROM LIGHTRAG_FULL_RELATIONS workspace=:workspace AND id IN ({{ids}})
                                """,
    "upsert_doc_full": """INSERT INTO LIGHTRAG_DOC_FULL (id, content, workspace)
                        VALUES (:id,:content,:workspace)
                        ON DUPLICATE KEY
                        UPDATE content = VALUES(content), update_time = CURRENT_TIMESTAMP
                    """,
    "upsert_llm_response_cache": """INSERT INTO LIGHTRAG_LLM_CACHE(workspace,id,original_prompt,return_value,chunk_id,cache_type,queryparam)
                                    VALUES (:workspace,:id,:original_prompt,:return_value,:chunk_id,:cache_type,:queryparam)
                                    ON DUPLICATE KEY
                                    UPDATE original_prompt = VALUES(original_prompt),
                                    return_value=VALUES(return_value),
                                    chunk_id=VALUES(chunk_id),
                                    cache_type=VALUES(cache_type),
                                    queryparam=VALUES(queryparam),
                                    update_time = CURRENT_TIMESTAMP
                                    """,
    "upsert_text_chunk": """INSERT INTO LIGHTRAG_DOC_CHUNKS (workspace, id, tokens,
                    chunk_order_index, full_doc_id, content, file_path, llm_cache_list,
                    create_time, update_time)
                    VALUES (:workspace, :id, :tokens,
                    :chunk_order_index, :full_doc_id, :content, :file_path, :llm_cache_list,
                    :create_time, :update_time)
                    ON DUPLICATE KEY
                    UPDATE tokens=VALUES(tokens),
                    chunk_order_index=VALUES(chunk_order_index),
                    full_doc_id=VALUES(full_doc_id),
                    content = VALUES(content),
                    file_path=VALUES(file_path),
                    llm_cache_list=VALUES(llm_cache_list),
                    update_time = VALUES(update_time)
                    """,
    "upsert_full_entities": """INSERT INTO LIGHTRAG_FULL_ENTITIES (workspace, id, entity_names, count,
                    create_time, update_time)
                    VALUES (:workspace, :id, :entity_names, :count,
                    :create_time, :update_time)
                    ON DUPLICATE KEY
                    UPDATE entity_names=VALUES(entity_names),
                    count=VALUES(count),
                    update_time = VALUES(update_time)
                    """,
    "upsert_full_relations": """INSERT INTO LIGHTRAG_FULL_RELATIONS (workspace, id, relation_pairs, count,
                    create_time, update_time)
                    VALUES (:workspace, :id, :relation_pairs, :count,
                    :create_time, :update_time)
                    ON DUPLICATE KEY
                    UPDATE relation_pairs=VALUES(relation_pairs),
                    count=VALUES(count),
                    update_time = VALUES(update_time)
                    """,
}


@final
@dataclass
class YashanKvStorage(BaseKVStorage):
    db: Optional[YashanDB] = None

    def __post_init__(self):
        pass

    async def initialize(self):
        async with get_data_init_lock():
            if self.db is None:
                self.db = await ClientManager.get_client()

            # Implement workspace priority: PostgreSQLDB.workspace > self.workspace > "default"
            if self.db.workspace:
                # Use PostgreSQLDB's workspace (highest priority)
                self.workspace = self.db.workspace
            elif self.workspace:
                # Use storage class's workspace (medium priority)
                pass
            else:
                # Use "default" for compatibility (lowest priority)
                self.workspace = "default"

    async def finalize(self):
        async with get_storage_lock():
            if self.db is not None:
                await ClientManager.release_client(self.db)
                self.db = None

    ################ QUERY METHODS ################
    async def get_all(self) -> dict[str, Any]:
        """Get all data from storage

        Returns:
            Dictionary containing all stored data
        """

        assert self.db

        table_name = namespace_to_table_name(self.namespace)
        if not table_name:
            logger.error(
                f"[{self.workspace}] Unknown namespace for get_all: {self.namespace}"
            )
            return {}

        sql = f"SELECT * FROM {table_name} WHERE workspace=:workspace"
        params = {"workspace": self.workspace}

        try:
            results = await self.db.query(sql, params, multirows=True)

            # Special handling for LLM cache to ensure compatibility with _get_cached_extraction_results
            if is_namespace(self.namespace, NameSpace.KV_STORE_LLM_RESPONSE_CACHE):
                processed_results = {}
                for row in results:
                    create_time = row.get("create_time", 0)
                    update_time = row.get("update_time", 0)
                    # Map field names and add cache_type for compatibility
                    processed_row = {
                        **row,
                        "return": row.get("return_value", ""),
                        "cache_type": row.get("original_prompt", "unknow"),
                        "original_prompt": row.get("original_prompt", ""),
                        "chunk_id": row.get("chunk_id"),
                        "mode": row.get("mode", "default"),
                        "create_time": create_time,
                        "update_time": create_time if update_time == 0 else update_time,
                    }
                    processed_results[row["id"]] = processed_row
                return processed_results

            # For text_chunks namespace, parse llm_cache_list JSON string back to list
            if is_namespace(self.namespace, NameSpace.KV_STORE_TEXT_CHUNKS):
                processed_results = {}
                for row in results:
                    llm_cache_list = row.get("llm_cache_list", [])
                    if isinstance(llm_cache_list, str):
                        try:
                            llm_cache_list = json.loads(llm_cache_list)
                        except json.JSONDecodeError:
                            llm_cache_list = []
                    row["llm_cache_list"] = llm_cache_list
                    create_time = row.get("create_time", 0)
                    update_time = row.get("update_time", 0)
                    row["create_time"] = create_time
                    row["update_time"] = (
                        create_time if update_time == 0 else update_time
                    )
                    processed_results[row["id"]] = row
                return processed_results

            # For FULL_ENTITIES namespace, parse entity_names JSON string back to list
            if is_namespace(self.namespace, NameSpace.KV_STORE_FULL_ENTITIES):
                processed_results = {}
                for row in results:
                    entity_names = row.get("entity_names", [])
                    if isinstance(entity_names, str):
                        try:
                            entity_names = json.loads(entity_names)
                        except json.JSONDecodeError:
                            entity_names = []
                    row["entity_names"] = entity_names
                    create_time = row.get("create_time", 0)
                    update_time = row.get("update_time", 0)
                    row["create_time"] = create_time
                    row["update_time"] = (
                        create_time if update_time == 0 else update_time
                    )
                    processed_results[row["id"]] = row
                return processed_results

            # For FULL_RELATIONS namespace, parse relation_pairs JSON string back to list
            if is_namespace(self.namespace, NameSpace.KV_STORE_FULL_RELATIONS):
                processed_results = {}
                for row in results:
                    relation_pairs = row.get("relation_pairs", [])
                    if isinstance(relation_pairs, str):
                        try:
                            relation_pairs = json.loads(relation_pairs)
                        except json.JSONDecodeError:
                            relation_pairs = []
                    row["relation_pairs"] = relation_pairs
                    create_time = row.get("create_time", 0)
                    update_time = row.get("update_time", 0)
                    row["create_time"] = create_time
                    row["update_time"] = (
                        create_time if update_time == 0 else update_time
                    )
                    processed_results[row["id"]] = row
                return processed_results

            # For other namespaces, return as-is
            return {row["id"]: row for row in results}
        except Exception as e:
            logger.error(
                f"[{self.workspace}] Error retrieving all data from {self.namespace}: {e}"
            )
            return {}

    async def get_by_id(self, id: str) -> dict[str, Any] | None:
        """Get data by id."""
        assert self.db

        sql = SQL_KV_TEMPLATES["get_by_id_" + self.namespace]
        params = {"workspace": self.workspace, "id": id}
        response = await self.db.query(sql, params)

        if response and is_namespace(self.namespace, NameSpace.KV_STORE_TEXT_CHUNKS):
            # Parse llm_cache_list JSON string back to list
            llm_cache_list = response.get("llm_cache_list", [])
            if isinstance(llm_cache_list, str):
                try:
                    llm_cache_list = json.loads(llm_cache_list)
                except json.JSONDecodeError:
                    llm_cache_list = []
            response["llm_cache_list"] = llm_cache_list
            create_time = response.get("create_time", 0)
            update_time = response.get("update_time", 0)
            response["create_time"] = create_time
            response["update_time"] = create_time if update_time == 0 else update_time

        # Special handling for LLM cache to ensure compatibility with _get_cached_extraction_results
        if response and is_namespace(
            self.namespace, NameSpace.KV_STORE_LLM_RESPONSE_CACHE
        ):
            create_time = response.get("create_time", 0)
            update_time = response.get("update_time", 0)
            # Parse queryparam JSON string back to dict
            queryparam = response.get("queryparam")
            if isinstance(queryparam, str):
                try:
                    queryparam = json.loads(queryparam)
                except json.JSONDecodeError:
                    queryparam = None
            # Map field names for compatibility (mode field removed)
            response = {
                **response,
                "return": response.get("return_value", ""),
                "cache_type": response.get("cache_type"),
                "original_prompt": response.get("original_prompt", ""),
                "chunk_id": response.get("chunk_id"),
                "queryparam": queryparam,
                "create_time": create_time,
                "update_time": create_time if update_time == 0 else update_time,
            }

        # Special handling for FULL_ENTITIES namespace
        if response and is_namespace(self.namespace, NameSpace.KV_STORE_FULL_ENTITIES):
            # Parse entity_names JSON string back to list
            entity_names = response.get("entity_names", [])
            if isinstance(entity_names, str):
                try:
                    entity_names = json.loads(entity_names)
                except json.JSONDecodeError:
                    entity_names = []
            response["entity_names"] = entity_names
            create_time = response.get("create_time", 0)
            update_time = response.get("update_time", 0)
            response["create_time"] = create_time
            response["update_time"] = create_time if update_time == 0 else update_time

        # Special handling for FULL_RELATIONS namespace
        if response and is_namespace(self.namespace, NameSpace.KV_STORE_FULL_RELATIONS):
            # Parse relation_pairs JSON string back to list
            relation_pairs = response.get("relation_pairs", [])
            if isinstance(relation_pairs, str):
                try:
                    relation_pairs = json.loads(relation_pairs)
                except json.JSONDecodeError:
                    relation_pairs = []
            response["relation_pairs"] = relation_pairs
            create_time = response.get("create_time", 0)
            update_time = response.get("update_time", 0)
            response["create_time"] = create_time
            response["update_time"] = create_time if update_time == 0 else update_time

        return response if response else None

    # Query by id
    async def get_by_ids(self, ids: list[str]) -> list[dict[str, Any]]:
        """Get data by ids"""
        assert self.db

        sql = SQL_KV_TEMPLATES["get_by_ids_" + self.namespace].format(
            ids=",".join([f"'{id}'" for id in ids])
        )
        params = {"workspace": self.workspace}
        results = await self.db.query(sql, params, multirows=True)

        if results and is_namespace(self.namespace, NameSpace.KV_STORE_TEXT_CHUNKS):
            # Parse llm_cache_list JSON string back to list for each result
            for result in results:
                llm_cache_list = result.get("llm_cache_list", [])
                if isinstance(llm_cache_list, str):
                    try:
                        llm_cache_list = json.loads(llm_cache_list)
                    except json.JSONDecodeError:
                        llm_cache_list = []
                result["llm_cache_list"] = llm_cache_list
                create_time = result.get("create_time", 0)
                update_time = result.get("update_time", 0)
                result["create_time"] = create_time
                result["update_time"] = create_time if update_time == 0 else update_time

        # Special handling for LLM cache to ensure compatibility with _get_cached_extraction_results
        if results and is_namespace(
            self.namespace, NameSpace.KV_STORE_LLM_RESPONSE_CACHE
        ):
            processed_results = []
            for row in results:
                create_time = row.get("create_time", 0)
                update_time = row.get("update_time", 0)
                # Parse queryparam JSON string back to dict
                queryparam = row.get("queryparam")
                if isinstance(queryparam, str):
                    try:
                        queryparam = json.loads(queryparam)
                    except json.JSONDecodeError:
                        queryparam = None
                # Map field names for compatibility (mode field removed)
                processed_row = {
                    **row,
                    "return": row.get("return_value", ""),
                    "cache_type": row.get("cache_type"),
                    "original_prompt": row.get("original_prompt", ""),
                    "chunk_id": row.get("chunk_id"),
                    "queryparam": queryparam,
                    "create_time": create_time,
                    "update_time": create_time if update_time == 0 else update_time,
                }
                processed_results.append(processed_row)
            return processed_results

        # Special handling for FULL_ENTITIES namespace
        if results and is_namespace(self.namespace, NameSpace.KV_STORE_FULL_ENTITIES):
            for result in results:
                # Parse entity_names JSON string back to list
                entity_names = result.get("entity_names", [])
                if isinstance(entity_names, str):
                    try:
                        entity_names = json.loads(entity_names)
                    except json.JSONDecodeError:
                        entity_names = []
                result["entity_names"] = entity_names
                create_time = result.get("create_time", 0)
                update_time = result.get("update_time", 0)
                result["create_time"] = create_time
                result["update_time"] = create_time if update_time == 0 else update_time

        # Special handling for FULL_RELATIONS namespace
        if results and is_namespace(self.namespace, NameSpace.KV_STORE_FULL_RELATIONS):
            for result in results:
                # Parse relation_pairs JSON string back to list
                relation_pairs = result.get("relation_pairs", [])
                if isinstance(relation_pairs, str):
                    try:
                        relation_pairs = json.loads(relation_pairs)
                    except json.JSONDecodeError:
                        relation_pairs = []
                result["relation_pairs"] = relation_pairs
                create_time = result.get("create_time", 0)
                update_time = result.get("update_time", 0)
                result["create_time"] = create_time
                result["update_time"] = create_time if update_time == 0 else update_time

        return results if results else []

    async def filter_keys(self, keys: set[str]) -> set[str]:
        """Filter out duplicated content"""
        assert self.db

        sql = SQL_COMMON_TEMPLATES["filter_keys"].format(
            table_name=namespace_to_table_name(self.namespace),
            ids=",".join([f"'{id}'" for id in keys]),
        )
        params = {"workspace": self.workspace}
        try:
            res = await self.db.query(sql, params, multirows=True)
            if res:
                exist_keys = [key["id"] for key in res]
            else:
                exist_keys = []
            new_keys = set([s for s in keys if s not in exist_keys])
            return new_keys
        except Exception as e:
            logger.error(
                f"[{self.workspace}] PostgreSQL database,\nsql:{sql},\nparams:{params},\nerror:{e}"
            )
            raise

    ################ INSERT METHODS ################
    async def upsert(self, data: dict[str, dict[str, Any]]) -> None:
        logger.debug(f"[{self.workspace}] Inserting {len(data)} to {self.namespace}")
        if not data:
            return

        assert self.db

        if is_namespace(self.namespace, NameSpace.KV_STORE_TEXT_CHUNKS):
            # Get current UTC time and convert to naive datetime for database storage
            current_time = datetime.datetime.now(timezone.utc).replace(tzinfo=None)
            for k, v in data.items():
                upsert_sql = SQL_KV_TEMPLATES["upsert_text_chunk"]
                _data = {
                    "workspace": self.workspace,
                    "id": k,
                    "tokens": v["tokens"],
                    "chunk_order_index": v["chunk_order_index"],
                    "full_doc_id": v["full_doc_id"],
                    "content": v["content"],
                    "file_path": v["file_path"],
                    "llm_cache_list": json.dumps(v.get("llm_cache_list", [])),
                    "create_time": current_time,
                    "update_time": current_time,
                }
                await self.db.execute(upsert_sql, _data)
        elif is_namespace(self.namespace, NameSpace.KV_STORE_FULL_DOCS):
            for k, v in data.items():
                upsert_sql = SQL_KV_TEMPLATES["upsert_doc_full"]
                _data = {
                    "id": k,
                    "content": v["content"],
                    "workspace": self.workspace,
                }
                await self.db.execute(upsert_sql, _data)
        elif is_namespace(self.namespace, NameSpace.KV_STORE_LLM_RESPONSE_CACHE):
            for k, v in data.items():
                upsert_sql = SQL_KV_TEMPLATES["upsert_llm_response_cache"]
                _data = {
                    "workspace": self.workspace,
                    "id": k,  # Use flattened key as id
                    "original_prompt": v["original_prompt"],
                    "return_value": v["return"],
                    "chunk_id": v.get("chunk_id"),
                    "cache_type": v.get(
                        "cache_type", "extract"
                    ),  # Get cache_type from data
                    "queryparam": json.dumps(v.get("queryparam"))
                    if v.get("queryparam")
                    else None,
                }

                await self.db.execute(upsert_sql, _data)
        elif is_namespace(self.namespace, NameSpace.KV_STORE_FULL_ENTITIES):
            # Get current UTC time and convert to naive datetime for database storage
            current_time = datetime.datetime.now(timezone.utc).replace(tzinfo=None)
            for k, v in data.items():
                upsert_sql = SQL_KV_TEMPLATES["upsert_full_entities"]
                _data = {
                    "workspace": self.workspace,
                    "id": k,
                    "entity_names": json.dumps(v["entity_names"]),
                    "count": v["count"],
                    "create_time": current_time,
                    "update_time": current_time,
                }
                await self.db.execute(upsert_sql, _data)
        elif is_namespace(self.namespace, NameSpace.KV_STORE_FULL_RELATIONS):
            # Get current UTC time and convert to naive datetime for database storage
            current_time = datetime.datetime.now(timezone.utc).replace(tzinfo=None)
            for k, v in data.items():
                upsert_sql = SQL_KV_TEMPLATES["upsert_full_relations"]
                _data = {
                    "workspace": self.workspace,
                    "id": k,
                    "relation_pairs": json.dumps(v["relation_pairs"]),
                    "count": v["count"],
                    "create_time": current_time,
                    "update_time": current_time,
                }
                await self.db.execute(upsert_sql, _data)

    async def index_done_callback(self) -> None:
        # PG handles persistence automatically
        pass

    async def delete(self, ids: list[str]) -> None:
        """Delete specific records from storage by their IDs

        Args:
            ids (list[str]): List of document IDs to be deleted from storage

        Returns:
            None
        """
        if not ids:
            return

        assert self.db

        table_name = namespace_to_table_name(self.namespace)
        if not table_name:
            logger.error(
                f"[{self.workspace}] Unknown namespace for deletion: {self.namespace}"
            )
            return

        ids_str = ids_format_query(ids)
        delete_sql = (
            f"DELETE FROM {table_name} WHERE workspace=:workspace AND id = {ids_str}"
        )

        try:
            await self.db.execute(delete_sql, {"workspace": self.workspace})
            logger.debug(
                f"[{self.workspace}] Successfully deleted {len(ids)} records from {self.namespace}"
            )
        except Exception as e:
            logger.error(
                f"[{self.workspace}] Error while deleting records from {self.namespace}: {e}"
            )

    async def drop(self) -> dict[str, str]:
        """Drop the storage"""
        assert self.db

        async with get_storage_lock():
            try:
                table_name = namespace_to_table_name(self.namespace)
                if not table_name:
                    return {
                        "status": "error",
                        "message": f"Unknown namespace: {self.namespace}",
                    }

                drop_sql = SQL_COMMON_TEMPLATES["drop_specific_table_workspace"].format(
                    table_name=table_name
                )
                await self.db.execute(drop_sql, {"workspace": self.workspace})
                return {"status": "success", "message": "data dropped"}
            except Exception as e:
                return {"status": "error", "message": str(e)}


@final
@dataclass
class YashanGraphStorage(BaseGraphStorage):
    db: Optional[YashanDB] = None

    def _get_workspace_graph_name(self) -> str:
        """
        Generate graph name based on workspace and namespace for data isolation.
        Rules:
        - If workspace is empty or "default": graph_name = namespace
        - If workspace has other value: graph_name = workspace_namespace

        Args:
            None

        Returns:
            str: The graph name for the current workspace
        """
        workspace = self.workspace
        namespace = self.namespace

        if workspace and workspace.strip() and workspace.strip().lower() != "default":
            # Ensure names comply with PostgreSQL identifier specifications
            safe_workspace = re.sub(r"[^a-zA-Z0-9_]", "_", workspace.strip())
            safe_namespace = re.sub(r"[^a-zA-Z0-9_]", "_", namespace)
            return f"{safe_workspace}_{safe_namespace}"
        else:
            # When the workspace is "default", use the namespace directly (for backward compatibility with legacy implementations)
            return re.sub(r"[^a-zA-Z0-9_]", "_", namespace)

    @staticmethod
    def _normalize_node_id(node_id: str) -> str:
        """
        Normalize node ID to ensure special characters are properly handled in Cypher queries.

        Args:
            node_id: The original node ID

        Returns:
            Normalized node ID suitable for Cypher queries
        """
        # Escape backslashes
        normalized_id = node_id
        normalized_id = normalized_id.replace("\\", "\\\\")
        normalized_id = normalized_id.replace('"', '\\"')
        normalized_id = normalized_id.replace("'", "''")
        return normalized_id

    async def initialize(self):
        async with get_data_init_lock():
            if self.db is None:
                self.db = await ClientManager.get_client()

            # Implement workspace priority: YashanDB.workspace > self.workspace > "default"
            if self.db.workspace:
                # Use YashanDB's workspace (highest priority)
                self.workspace = self.db.workspace
            if hasattr(self, "workspace") and self.workspace:
                # Use storage class's workspace (medium priority)
                self.workspace = self.workspace
            else:
                # Use "default" for compatibility (lowest priority)
                self.workspace = "default"

            # Dynamically generate graph name based on workspace
            self.graph_name = self._get_workspace_graph_name()

            # Log the graph initialization for debugging
            logger.info(
                f"[{self.workspace}] YashanDB Graph initialized: graph_name='{self.graph_name}'"
            )

            # Execute each statement separately and ignore errors
            self.edges_table_name = f"{self.graph_name}_edges"
            self.vertices_table_name = f"{self.graph_name}_vertices"
            self.graph_table_name = f"{self.graph_name}_graph"
            # Note: node_id is entity_id, and is label of vertex too.
            # Graph_node structure:
            # {entity_id: node_id
            # properties:
            # {"created_at": "2023-01-01",
            # "file_path": "path/to/file",
            # "source_id": "chunk-xxxxxx",
            # "description": "xxxxxx"
            # "entity_type":"xxxxxx",
            # "entity_id": node_id
            # }}
            # Graph_edge structure:
            # {id: edge_id
            # start_entity_id: node_id
            # end_entity_id: node_id
            # properties:
            # {"file_path":"xxxxxx",
            # "source_id":"xxxxxx",
            # "keywords":"xxxxxx",
            # "description":"xxxxxx",
            # "weight": double
            # }}

            queries = [
                """CREATE OR REPLACE PROCEDURE create_property_graph(
                    p_graph_name IN VARCHAR2,
                    p_vertex_table_name IN VARCHAR2,
                    p_edge_table_name IN VARCHAR2
                )
                IS
                    v_vertex_count NUMBER;
                    v_edge_count NUMBER;
                    v_graph_exists NUMBER;
                    v_ddl_stmt    CLOB; -- 使用 CLOB 存储可能较长的 DDL 语句
                BEGIN    
                    -- 检查顶点表是否存在
                    SELECT COUNT(*) INTO v_vertex_count
                    FROM user_tables 
                    WHERE table_name = UPPER(p_vertex_table_name);
                    
                    IF v_vertex_count = 0 THEN
                        RAISE_APPLICATION_ERROR(20001, 'Vertex table ' || p_vertex_table_name || ' does not exist');
                    END IF;
                    
                    -- 检查边表是否存在
                    SELECT COUNT(*) INTO v_edge_count
                    FROM user_tables 
                    WHERE table_name = UPPER(p_edge_table_name);
                    
                    IF v_edge_count = 0 THEN
                        RAISE_APPLICATION_ERROR(-30001, 'Edge table ' || p_edge_table_name || ' does not exist');
                    END IF;
                    
                    -- 创建属性图
                    -- 先drop原属性图
                    v_ddl_stmt := 'DROP PROPERTY GRAPH IF EXISTS ' || p_graph_name;
                    EXECUTE IMMEDIATE v_ddl_stmt;
                    
                    -- 动态构造并执行 CREATE PROPERTY GRAPH DDL
                    v_ddl_stmt := 
                        'CREATE PROPERTY GRAPH ' || p_graph_name || 
                        ' VERTEX TABLES ( ' ||
                            p_vertex_table_name || 
                            ' KEY (entity_id) ' ||
                            ' LABEL ' || p_vertex_table_name ||
                        ' ) ' ||
                        ' EDGE TABLES ( ' ||
                            p_edge_table_name ||
                            ' SOURCE KEY (start_entity_id) REFERENCES ' || p_vertex_table_name || '(entity_id) ' ||
                            ' DESTINATION KEY (end_entity_id) REFERENCES ' || p_vertex_table_name || '(entity_id) ' ||
                            ' LABEL ' || p_edge_table_name ||
                        ' )';
                    
                    EXECUTE IMMEDIATE v_ddl_stmt;
                    
                    DBMS_OUTPUT.PUT_LINE('Created graph: ' || p_graph_name);
                    DBMS_OUTPUT.PUT_LINE('Using vertex table: ' || p_vertex_table_name);
                    DBMS_OUTPUT.PUT_LINE('Using edge table: ' || p_edge_table_name);
                EXCEPTION
                    WHEN OTHERS THEN
                        DBMS_OUTPUT.PUT_LINE('Error creating graph: ' || SQLERRM);
                        RAISE;
                END create_property_graph;
                """,
                f"""CREATE TABLE IF NOT EXISTS {self.vertices_table_name} (
                    entity_id VARCHAR(512) PRIMARY KEY,
                    properties JSON
                )""",
                f"CREATE SEQUENCE {self.edges_table_name}_id_seq START WITH 1 INCREMENT BY 1 NOCACHE",
                f"""CREATE TABLE IF NOT EXISTS {self.edges_table_name} (
                    id NUMBER DEFAULT {self.edges_table_name}_id_seq.NEXTVAL PRIMARY KEY,
                    start_entity_id VARCHAR(512),
                    end_entity_id VARCHAR(512),
                    properties JSON,
                    CONSTRAINT c_{self.edges_table_name}_start_id FOREIGN KEY (start_entity_id) REFERENCES {self.vertices_table_name}(entity_id) ON DELETE CASCADE,
                    CONSTRAINT c_{self.edges_table_name}_end_id FOREIGN KEY (end_entity_id) REFERENCES {self.vertices_table_name}(entity_id) ON DELETE CASCADE 
                )""",
                f"CREATE INDEX {self.edges_table_name}_start_id_idx ON {self.edges_table_name}(start_entity_id)",
                f"CREATE INDEX {self.edges_table_name}_end_id_idx ON {self.edges_table_name}(end_entity_id)",
                f"""
                BEGIN
                    create_property_graph(
                        p_graph_name => '{self.graph_table_name}',
                        p_vertex_table_name => '{self.vertices_table_name}',
                        p_edge_table_name => '{self.edges_table_name}'
                    );
                END;
                """,
                """
                CREATE OR REPLACE PROCEDURE upsert_node(
                    p_vertices IN VARCHAR2,
                    p_entity_id IN VARCHAR2,
                    p_properties IN JSON
                ) AS
                    v_node_exists NUMBER;
                    v_dynamic_sql VARCHAR2(4000); -- 用于构建动态SQL
                BEGIN
                    -- 1. 动态检查节点是否存在
                    v_dynamic_sql := 'SELECT COUNT(*) FROM ' || p_vertices || 
                                    ' WHERE entity_id = :entity_id';
                    EXECUTE IMMEDIATE v_dynamic_sql INTO v_node_exists USING p_entity_id;
                    -- 2. 根据检查结果执行插入或更新
                    IF v_node_exists = 0 THEN
                        -- 动态插入新节点
                        v_dynamic_sql := 'INSERT INTO ' || p_vertices || 
                                        ' (entity_id, properties) VALUES (:entity_id, :properties)';
                        EXECUTE IMMEDIATE v_dynamic_sql USING p_entity_id, p_properties;
                        COMMIT;
                        DBMS_OUTPUT.PUT_LINE('Created new node: ' || p_entity_id);
                    ELSE
                        -- 动态更新现有节点
                        v_dynamic_sql := 'UPDATE ' || p_vertices || 
                                        ' SET properties = :properties WHERE entity_id = :entity_id';
                        EXECUTE IMMEDIATE v_dynamic_sql USING p_properties, p_entity_id;
                        COMMIT;
                        DBMS_OUTPUT.PUT_LINE('Updated existing node: ' || p_entity_id);
                    END IF;
                EXCEPTION
                    WHEN OTHERS THEN
                        DBMS_OUTPUT.PUT_LINE('Error in upsert_node: ' || SQLERRM);
                        RAISE; -- 重新抛出异常
                END upsert_node;
                """,
                """
                CREATE OR REPLACE PROCEDURE upsert_edge(
                    p_edges IN VARCHAR2,
                    p_vertices IN VARCHAR2,
                    p_start_entity_id IN VARCHAR2,
                    p_end_entity_id IN VARCHAR2,
                    p_properties IN JSON
                ) AS
                    v_edge_exists NUMBER;
                    v_start_entity_id VARCHAR(200);
                    v_end_entity_id VARCHAR(200);
                    v_dynamic_sql VARCHAR2(4000); -- 用于构建动态SQL
                BEGIN
                    -- 1. 动态检查节点是否存在
                    v_dynamic_sql := 'SELECT entity_id FROM ' || p_vertices || ' WHERE entity_id = :start_entity_id';
                    EXECUTE IMMEDIATE v_dynamic_sql INTO v_start_entity_id USING p_start_entity_id;
                    v_dynamic_sql := 'SELECT entity_id FROM ' || p_vertices || ' WHERE entity_id = :end_entity_id';
                    EXECUTE IMMEDIATE v_dynamic_sql INTO v_end_entity_id USING p_end_entity_id;
                    DBMS_OUTPUT.PUT_LINE('Start Entity ID: ' || v_start_entity_id || ', End Entity ID: ' || v_end_entity_id);
                    -- 2. 动态检查边是否存在 (因为是双向边，所以需要检查以开始或者结束节点为起始的边)
                    v_dynamic_sql := 'SELECT COUNT(*) FROM ' || p_edges ||
                                    ' WHERE (start_entity_id = :start_entity_id AND end_entity_id = :end_entity_id) OR (start_entity_id = :end_entity_id AND end_entity_id = :start_entity_id)';
                    EXECUTE IMMEDIATE v_dynamic_sql INTO v_edge_exists USING v_start_entity_id, v_end_entity_id, v_end_entity_id, v_start_entity_id;
                    DBMS_OUTPUT.PUT_LINE('Edge exists: ' || v_edge_exists);
                    -- 2. 根据检查结果执行插入或更新
                    IF v_edge_exists = 0 THEN
                        -- 动态插入新节点
                        v_dynamic_sql := 'INSERT INTO ' || p_edges || 
                                        ' (start_entity_id, end_entity_id, properties) VALUES (:start_entity_id, :end_entity_id, :properties)';
                        EXECUTE IMMEDIATE v_dynamic_sql USING v_start_entity_id, v_end_entity_id, p_properties;
                        COMMIT;
                        DBMS_OUTPUT.PUT_LINE('Created new edge: ' || v_start_entity_id || ' -> ' || v_end_entity_id);
                    ELSE
                        -- 动态更新现有节点, 由于是双向边, 需要判断以开始或者结束节点为起始节点的边
                        v_dynamic_sql := 'UPDATE ' || p_edges || 
                                        ' SET properties = :properties WHERE (start_entity_id = :start_entity_id AND end_entity_id = :end_entity_id) OR (start_entity_id = :start_entity_id AND end_entity_id = :start_entity_id)';
                        EXECUTE IMMEDIATE v_dynamic_sql USING p_properties, v_start_entity_id, v_end_entity_id, v_end_entity_id, v_start_entity_id;
                        COMMIT;
                        DBMS_OUTPUT.PUT_LINE('Updated existing edge: ' || v_start_entity_id || ' -> ' || v_end_entity_id);
                    END IF;
                EXCEPTION
                    WHEN OTHERS THEN
                        DBMS_OUTPUT.PUT_LINE('Error in upsert_edge: ' || SQLERRM);
                        RAISE; -- 重新抛出异常
                END upsert_edge;
                """,
            ]

            for query in queries:
                # Use the new flag to silently ignore "already exists" errors
                # at the source, preventing log spam.

                await self.db.execute(
                    sql=query,
                    ignore_if_exists=True,  # Pass the new flag
                )

    async def finalize(self):
        async with get_graph_db_lock():
            if self.db is not None:
                await ClientManager.release_client(self.db)
                self.db = None

    async def index_done_callback(self) -> None:
        # YashanDB handles persistence automatically
        pass

    async def has_node(self, node_id: str) -> bool:
        """
        Check if a node with the given label exists in the database

        Args:
            node_id: Label of the node to check

        Returns:
            bool: True if node exists, False otherwise

        Raises:
            ValueError: If node_id is invalid
            Exception: If there is an error executing the query
        """
        # Check if node_id is valid
        if not node_id:
            raise ValueError("Node ID cannot be empty")
        label = node_id

        # Query to check if node exists
        # 查询性能优化：直接查询节点表，不走图查询
        query = f"""
        SELECT COUNT(*) > 0 AS NODE_EXISTS 
        FROM {self.vertices_table_name} 
        WHERE entity_id = :entity_id
        """
        try:
            assert self.db
            params = {"entity_id": label}
            result = await self.db.query(query, params)
            return result["node_exists"]
        except Exception as e:
            logger.error(f"Error checking node existence: {e}")
            raise

    async def has_edge(self, source_node_id: str, target_node_id: str) -> bool:
        """
        Check if an edge exists between two nodes in the database

        Args:
            source_node_id: ID of the source node
            target_node_id: ID of the target node

        Returns:
            bool: True if edge exists, False otherwise

        Raises:
            ValueError: If source_node_id or target_node_id is invalid
            Exception: If there is an error executing the query
        """
        # Check if node_ids are valid
        if not source_node_id or not target_node_id:
            raise ValueError("Source and target node IDs cannot be empty")

        # Query to check if edge exists
        # 查询性能优化，直接查询边表，不走图查询, 双向边, 所以需要判断以开始或者结束节点开始的边
        query = f"""
        SELECT COUNT(id) > 0 AS EDGE_EXISTS 
        FROM {self.edges_table_name} 
        WHERE (start_entity_id = :source_node_id AND end_entity_id = :target_node_id)
        OR (start_entity_id = :target_node_id AND end_entity_id = :source_node_id)
        """

        try:
            assert self.db
            params = {
                "source_node_id": source_node_id,
                "target_node_id": target_node_id,
            }
            result = await self.db.query(query, params)
            return result["edge_exists"]
        except Exception as e:
            logger.error(f"Error checking edge existence: {e}")
            raise

    async def get_node(self, node_id: str) -> dict[str, str] | None:
        """Get node by its label identifier, return only node properties

        Args:
            node_id: The node label to look up

        Returns:
            dict: Node properties if found
            None: If node not found

        Raises:
            ValueError: If node_id is invalid
            Exception: If there is an error executing the query
        """
        # Check if node_id is valid
        if not node_id:
            raise ValueError("Node ID cannot be empty")
        label = node_id
        # Query to get node properties
        # 查询性能优化，直接查询节点表，不走图查询
        query = f"""
        SELECT entity_id, CAST(properties AS CLOB) AS properties FROM {self.vertices_table_name} where entity_id = :entity_id LIMIT 1
        """
        try:
            assert self.db
            params = {"entity_id": label}
            result = await self.db.query(query, params)
            if result:
                return dict(json.loads(result["properties"]))
            return None
        except Exception as e:
            logger.error(f"Error getting node: {e}")
            raise

    async def get_nodes_batch(self, node_ids: list[str]) -> dict[str, dict]:
        """
        Retrieve multiple nodes in one query using optimized batch query.

        Args:
            node_ids: List of node entity IDs to fetch.

        Returns:
            A dictionary mapping each node_id to its node data (or None if not found).
        """
        # Check if node_ids is valid
        if not node_ids:
            logger.warning("Node IDs list cannot be empty")
            return {}

        # 使用参数化查询避免SQL注入并减少解析性能损耗
        if len(node_ids) == 1:
            # 单节点查询优化，直接查询节点表，不走图查询
            query = f"""SELECT entity_id, CAST(properties AS CLOB) AS properties FROM {self.vertices_table_name} where n.entity_id = :node_id_0 limit 1"""
            params = {"node_id_0": node_ids[0]}
        else:
            # 批量查询优化 - 直接查询节点表，不走图查询
            # 构建参数化IN子句
            placeholders = ", ".join([f":node_id_{i}" for i in range(len(node_ids))])
            query = f"""SELECT entity_id, CAST(properties AS CLOB) AS properties FROM {self.vertices_table_name} where entity_id IN ({placeholders})"""
            params = {f"node_id_{i}": node_id for i, node_id in enumerate(node_ids)}

        try:
            assert self.db
            result = await self.db.query(query, params=params, multirows=True)

            # 构建结果字典，保持原始顺序
            result_dict = {}
            for node_id in node_ids:
                # 在查询结果中查找对应的节点
                for row in result:
                    if row.get("entity_id") == node_id:
                        result_dict[node_id] = json.loads(row.get("properties", {}))
                        break
                    else:
                        result_dict[node_id] = None

            return result_dict
        except Exception as e:
            logger.error(f"Error getting nodes batch: {e}")
            raise

    async def node_degree(self, node_id: str) -> int:
        """Get the degree (number of relationships) of a node with the given label.
        If multiple nodes have the same label, returns the degree of the first node.
        If no node is found, returns 0.

        Args:
            node_id: The label of the node

        Returns:
            int: The number of relationships the node has, or 0 if no node found

        Raises:
            ValueError: If node_id is invalid
            Exception: If there is an error executing the query
        """
        # Check if node_id is valid
        if not node_id:
            logger.warning("Node ID cannot be empty")
            return 0

        # Query to get node degree
        # 性能优化：改为直接查询边表
        query = f"""
        SELECT COUNT(id) AS DEGREE FROM {self.edges_table_name} WHERE :normalized_id IN (start_entity_id,end_entity_id)
        """
        params = {"normalized_id": node_id}
        try:
            assert self.db
            result = await self.db.query(query, params=params)
            if result:
                return result["degree"]
            return 0
        except Exception as e:
            logger.error(f"Error getting node degree: {e}")
            raise

    async def node_degrees_batch(self, node_ids: list[str]) -> dict[str, int]:
        """
        Retrieve the degree for multiple nodes in a single query using UNWIND.

        Args:
            node_ids: List of node labels (entity_id values) to look up.

        Returns:
            A dictionary mapping each node_id to its degree (number of relationships).
            If a node is not found, its degree will be set to 0.
        """
        # Check if node_ids is valid
        if not node_ids:
            logger.warning("Node IDs list cannot be empty")
            return {}

        placeholders = [f":node_id_{i}" for i in range(len(node_ids))]
        placeholders_str = f"({', '.join(placeholders)})"
        params = {f"node_id_{i}": node_id for i, node_id in enumerate(node_ids)}
        # Query to get node degrees
        # 性能优化：改为直接查询边表, 孤点会产生warning，但degree=0是正确的
        query = f"""
            SELECT entity_id, SUM(degree) AS degree
            FROM (
                (SELECT start_entity_id AS entity_id, COUNT(id) AS degree 
                FROM {self.edges_table_name} 
                where start_entity_id in {placeholders_str}
                GROUP BY start_entity_id)
                UNION ALL
                (SELECT end_entity_id AS entity_id, COUNT(id) AS degree 
                FROM {self.edges_table_name} 
                where end_entity_id in {placeholders_str}
                GROUP BY end_entity_id)
            )
            GROUP BY entity_id
        """
        try:
            assert self.db
            result = await self.db.query(query, params=params, multirows=True)
            # Handle cases where nodes are not found
            degrees = {
                row[str("entity_id").lower()]: row[str("degree").lower()]
                for row in result
            }
            for nid in node_ids:
                if nid not in degrees:
                    logger.warning(
                        f"[{self.graph_table_name}] No node found with label '{nid}'"
                    )
                    degrees[nid] = 0
            return degrees
        except Exception as e:
            logger.error(f"Error getting node degrees batch: {e}")
            raise

    async def edge_degree(self, src_id: str, tgt_id: str) -> int:
        """Get the total degree (sum of relationships) of two nodes.

        Args:
            src_id: Label of the source node
            tgt_id: Label of the target node

        Returns:
            int: Sum of the degrees of both nodes
        """
        src_degree = await self.node_degree(src_id)
        tgt_degree = await self.node_degree(tgt_id)

        # Convert None to 0 for addition
        src_degree = 0 if src_degree is None else src_degree
        tgt_degree = 0 if tgt_degree is None else tgt_degree

        return int(src_degree) + int(tgt_degree)

    async def edge_degrees_batch(
        self, edge_pairs: list[tuple[str, str]]
    ) -> dict[tuple[str, str], int]:
        """
        Calculate the combined degree for each edge (sum of the source and target node degrees)
        in batch using the already implemented node_degrees_batch.

        Args:
            edge_pairs: List of (src, tgt) tuples.

        Returns:
            A dictionary mapping each (src, tgt) tuple to the sum of their degrees.
        """
        # Check if edge_pairs is valid
        if not edge_pairs:
            logger.warning("Edge pairs list cannot be empty")
            return {}

        # Collect unique node IDs from all edge pairs.
        unique_node_ids = {src for src, _ in edge_pairs}
        unique_node_ids.update({tgt for _, tgt in edge_pairs})

        # Get degrees for all nodes in one go.
        degrees = await self.node_degrees_batch(list(unique_node_ids))

        # Sum up degrees for each edge pair.
        edge_degrees = {}
        for src, tgt in edge_pairs:
            edge_degrees[(src, tgt)] = degrees.get(src, 0) + degrees.get(tgt, 0)
        return edge_degrees

    async def get_edge(
        self, source_node_id: str, target_node_id: str
    ) -> dict[str, str] | None:
        """Get edge properties between two nodes.

        Args:
            source_node_id: Label of the source node
            target_node_id: Label of the target node

        Returns:
            dict: Edge properties if found, default properties if not found or on error

        Raises:
            ValueError: If either node_id is invalid
            Exception: If there is an error executing the query
        """
        # Check if node_ids is valid
        if not source_node_id or not target_node_id:
            logger.warning("Node IDs list cannot be empty")
            return None

        # Query to get edge properties
        # 性能优化：改为直接查询边表
        query = f"""
        SELECT CAST(properties AS CLOB) AS EDGE_PROPERTIES
        FROM {self.edges_table_name}
        WHERE (start_entity_id = :source_node_id AND end_entity_id = :target_node_id)
        or (start_entity_id = :target_node_id AND end_entity_id = :source_node_id)
        """
        params = {
            "source_node_id": source_node_id,
            "target_node_id": target_node_id,
        }
        try:
            assert self.db
            result = await self.db.query(query, params=params, multirows=True)
            if len(result) > 1:
                logger.warning(
                    f"[{self.graph_table_name}] Multiple edges found between {source_node_id} and {target_node_id}"
                )

            if result:
                try:
                    edge_result = dict(json.loads(result[0]["edge_properties"]))

                    required_keys = {
                        "weight": 1.0,
                        "source_id": None,
                        "description": None,
                        "keywords": None,
                    }
                    for key, default_value in required_keys.items():
                        if key not in edge_result:
                            edge_result[key] = default_value
                            logger.warning(
                                f"[{self.workspace}] Edge between {source_node_id} and {target_node_id} "
                                f"missing {key}, using default: {default_value}"
                            )
                    return edge_result
                except (KeyError, TypeError, ValueError) as e:
                    logger.error(
                        f"[{self.workspace}] Error processing edge properties between {source_node_id} "
                        f"and {target_node_id}: {str(e)}"
                    )
                    # Return default edge properties on error
                    return {
                        "weight": 1.0,
                        "source_id": None,
                        "description": None,
                        "keywords": None,
                    }
            return None
        except Exception as e:
            logger.error(
                f"[{self.workspace}] Error in get_edge between {source_node_id} and {target_node_id}: {str(e)}"
            )
            raise

    async def get_edges_batch(
        self, pairs: list[dict[str, str]]
    ) -> dict[tuple[str, str], dict]:
        """
        Retrieve edge properties for multiple (src, tgt) pairs in one query using optimized batch query.

        Args:
            pairs: List of dictionaries, e.g. [{"src": "node1", "tgt": "node2"}, ...]

        Returns:
            A dictionary mapping (src, tgt) tuples to their edge properties.
        """
        if not pairs:
            logger.warning("Edge pairs list cannot be empty")
            return {}
        normalized_pairs = []
        for pair in pairs:
            src = pair.get("src")
            tgt = pair.get("tgt")
            if not src or not tgt:
                logger.warning("Edge pair is missing 'src' or 'tgt'")
                continue
            normalized_pairs.append((src, tgt))

        if not normalized_pairs:
            return {}

        # Build parameterized query for batch edge retrieval
        # Create placeholders for each pair
        pair_conditions = []
        params = {}
        for i, (src, tgt) in enumerate(normalized_pairs):
            pair_conditions.append(
                f"((start_entity_id = :src_{i} AND end_entity_id = :tgt_{i}) OR (start_entity_id = :tgt_{i} AND end_entity_id = :src_{i}))"
            )
            params[f"src_{i}"] = src
            params[f"tgt_{i}"] = tgt

        # Build the batch query
        # 性能优化: 因为只有一跳，改为直接查询边表
        where_clause = " OR ".join(pair_conditions)
        query = f"""
            SELECT 
                start_entity_id as source_entity_id,
                end_entity_id as target_entity_id,
                CAST(properties AS CLOB) AS edge_properties
            FROM {self.edges_table_name}
            WHERE {where_clause}
        """

        try:
            assert self.db
            result = await self.db.query(query, multirows=True, params=params)

            # Build result dictionary
            edges = {}
            for row in result:
                source_id = str(row["source_entity_id"])
                target_id = str(row["target_entity_id"])
                edge_props = dict(json.loads(row["edge_properties"]))

                # Ensure required keys exist with defaults
                required_keys = {
                    "weight": 1.0,
                    "source_id": None,
                    "description": None,
                    "keywords": None,
                }
                for key, default_value in required_keys.items():
                    if key not in edge_props:
                        edge_props[key] = default_value

                edges[(source_id, target_id)] = edge_props
                edges[(target_id, source_id)] = edge_props

            # Add None for missing edges to maintain input order
            result_edges = {}
            for src, tgt in normalized_pairs:
                result_edges[(src, tgt)] = edges.get((src, tgt))
            return result_edges

        except Exception as e:
            logger.error(f"[{self.workspace}] Error in get_edges_batch: {str(e)}")
            # Fallback to individual queries on error
            logger.warning(
                "Falling back to individual edge queries due to batch query error"
            )
            edges = {}
            for src, tgt in pairs:
                try:
                    edge_result = await self.get_edge(src, tgt)
                    edges[(src, tgt)] = edge_result
                except Exception as individual_error:
                    logger.error(
                        f"Error getting edge {src}->{tgt}: {str(individual_error)}"
                    )
                    edges[(src, tgt)] = None
            return edges

    async def get_node_edges(self, source_node_id: str) -> list[tuple[str, str]] | None:
        """Retrieves all edges (relationships) for a particular node identified by its label using optimized parameterized query.

        Args:
            source_node_id: Label of the node to get edges for

        Returns:
            list[tuple[str, str]]: List of (source_label, target_label) tuples representing edges
            None: If no edges found

        Raises:
            ValueError: If source_node_id is invalid
            Exception: If there is an error executing the query
        """

        # Use parameterized query to prevent SQL injection
        # 性能优化：改为直接查询边表
        query = f"""
        SELECT start_entity_id AS source_node_id,
            end_entity_id AS target_node_id
        FROM   {self.edges_table_name}
        WHERE  start_entity_id = :node_id
        UNION ALL
        SELECT end_entity_id AS source_node_id,
            start_entity_id AS target_node_id
        FROM   {self.edges_table_name}
        WHERE  end_entity_id = :node_id 
        """
        try:
            assert self.db
            result = await self.db.query(
                query, multirows=True, params={"node_id": source_node_id}
            )
            edges = [(row["source_node_id"], row["target_node_id"]) for row in result]
            return edges
        except Exception as e:
            logger.error(f"[{self.workspace}] Error in get_node_edges: {str(e)}")
            raise

    async def get_nodes_edges_batch(
        self, node_ids: list[str]
    ) -> dict[str, list[tuple[str, str]]]:
        """
        Batch retrieve edges for multiple nodes in one query using optimized parameterized query.
        For each node, returns both outgoing and incoming edges to properly represent
        the undirected graph nature.

        Args:
            node_ids: List of node IDs (entity_id) for which to retrieve edges.

        Returns:
            A dictionary mapping each node ID to its list of edge tuples (source, target).
            For each node, the list includes both:
            - Outgoing edges: (queried_node, connected_node)
            - Incoming edges: (connected_node, queried_node)
        """
        if not node_ids:
            logger.warning("Node ID list cannot be empty")
            return {}

        # Use parameterized query to prevent SQL injection
        # Build placeholders for IN clause
        placeholders = []
        params = {}
        for i, node_id in enumerate(node_ids):
            placeholders.append(f":node_id_{i}")
            params[f"node_id_{i}"] = node_id

        placeholders_str = f"({', '.join(placeholders)})"

        # 性能改进: 由于是双边单跳查询，直接查询边表
        query = f"""
        (SELECT 
            start_entity_id as node_id,
            start_entity_id as source_node_id, 
            end_entity_id as target_node_id,
            CAST(e.properties AS CLOB) AS edge_properties
        FROM 
            {self.edges_table_name} e
        WHERE 
            start_entity_id IN {placeholders_str}
        )
        UNION ALL
        (SELECT 
            end_entity_id as node_id,
            end_entity_id as source_node_id, 
            start_entity_id as target_node_id,
            CAST(e.properties AS CLOB) AS edge_properties
        FROM 
            {self.edges_table_name} e
        WHERE 
            end_entity_id IN {placeholders_str}
        )

        """
        try:
            assert self.db
            result = await self.db.query(query, multirows=True, params=params)

            # Group edges by node_id
            edges_by_node = {}
            for row in result:
                node_id = row["node_id"]
                edge_tuple = (row["source_node_id"], row["target_node_id"])

                if node_id not in edges_by_node:
                    edges_by_node[node_id] = []
                edges_by_node[node_id].append(edge_tuple)

            return edges_by_node

        except Exception as e:
            logger.error(f"[{self.workspace}] Error in get_nodes_edges_batch: {str(e)}")
            raise

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type(
            (
                ConnectionResetError,
                OSError,
            )
        ),
    )
    async def upsert_node(self, node_id: str, node_data: dict[str, str]) -> None:
        """
        Upsert a node in the Yashandb database.

        Args:
            node_id: The unique identifier for the node (used as label)
            node_data: Dictionary of node properties
        """
        properties = node_data
        if not node_id or node_id == "":
            raise ValueError("Yashandb: entity_id should not be null")

        if "entity_id" not in properties:
            raise ValueError(
                "Yashandb: node properties must contain an 'entity_id' field"
            )

        # Use parameterized query to prevent SQL injection
        query = """
            BEGIN
                upsert_node(
                    p_vertices => :vertices_table,
                    p_entity_id => :entity_id,
                    p_properties => :properties_json
                );
            END;
        """

        params = {
            "vertices_table": self.vertices_table_name,
            "entity_id": node_id,
            "properties_json": properties,
        }

        try:
            assert self.db
            await self.db.execute(query, data=params)
        except Exception as e:
            logger.error(f"[{self.workspace}] Error in upsert_node: {str(e)}")
            raise

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type(
            (
                ConnectionResetError,
                OSError,
            )
        ),
    )
    async def upsert_edge(
        self, source_node_id: str, target_node_id: str, edge_data: dict[str, str]
    ) -> None:
        """
        Upsert an edge and its properties between two nodes identified by their labels.
        Ensures both source and target nodes exist and are unique before creating the edge.
        Uses entity_id property to uniquely identify nodes.

        Args:
            source_node_id (str): Label of the source node (used as identifier)
            target_node_id (str): Label of the target node (used as identifier)
            edge_data (dict): Dictionary of properties to set on the edge

        Raises:
            ValueError: If either source or target node does not exist or is not unique
        """
        properties = edge_data

        # Use parameterized query to prevent SQL injection
        query = """
            BEGIN
                upsert_edge(
                    p_edges => :edges_table,
                    p_vertices => :vertices_table,
                    p_start_entity_id => :source_entity_id,
                    p_end_entity_id => :target_entity_id,
                    p_properties => :properties_json
                );
            END;
        """

        params = {
            "edges_table": self.edges_table_name,
            "vertices_table": self.vertices_table_name,
            "source_entity_id": source_node_id,
            "target_entity_id": target_node_id,
            "properties_json": properties,
        }

        try:
            assert self.db
            await self.db.execute(query, data=params)
            # logger.info(
            #     f"[{self.workspace}] Edge upserted successfully: {source_node_id} -> {target_node_id}"
            # )
        except Exception as e:
            logger.error(f"[{self.workspace}] Error in upsert_edge: {str(e)}")
            raise

    async def delete_node(self, node_id: str) -> None:
        """
        Delete a node with the specified label using parameterized query to prevent SQL injection.

        Args:
            node_id: The label of the node to delete
        """

        # Use parameterized query to prevent SQL injection
        query = f"DELETE FROM {self.vertices_table_name} WHERE entity_id = :entity_id"

        try:
            assert self.db
            await self.db.execute(query, data={"entity_id": node_id})
            logger.info(f"[{self.workspace}] Node deleted successfully: {node_id}")
        except Exception as e:
            logger.error(f"[{self.workspace}] Error in delete_node: {str(e)}")
            raise

    async def remove_nodes(self, nodes: list[str]):
        """
        Remove multiple nodes from the graph.

        Args:
            node_ids (list[str]): A list of node IDs to remove.
        """
        for node in nodes:
            await self.delete_node(node)

    async def remove_edges(self, edges: list[tuple[str, str]]):
        """
        Delete multiple edges using parameterized batch query to prevent SQL injection.

        Args:
            edges: List of edges to be deleted, each edge is a (source, target) tuple
        """

        if edges:
            # Use parameterized query to prevent SQL injection
            # Build placeholders for batch deletion
            placeholders = []
            params = {}
            for i, (start_id, end_id) in enumerate(edges):
                placeholders.append(f"(:start_id_{i}, :end_id_{i})")
                params[f"start_id_{i}"] = start_id
                params[f"end_id_{i}"] = end_id

            placeholders_str = f"({', '.join(placeholders)})"

            query = f"""
                DELETE FROM {self.edges_table_name}
                WHERE (start_entity_id, end_entity_id) IN {placeholders_str}
            """
            try:
                assert self.db
                await self.db.execute(query, data=params)
                logger.info(
                    f"[{self.workspace}] Successfully removed {len(edges)} edges"
                )
            except Exception as e:
                logger.error(f"[{self.graph_name}] Error in remove_edges: {str(e)}")
                raise
        else:
            logger.error(
                f"[{self.graph_name}] remove_edges: edges is empty, skip remove edges"
            )

    async def get_all_labels(self) -> list[str]:
        """
        Get all existing node labels (node IDs) in the database
        Returns:
            ["Person", "Company", ...]  # Alphabetically sorted label list
        """
        query = f"""
            SELECT DISTINCT entity_id FROM {self.vertices_table_name}
        """
        assert self.db
        result = await self.db.query(query, multirows=True)
        labels = {row["entity_id"] for row in result}
        return sorted(list(labels))

    async def _bfs_subgraph(
        self, node_label: str, max_depth: int, max_nodes: int
    ) -> KnowledgeGraph:
        """
        Implements a true breadth-first search algorithm for subgraph retrieval.
        This method is used as a fallback when the standard Cypher query is too slow
        or when we need to guarantee BFS ordering.

        Args:
            node_label: Label of the starting node
            max_depth: Maximum depth of the subgraph
            max_nodes: Maximum number of nodes to return

        Returns:
            KnowledgeGraph object containing nodes and edges
        """
        from collections import deque

        result = KnowledgeGraph()
        visited_nodes = set()
        visited_node_ids = set()
        visited_edges = set()
        visited_edge_pairs = set()

        # Get starting node data
        # Use parameterized query to prevent SQL injection
        query = f"""
                SELECT ENTITY_ID as ID, CAST(properties AS CLOB) AS PROPERTIES FROM {self.vertices_table_name}
                WHERE entity_id = :entity_id LIMIT 1
                """
        assert self.db
        node_result = await self.db.query(query, params={"entity_id": node_label})
        if not node_result or not node_result["properties"]:
            return result

        # Create initial KnowledgeGraphNode
        start_node_data = node_result
        entity_id = json.loads(start_node_data["properties"])["entity_id"]
        internal_id = str(start_node_data["id"])

        start_node = KnowledgeGraphNode(
            id=internal_id,
            labels=[entity_id],
            properties=json.loads(start_node_data["properties"]),
        )

        # Initialize BFS queue, each element is a tuple of (node, depth)
        queue = deque([(start_node, 0)])

        visited_nodes.add(entity_id)
        visited_node_ids.add(internal_id)
        result.nodes.append(start_node)

        result.is_truncated = False

        # BFS search main loop
        while queue:
            # Get all nodes at the current depth
            current_level_nodes = []
            current_depth = None

            # Determine current depth
            if queue:
                current_depth = queue[0][1]

            # Extract all nodes at current depth from the queue
            while queue and queue[0][1] == current_depth:
                node, depth = queue.popleft()
                if depth > max_depth:
                    continue
                current_level_nodes.append(node)

            if not current_level_nodes:
                continue

            # Check depth limit
            if current_depth > max_depth:
                continue

            # Prepare node IDs list
            node_ids = [node.labels[0] for node in current_level_nodes]
            # Use parameterized query to prevent SQL injection
            # Build placeholders for IN clause
            placeholders = []
            params = {}
            for i, node_id in enumerate(node_ids):
                placeholders.append(f":node_id_{i}")
                params[f"node_id_{i}"] = node_id

            placeholders_str = f"({', '.join(placeholders)})"

            # 性能优化：出边和入边两条查询改为一条查询,
            # 修正错误：增加对孤立点的处理
            edge_query = f"""
                /* ---------- 1. 出边 ---------- */
                SELECT  start_entity_id            AS current_internal_id,
                        start_entity_id            AS current_id,
                        end_entity_id              AS neighbor_id,
                        end_entity_id              AS neighbor_internal_id,
                        CAST(edge.properties AS CLOB)            AS edge_properties,
                        edge.id                    AS edge_id,
                        CAST(neighbor.properties AS CLOB)        AS neighbor_properties,
                        true                       AS is_outgoing
                FROM    {self.edges_table_name} edge
                JOIN    {self.vertices_table_name} neighbor ON neighbor.entity_id = edge.end_entity_id
                WHERE   edge.start_entity_id IN {placeholders_str}

                UNION ALL

                /* ---------- 2. 入边 ---------- */
                SELECT  end_entity_id AS current_internal_id,
                        end_entity_id AS current_id,
                        start_entity_id AS neighbor_id,
                        start_entity_id AS neighbor_internal_id,
                        CAST(edge.properties AS CLOB) AS edge_properties,
                        edge.id AS edge_id,
                        CAST(neighbor.properties AS CLOB) AS neighbor_properties,
                        false AS is_outgoing
                FROM    {self.edges_table_name} edge
                JOIN    {self.vertices_table_name} neighbor ON neighbor.entity_id = edge.start_entity_id
                WHERE   edge.end_entity_id IN {placeholders_str}

                UNION ALL

                /* ---------- 3. 孤立点（无邻居） ---------- */
                SELECT  v.entity_id AS current_internal_id,
                        v.entity_id AS current_id,
                        NULL AS neighbor_id,
                        NULL AS neighbor_internal_id,
                        NULL AS edge_properties,
                        NULL AS edge_id,
                        NULL AS neighbor_properties,
                        NULL AS is_outgoing
                FROM    {self.vertices_table_name} v
                WHERE   v.entity_id IN {placeholders_str}
                AND NOT EXISTS (
                        SELECT 1
                        FROM   {self.edges_table_name} e
                        WHERE  e.start_entity_id = v.entity_id
                        OR  e.end_entity_id   = v.entity_id
                    )
                """
            # Execute single optimized query
            assert self.db
            edge_results = await self.db.query(
                edge_query, multirows=True, params=params
            )

            # Create mapping from node ID to node object
            node_map = {node.labels[0]: node for node in current_level_nodes}

            # Process all results in a single loop
            for record in edge_results:
                if not record.get("neighbor_properties") or not record.get(
                    "edge_properties"
                ):
                    continue

                # Get current node information
                current_entity_id = record["current_id"]
                current_node = node_map[current_entity_id]

                # Get neighbor node information
                neighbor_entity_id = record["neighbor_id"]
                neighbor_internal_id = str(record["neighbor_internal_id"])
                is_outgoing = record["is_outgoing"]

                # Determine edge direction
                if is_outgoing:
                    source_id = current_node.id
                    target_id = neighbor_internal_id
                else:
                    source_id = neighbor_internal_id
                    target_id = current_node.id

                if not neighbor_entity_id:
                    continue

                # Get edge and node information
                b_node_properties = json.loads(record["neighbor_properties"])
                rel_properties = json.loads(record["edge_properties"])
                edge_id = str(record["edge_id"])

                # Create neighbor node object
                neighbor_node = KnowledgeGraphNode(
                    id=neighbor_internal_id,
                    labels=[neighbor_entity_id],
                    properties=dict(b_node_properties),
                )

                # Sort entity_ids to ensure (A,B) and (B,A) are treated as the same edge
                sorted_pair = tuple(sorted([current_entity_id, neighbor_entity_id]))

                # Create edge object
                edge = KnowledgeGraphEdge(
                    id=edge_id,
                    type="DIRECTED",
                    source=source_id,
                    target=target_id,
                    properties=dict(rel_properties),
                )

                if neighbor_internal_id in visited_node_ids:
                    # Add backward edge if neighbor node is already visited
                    if (
                        edge_id not in visited_edges
                        and sorted_pair not in visited_edge_pairs
                    ):
                        result.edges.append(edge)
                        visited_edges.add(edge_id)
                        visited_edge_pairs.add(sorted_pair)
                else:
                    if len(visited_node_ids) < max_nodes and current_depth < max_depth:
                        # Add new node to result and queue
                        result.nodes.append(neighbor_node)
                        visited_nodes.add(neighbor_entity_id)
                        visited_node_ids.add(neighbor_internal_id)

                        # Add node to queue with incremented depth
                        queue.append((neighbor_node, current_depth + 1))

                        # Add forward edge
                        if (
                            edge_id not in visited_edges
                            and sorted_pair not in visited_edge_pairs
                        ):
                            result.edges.append(edge)
                            visited_edges.add(edge_id)
                            visited_edge_pairs.add(sorted_pair)
                    else:
                        if current_depth < max_depth:
                            result.is_truncated = True

        return result

    async def get_knowledge_graph(
        self,
        node_label: str,
        max_depth: int = 3,
        max_nodes: int = None,
    ) -> KnowledgeGraph:
        """
        Retrieve a connected subgraph of nodes where the label includes the specified `node_label`.

        Args:
            node_label: Label of the starting node，* means all nodes
            max_depth: Maximum depth of the subgraph, Defaults to 3
            max_nodes: Maxiumu nodes to return by BFS, Defaults to 1000

        Returns:
            KnowledgeGraph object containing nodes and edges, with an is_truncated flag
            indicating whether the graph was truncated due to max_nodes limit
        """
        # ToDo: add get subgraph by node label
        # Get max_nodes from global_config if not provided
        if max_nodes is None:
            max_nodes = self.global_config.get("max_graph_nodes", 1000)
        else:
            # Limit max_nodes to not exceed global_config max_graph_nodes
            max_nodes = min(max_nodes, self.global_config.get("max_graph_nodes", 1000))
        kg = KnowledgeGraph()

        if node_label == "*":
            # First check total node count to determine if graph should be truncated
            query = f"SELECT COUNT(*) as TOTAL_NODES FROM {self.vertices_table_name}"
            assert self.db
            result = await self.db.query(query, multirows=False)
            total_nodes = result["total_nodes"]
            is_truncated = total_nodes > max_nodes
            # Get max_nodes with highest degrees
            # 性能优化：改为单向边
            query = f"""
            SELECT  n.entity_id as entity_id,
                    COALESCE(degree, 0) AS degree
            FROM (
                    /* 1. 全节点（孤立点也出来） */
                    SELECT entity_id
                    FROM   GRAPH_TABLE({self.graph_table_name}
                        MATCH (v is {self.vertices_table_name})
                        COLUMNS (v.entity_id AS entity_id))
                ) n
            LEFT JOIN (
                    /* 2. 无向边：UNION ALL 把两端都当成一行 */
                    SELECT entity_id, COUNT(*) AS degree
                    FROM (
                            /* 出端 */
                            SELECT entity_id AS entity_id
                            FROM   GRAPH_TABLE({self.graph_table_name}
                                MATCH (n is {self.vertices_table_name})-[r is {self.edges_table_name}]->(m is {self.vertices_table_name})
                                COLUMNS (n.entity_id, r.id))
                            UNION ALL
                            /* 入端 */
                            SELECT entity_id AS entity_id
                            FROM  GRAPH_TABLE({self.graph_table_name}
                                MATCH (n is {self.vertices_table_name})-[r is {self.edges_table_name}]->(m is {self.vertices_table_name})
                                COLUMNS (m.entity_id, r.id))
                        ) both_ends
                    GROUP BY entity_id
                ) d
            ON n.entity_id = d.entity_id
            ORDER BY degree DESC
            LIMIT {max_nodes}
            """
            assert self.db
            result = await self.db.query(query, multirows=True)
            node_ids = {row["entity_id"] for row in result}
            logger.info(
                f"[{self.workspace}] Total nodes: {total_nodes}, Selected nodes: {len(node_ids)}"
            )
            if node_ids:
                # Use parameterized query to prevent SQL injection and improve performance
                # Build placeholders and params for the IN clause
                normalized_node_ids = list(node_ids)
                placeholders = [
                    f":node_id_{i}" for i in range(len(normalized_node_ids))
                ]
                placeholders_str = f"({', '.join(placeholders)})"
                params = {
                    f"node_id_{i}": node_id
                    for i, node_id in enumerate(normalized_node_ids)
                }

                # Construct batch query for subgraph within max_nodes
                # 性能优化：改为单向边
                query = f"""
                SELECT  n.start_node_entity_id,
                        n.start_node_properties,
                        edge_id,
                        edge_start_entity_id,
                        edge_end_entity_id,
                        edge_properties,
                        target_node_entity_id,
                        target_node_properties
                FROM (
                        SELECT entity_id AS start_node_entity_id,
                            CAST(properties AS CLOB) AS start_node_properties
                        FROM   GRAPH_TABLE({self.graph_table_name}
                            MATCH (n WHERE n.entity_id IN {placeholders_str})
                            COLUMNS (n.entity_id, n.properties))
                    ) n
                LEFT JOIN (
                        (SELECT
                            start_node_entity_id,
                            CAST(start_node_properties AS CLOB) AS start_node_properties,
                            edge_id,
                            edge_start_entity_id,
                            edge_end_entity_id,
                            CAST(edge_properties AS CLOB) AS edge_properties,
                            target_node_entity_id,
                            CAST(target_node_properties AS CLOB) AS target_node_properties
                        FROM GRAPH_TABLE({self.graph_table_name}
                        MATCH(n is {self.vertices_table_name} WHERE n.entity_id in {placeholders_str})-[r is {self.edges_table_name}]-> (m is {self.vertices_table_name})
                        COLUMNS(
                            n.entity_id as start_node_entity_id,
                            n.properties as start_node_properties,
                            r.id as edge_id,
                            r.start_entity_id as edge_start_entity_id,
                            r.end_entity_id as edge_end_entity_id,
                            r.properties as edge_properties,
                            m.entity_id as target_node_entity_id,
                            m.properties as target_node_properties
                        )))
                        UNION ALL
                        (SELECT
                            start_node_entity_id,
                            CAST(start_node_properties AS CLOB) AS start_node_properties,
                            edge_id,
                            edge_start_entity_id,
                            edge_end_entity_id,
                            CAST(edge_properties AS CLOB) AS edge_properties,
                            target_node_entity_id,
                            CAST(target_node_properties AS CLOB) AS target_node_properties
                        FROM GRAPH_TABLE({self.graph_table_name}
                        MATCH(n is {self.vertices_table_name})-[r is {self.edges_table_name}]-> (m is {self.vertices_table_name} WHERE m.entity_id in {placeholders_str})
                        COLUMNS(
                            n.entity_id as start_node_entity_id,
                            n.properties as start_node_properties,
                            r.id as edge_id,
                            r.start_entity_id as edge_start_entity_id,
                            r.end_entity_id as edge_end_entity_id,
                            r.properties as edge_properties,
                            m.entity_id as target_node_entity_id,
                            m.properties as target_node_properties
                        ))) 
                    ) e
                ON n.start_node_entity_id = e.start_node_entity_id               
                """
                assert self.db
                result = await self.db.query(query, multirows=True, params=params)
                nodes_dict = {}
                edges_dict = {}
                # ToDo: add node and edge to kg
                # postgres_impl.py:4140~4183
                for row in result:
                    node_id = str(row[str("start_node_entity_id").lower()])
                    if node_id not in nodes_dict and (
                        str("start_node_properties").lower() in row
                    ):
                        nodes_dict[node_id] = KnowledgeGraphNode(
                            id=node_id,
                            labels=[
                                json.loads(row["start_node_properties"])["entity_id"]
                            ],
                            properties=json.loads(row["start_node_properties"]),
                        )
                    # Edge
                    if row.get("edge_id") and isinstance(
                        json.loads(row["edge_properties"]), dict
                    ):
                        edge_id = str(row[str("edge_id").lower()])
                        if edge_id not in edges_dict:
                            edges_dict[edge_id] = KnowledgeGraphEdge(
                                id=edge_id,
                                type="DIRECTED",
                                source=row[str("edge_start_entity_id").lower()],
                                target=row[str("edge_end_entity_id").lower()],
                                properties=json.loads(row["edge_properties"]),
                            )
                    # Target Node
                    if row.get("target_node_entity_id") and isinstance(
                        json.loads(row["target_node_properties"]), dict
                    ):
                        target_node_id = str(row[str("target_node_entity_id").lower()])
                        if target_node_id not in nodes_dict:
                            nodes_dict[target_node_id] = KnowledgeGraphNode(
                                id=target_node_id,
                                labels=[
                                    json.loads(row["target_node_properties"])[
                                        "entity_id"
                                    ]
                                ],
                                properties=json.loads(row["target_node_properties"]),
                            )
                kg = KnowledgeGraph(
                    nodes=list(nodes_dict.values()),
                    edges=list(edges_dict.values()),
                    is_truncated=is_truncated,
                )
            else:
                # For single node query, use BFS algorithm
                kg = await self._bfs_subgraph(node_label, max_depth, max_nodes)

            logger.info(
                f"[{self.workspace}] Subgraph query successful | Node count: {len(kg.nodes)} | Edge count: {len(kg.edges)}"
            )
        else:
            # For non-wildcard queries, use the BFS algorithm
            kg = await self._bfs_subgraph(node_label, max_depth, max_nodes)
            logger.info(
                f"[{self.workspace}] Subgraph query for '{node_label}' successful | Node count: {len(kg.nodes)} | Edge count: {len(kg.edges)}"
            )
        return kg

    async def get_nodes_by_chunk_ids(self, chunk_ids: list[str]) -> list[dict]:
        """
        Retrieves nodes from the graph that are associated with a given list of chunk IDs.
        This method uses a PGQL query to efficiently find all nodes
        where the `source_id` property contains any of the specified chunk IDs.

        Args:
            chunk_ids (list[str]): A list of chunk IDs to search for.

        Returns:
            list[dict]: A list of nodes that have a `source_id` property containing any of the specified chunk IDs.
        """
        if not chunk_ids:
            return []
        # Use parameterized query to prevent SQL injection
        # Build placeholders and params for the IN clause
        placeholders = [f":chunk_id_{i}" for i in range(len(chunk_ids))]
        placeholders_str = f"({', '.join(placeholders)})"
        params = {f"chunk_id_{i}": chunk_id for i, chunk_id in enumerate(chunk_ids)}

        query = f"""
            SELECT entity_id, CAST(properties AS CLOB) AS properties FROM {self.vertices_table_name}
            WHERE split(JSON_VALUE(properties, '$.source_id'),'{GRAPH_FIELD_SEP}',2) in {placeholders_str}
            OR split(JSON_VALUE(properties, '$.source_id'),'{GRAPH_FIELD_SEP}',1) in {placeholders_str}
        """
        assert self.db
        results = await self.db.query(query, multirows=True, params=params)
        nodes = []
        for result in results:
            node = result["entity_id"]
            node_dict = dict(json.loads(result["properties"]))
            # Add node id (entity_id) to the dictionary for easier access
            node_dict["id"] = node
            nodes.append(node_dict)

        return nodes

    async def get_edges_by_chunk_ids(self, chunk_ids: list[str]) -> list[dict]:
        """
        Retrieves edges from the graph that are associated with a given list of chunk IDs.
        This method uses a PGQL to efficiently find all edges
        where the `source_id` property contains any of the specified chunk IDs.
        """
        if not chunk_ids:
            return []
        # Use parameterized query to prevent SQL injection
        # Build placeholders and params for the IN clause
        placeholders = [f":chunk_id_{i}" for i in range(len(chunk_ids))]
        placeholders_str = f"({', '.join(placeholders)})"
        params = {f"chunk_id_{i}": chunk_id for i, chunk_id in enumerate(chunk_ids)}

        query = f"""
            SELECT start_entity_id, end_entity_id, CAST(properties AS CLOB) AS properties FROM {self.edges_table_name}
            WHERE split(JSON_VALUE(properties, '$.source_id'),'{GRAPH_FIELD_SEP}',2) in {placeholders_str}
            OR split(JSON_VALUE(properties, '$.source_id'),'{GRAPH_FIELD_SEP}',1) in {placeholders_str}
        """
        assert self.db
        results = await self.db.query(query, multirows=True, params=params)
        edges = []
        for result in results:
            edge = (result["start_entity_id"], result["end_entity_id"])
            edge_dict = dict(json.loads(result["properties"]))
            # Add edge ids (start_entity_id, end_entity_id) to the dictionary for easier access
            edge_dict["source"] = edge[0]
            edge_dict["target"] = edge[1]
            edges.append(edge_dict)
        return edges

    async def get_all_nodes(self) -> list[dict]:
        """Get all nodes in the graph.

        Returns:
            A list of all nodes, where each node is a dictionary of its properties
        """
        query = f"""
            SELECT entity_id, properties FROM {self.vertices_table_name}
        """
        assert self.db
        results = await self.db.query(query, multirows=True)
        nodes = []
        for result in results:
            node = result["entity_id"]
            node_dict = dict(json.loads(result["properties"]))
            # Add node id (entity_id) to the dictionary for easier access
            node_dict["id"] = node
            nodes.append(node_dict)
        return nodes

    async def get_all_edges(self) -> list[dict]:
        """Get all edges in the graph.

        Returns:
            A list of all edges, where each edge is a dictionary of its properties
        """
        query = f"""
            SELECT start_entity_id, end_entity_id, CAST(properties AS CLOB) AS properties FROM {self.edges_table_name}
        """
        assert self.db
        results = await self.db.query(query, multirows=True)
        edges = []
        for result in results:
            edge = (result["start_entity_id"], result["end_entity_id"])
            edge_dict = dict(json.loads(result["properties"]))
            # Add edge ids (start_entity_id, end_entity_id) to the dictionary for easier access
            edge_dict["source"] = edge[0]
            edge_dict["target"] = edge[1]
            edges.append(edge_dict)
        return edges

    async def get_popular_labels(self, limit: int = 300) -> list[str]:
        """Get popular labels by node degree (most connected entities) using native SQL for performance."""
        try:
            # Native SQL query to calculate node degrees directly from edge tables
            # This is significantly faster than using the cypher() function wrapper
            # Use node id as label instead of querying from vertex table properties(entity_id)
            query = f"""
            WITH node_degrees AS (
                SELECT
                    node_id,
                    COUNT(*) AS degree
                FROM (
                    SELECT start_entity_id AS node_id FROM {self.edges_table_name}
                    UNION ALL
                    SELECT end_entity_id AS node_id FROM {self.edges_table_name}
                ) AS all_edges
                GROUP BY node_id
            )
            SELECT
                node_id AS label
            FROM
                node_degrees d
            WHERE
                node_id IS NOT NULL
            ORDER BY
                d.degree DESC,
                label ASC
            LIMIT :limit
            """
            assert self.db
            results = await self.db.query(
                query, params={"limit": limit}, multirows=True
            )
            labels = [
                result["label"] for result in results if result and "label" in result
            ]

            logger.debug(
                f"[{self.workspace}] Retrieved {len(labels)} popular labels (limit: {limit})"
            )
            return labels
        except Exception as e:
            logger.error(f"[{self.workspace}] Error getting popular labels: {str(e)}")
            return []

    async def search_labels(self, query: str, limit: int = 50) -> list[str]:
        """Search labels with fuzzy matching using native, parameterized SQL for performance and security."""
        query_lower = query.lower().strip()
        if not query_lower:
            return []

        try:
            # Re-implementing with the correct agtype access operator and full scoring logic.
            sql_query = f"""
            WITH ranked_labels AS (
            SELECT
                entity_id AS label,
                LOWER(entity_id) AS label_lower
            FROM
                {self.vertices_table_name}
            WHERE
                entity_id IS NOT NULL
                AND LOWER(entity_id) LIKE LOWER('%' || :query_lower || '%')
            )
            SELECT
                label
            FROM (
                SELECT
                    label,
                    CASE
                        WHEN label_lower = LOWER(:query_lower) THEN 1000
                        WHEN label_lower LIKE LOWER(:query_lower || '%') THEN 500
                        ELSE (100 - LENGTH(label))
                    END +
                    CASE
                        WHEN label_lower LIKE LOWER('% ' || :query_lower || '%') OR label_lower LIKE LOWER('%_' || :query_lower || '%') THEN 50
                        ELSE 0
                    END AS score
                FROM
                    ranked_labels
            ) AS scored_labels
            ORDER BY
                score DESC,
                label ASC
            LIMIT :limit
            """
            params = {"query_lower": query_lower, "limit": limit}
            assert self.db
            results = await self.db.query(sql_query, params=params, multirows=True)
            labels = [
                result["label"] for result in results if result and "label" in result
            ]

            logger.debug(
                f"[{self.workspace}] Search query '{query}' returned {len(labels)} results (limit: {limit})"
            )
            return labels
        except Exception as e:
            logger.error(
                f"[{self.workspace}] Error searching labels with query '{query}': {str(e)}"
            )
            return []

    async def drop(self) -> dict[str, str]:
        """Drop all data from current workspace storage and clean up resources

        This method will delete all nodes and relationships in the current workspace only.

        Returns:
            dict[str, str]: Operation status and message
            - On success: {"status": "success", "message": "workspace data dropped"}
            - On failure: {"status": "error", "message": "<error details>"}
        """
        try:
            assert self.db
            await self.db.execute(
                sql=f"DELETE FROM {self.edges_table_name}",
            )
            await self.db.execute(
                sql=f"DELETE FROM {self.vertices_table_name}",
            )

            return {"status": "success", "message": "workspace data dropped"}
        except Exception as e:
            logger.error(f"[{self.workspace}] Error dropping workspace data: {e}")
            return {"status": "error", "message": str(e)}


#######################################################################################################################
# YashanDB as Vector DB storage
#######################################################################################################################

SQL_VDB_TEMPLATES = {
    "upsert_chunk": """INSERT INTO LIGHTRAG_VDB_CHUNKS (workspace, id, tokens,
                      chunk_order_index, full_doc_id, content, content_vector, file_path,
                      create_time, update_time)
                      VALUES (:workspace, :id, :tokens,
                      :chunk_order_index, :full_doc_id, :content, :content_vector, :file_path,
                      :create_time, :update_time)
                      ON DUPLICATE KEY
                      UPDATE tokens=VALUES(tokens),
                      chunk_order_index=VALUES(chunk_order_index),
                      full_doc_id=VALUES(full_doc_id),
                      content = VALUES(content),
                      content_vector=VALUES(content_vector),
                      file_path=VALUES(file_path),
                      update_time = VALUES(update_time)
                     """,
    "upsert_entity": """INSERT INTO LIGHTRAG_VDB_ENTITY (workspace, id, entity_name, content,
                      content_vector, chunk_ids, file_path, create_time, update_time)
                      VALUES (:workspace, :id, :entity_name, :content, :content_vector, :chunk_ids, :file_path, :create_time, :update_time)
                      ON DUPLICATE KEY
                      UPDATE entity_name=VALUES(entity_name),
                      content=VALUES(content),
                      content_vector=VALUES(content_vector),
                      chunk_ids=VALUES(chunk_ids),
                      file_path=VALUES(file_path),
                      update_time=VALUES(update_time)
                     """,
    "upsert_relationship": """INSERT INTO LIGHTRAG_VDB_RELATION (workspace, id, source_id,
                      target_id, content, content_vector, chunk_ids, file_path, create_time, update_time)
                      VALUES (:workspace, :id, :source_id, :target_id, :content, :content_vector, :chunk_ids, :file_path, :create_time, :update_time)
                      ON DUPLICATE KEY
                      UPDATE source_id=VALUES(source_id),
                      target_id=VALUES(target_id),
                      content=VALUES(content),
                      content_vector=VALUES(content_vector),
                      chunk_ids=VALUES(chunk_ids),
                      file_path=VALUES(file_path),
                      update_time = VALUES(update_time)
                     """,
    "relationships": f"""
                     SELECT r.source_id AS src_id,
                            r.target_id AS tgt_id,
                            {extract_epoch_seconds_sql("r.create_time")} AS created_at
                     FROM LIGHTRAG_VDB_RELATION r
                     WHERE r.workspace = :workspace
                       AND r.content_vector <=> :embedding < :threshold
                     ORDER BY r.content_vector <=> :embedding
                     FETCH APPROX FIRST :top_k ROWS ONLY
                     """,
    "entities": f"""
                SELECT e.entity_name,
                       {extract_epoch_seconds_sql("e.create_time")} AS created_at
                FROM LIGHTRAG_VDB_ENTITY e
                WHERE e.workspace = :workspace
                  AND e.content_vector <=> :embedding < :threshold
                ORDER BY e.content_vector <=> :embedding
                FETCH APPROX FIRST :top_k ROWS ONLY
                """,
    "chunks": f"""
              SELECT c.id,
                     c.content,
                     c.file_path,
                     {extract_epoch_seconds_sql("c.create_time")} AS created_at
              FROM LIGHTRAG_VDB_CHUNKS c
              WHERE c.workspace = :workspace
                AND c.content_vector <=> :embedding < :threshold
              ORDER BY c.content_vector <=> :embedding
              FETCH APPROX FIRST :top_k ROWS ONLY
              """,
}


@final
@dataclass
class YashanVectorDBStorage(BaseVectorStorage):
    db: Optional[YashanDB] = None

    def __post_init__(self):
        self._max_batch_size = self.global_config["embedding_batch_num"]
        config = self.global_config.get("vector_db_storage_cls_kwargs", {})
        cosine_threshold = config.get("cosine_better_than_threshold")
        if cosine_threshold is None:
            raise ValueError(
                "cosine_better_than_threshold must be specified in vector_db_storage_cls_kwargs"
            )
        self.cosine_better_than_threshold = cosine_threshold

    async def initialize(self):
        async with get_data_init_lock():
            if self.db is None:
                self.db = await ClientManager.get_client()

            # Implement workspace priority: PostgreSQLDB.workspace > self.workspace > "default"
            if self.db.workspace:
                # Use PostgreSQLDB's workspace (highest priority)
                self.workspace = self.db.workspace
            elif self.workspace:
                # Use storage class's workspace (medium priority)
                pass
            else:
                # Use "default" for compatibility (lowest priority)
                self.workspace = "default"

    async def finalize(self):
        async with get_storage_lock():
            if self.db is not None:
                await ClientManager.release_client(self.db)
                self.db = None

    @classmethod
    def __get_vector(cls, elems: list[Any]) -> array.array:
        return array.array("f", (float(e) for e in elems))

    def _upsert_chunks(
        self, item: dict[str, Any], current_time: datetime.datetime
    ) -> tuple[str, dict[str, Any]]:
        try:
            upsert_sql = SQL_VDB_TEMPLATES["upsert_chunk"]
            data: dict[str, Any] = {
                "workspace": self.workspace,
                "id": item["__id__"],
                "tokens": item["tokens"],
                "chunk_order_index": item["chunk_order_index"],
                "full_doc_id": item["full_doc_id"],
                "content": item["content"],
                "content_vector": self.__get_vector(item["__vector__"].tolist()),
                "file_path": item["file_path"],
                "create_time": current_time,
                "update_time": current_time,
            }
        except Exception as e:
            logger.error(
                f"[{self.workspace}] Error to prepare upsert,\nsql: {e}\nitem: {item}"
            )
            raise

        return upsert_sql, data

    def _upsert_entities(
        self, item: dict[str, Any], current_time: datetime.datetime
    ) -> tuple[str, dict[str, Any]]:
        upsert_sql = SQL_VDB_TEMPLATES["upsert_entity"]
        source_id = item["source_id"]
        if isinstance(source_id, str) and "<SEP>" in source_id:
            chunk_ids = source_id.split("<SEP>")
        else:
            chunk_ids = [source_id]

        data: dict[str, Any] = {
            "workspace": self.workspace,
            "id": item["__id__"],
            "entity_name": item["entity_name"],
            "content": item["content"],
            "content_vector": self.__get_vector(item["__vector__"].tolist()),
            "chunk_ids": chunk_ids,
            "file_path": item.get("file_path", None),
            "create_time": current_time,
            "update_time": current_time,
        }
        return upsert_sql, data

    def _upsert_relationships(
        self, item: dict[str, Any], current_time: datetime.datetime
    ) -> tuple[str, dict[str, Any]]:
        upsert_sql = SQL_VDB_TEMPLATES["upsert_relationship"]
        source_id = item["source_id"]
        if isinstance(source_id, str) and "<SEP>" in source_id:
            chunk_ids = source_id.split("<SEP>")
        else:
            chunk_ids = [source_id]

        data: dict[str, Any] = {
            "workspace": self.workspace,
            "id": item["__id__"],
            "source_id": item["src_id"],
            "target_id": item["tgt_id"],
            "content": item["content"],
            "content_vector": self.__get_vector(item["__vector__"].tolist()),
            "chunk_ids": chunk_ids,
            "file_path": item.get("file_path", None),
            "create_time": current_time,
            "update_time": current_time,
        }
        return upsert_sql, data

    async def upsert(self, data: dict[str, dict[str, Any]]) -> None:
        logger.debug(f"[{self.workspace}] Inserting {len(data)} to {self.namespace}")
        if not data:
            return

        assert self.db

        # Get current UTC time and convert to naive datetime for database storage
        current_time = datetime.datetime.now(timezone.utc).replace(tzinfo=None)
        list_data = [
            {
                "__id__": k,
                **{k1: v1 for k1, v1 in v.items()},
            }
            for k, v in data.items()
        ]
        contents = [v["content"] for v in data.values()]
        batches = [
            contents[i : i + self._max_batch_size]
            for i in range(0, len(contents), self._max_batch_size)
        ]

        embedding_tasks = [self.embedding_func(batch) for batch in batches]
        embeddings_list = await asyncio.gather(*embedding_tasks)

        embeddings = np.concatenate(embeddings_list)
        for i, d in enumerate(list_data):
            d["__vector__"] = embeddings[i]
        for item in list_data:
            if is_namespace(self.namespace, NameSpace.VECTOR_STORE_CHUNKS):
                upsert_sql, data = self._upsert_chunks(item, current_time)
            elif is_namespace(self.namespace, NameSpace.VECTOR_STORE_ENTITIES):
                upsert_sql, data = self._upsert_entities(item, current_time)
            elif is_namespace(self.namespace, NameSpace.VECTOR_STORE_RELATIONSHIPS):
                upsert_sql, data = self._upsert_relationships(item, current_time)
            else:
                raise ValueError(f"{self.namespace} is not supported")

            await self.db.execute(upsert_sql, data)

    #################### query method ###############
    async def query(
        self,
        query: str,
        top_k: int,
        query_embedding: Optional[list[float]] = None,
    ) -> list[dict[str, Any]]:
        assert self.db

        if query_embedding is not None:
            embedding = query_embedding
        else:
            # higher priority for query
            embeddings = await self.embedding_func([query], _priority=5)
            embedding = embeddings[0]

        embedding = self.__get_vector(embedding)
        sql = SQL_VDB_TEMPLATES[self.namespace]
        params = {
            "workspace": self.workspace,
            "embedding": embedding,
            "threshold": 1 - self.cosine_better_than_threshold,
            "top_k": top_k,
        }
        results = await self.db.query(sql, params=params, multirows=True)
        return results

    async def index_done_callback(self) -> None:
        # PG handles persistence automatically
        pass

    async def delete(self, ids: list[str]) -> None:
        """Delete vectors with specified IDs from the storage.

        Args:
            ids: List of vector IDs to be deleted
        """
        if not ids:
            return

        table_name = namespace_to_table_name(self.namespace)
        if not table_name:
            logger.error(
                f"[{self.workspace}] Unknown namespace for vector deletion: {self.namespace}"
            )
            return

        ids_str = ids_format_query(ids)
        delete_sql = (
            f"DELETE FROM {table_name} WHERE workspace=:workspace AND id = {ids_str}"
        )

        try:
            assert self.db
            await self.db.execute(delete_sql, {"workspace": self.workspace})
            logger.debug(
                f"[{self.workspace}] Successfully deleted {len(ids)} vectors from {self.namespace}"
            )
        except Exception as e:
            logger.error(
                f"[{self.workspace}] Error while deleting vectors from {self.namespace}: {e}"
            )

    async def delete_entity(self, entity_name: str) -> None:
        """Delete an entity by its name from the vector storage.

        Args:
            entity_name: The name of the entity to delete
        """
        try:
            # Construct SQL to delete the entity
            delete_sql = """DELETE FROM LIGHTRAG_VDB_ENTITY
                            WHERE workspace=:workspace AND entity_name=:entity_name"""

            assert self.db
            await self.db.execute(
                delete_sql, {"workspace": self.workspace, "entity_name": entity_name}
            )
            logger.debug(
                f"[{self.workspace}] Successfully deleted entity {entity_name}"
            )
        except Exception as e:
            logger.error(f"[{self.workspace}] Error deleting entity {entity_name}: {e}")

    async def delete_entity_relation(self, entity_name: str) -> None:
        """Delete all relations associated with an entity.

        Args:
            entity_name: The name of the entity whose relations should be deleted
        """
        try:
            # Delete relations where the entity is either the source or target
            delete_sql = """DELETE FROM LIGHTRAG_VDB_RELATION
                            WHERE workspace=:workspace AND :entity_name IN (source_id, target_id)"""

            assert self.db
            await self.db.execute(
                delete_sql, {"workspace": self.workspace, "entity_name": entity_name}
            )
            logger.debug(
                f"[{self.workspace}] Successfully deleted relations for entity {entity_name}"
            )
        except Exception as e:
            logger.error(
                f"[{self.workspace}] Error deleting relations for entity {entity_name}: {e}"
            )

    async def get_by_id(self, id: str) -> dict[str, Any] | None:
        """Get vector data by its ID

        Args:
            id: The unique identifier of the vector

        Returns:
            The vector data if found, or None if not found
        """
        table_name = namespace_to_table_name(self.namespace)
        if not table_name:
            logger.error(
                f"[{self.workspace}] Unknown namespace for ID lookup: {self.namespace}"
            )
            return None

        query = f"""
        SELECT *,
               {extract_epoch_seconds_sql("create_time")} AS created_at
        FROM {table_name}
        WHERE workspace=:workspace AND id=:id"""
        params = {"workspace": self.workspace, "id": id}

        try:
            assert self.db
            return await self.db.query(query, params)
        except Exception as e:
            logger.error(
                f"[{self.workspace}] Error retrieving vector data for ID {id}: {e}"
            )
            return None

    async def get_by_ids(self, ids: list[str]) -> list[dict[str, Any]]:
        """Get multiple vector data by their IDs

        Args:
            ids: List of unique identifiers

        Returns:
            List of vector data objects that were found
        """
        if not ids:
            return []

        table_name = namespace_to_table_name(self.namespace)
        if not table_name:
            logger.error(
                f"[{self.workspace}] Unknown namespace for IDs lookup: {self.namespace}"
            )
            return []

        ids_str = ",".join([f"'{id}'" for id in ids])
        query = f"""
        SELECT *,
               {extract_epoch_seconds_sql("create_time")} AS created_at
        FROM {table_name}
        WHERE workspace=:workspace AND id IN ({ids_str})"""
        params = {"workspace": self.workspace}

        try:
            assert self.db
            return await self.db.query(query, params, multirows=True)
        except Exception as e:
            logger.error(
                f"[{self.workspace}] Error retrieving vector data for IDs {ids}: {e}"
            )
            return []

    async def get_vectors_by_ids(self, ids: list[str]) -> dict[str, list[float]]:
        """Get vectors by their IDs, returning only ID and vector data for efficiency

        Args:
            ids: List of unique identifiers

        Returns:
            Dictionary mapping IDs to their vector embeddings
            Format: {id: [vector_values], ...}
        """
        if not ids:
            return {}

        table_name = namespace_to_table_name(self.namespace)
        if not table_name:
            logger.error(
                f"[{self.workspace}] Unknown namespace for vector lookup: {self.namespace}"
            )
            return {}

        ids_str = ",".join([f"'{id}'" for id in ids])
        query = f"SELECT id, content_vector FROM {table_name} WHERE workspace=:workspace AND id IN ({ids_str})"
        params = {"workspace": self.workspace}

        try:
            assert self.db
            results = await self.db.query(query, params, multirows=True)
            vectors_dict: dict[str, list[float]] = {}

            for result in results:
                if result and "content_vector" in result and "id" in result:
                    try:
                        assert isinstance(result["content_vector"], array.array)
                        vectors_dict[result["id"]] = result["content_vector"].tolist()
                    except (json.JSONDecodeError, TypeError) as e:
                        logger.warning(
                            f"[{self.workspace}] Failed to parse vector data for ID {result['id']}: {e}"
                        )

            return vectors_dict
        except Exception as e:
            logger.error(
                f"[{self.workspace}] Error retrieving vectors by IDs from {self.namespace}: {e}"
            )
            return {}

    async def drop(self) -> dict[str, str]:
        """Drop the storage"""
        async with get_storage_lock():
            try:
                table_name = namespace_to_table_name(self.namespace)
                if not table_name:
                    return {
                        "status": "error",
                        "message": f"Unknown namespace: {self.namespace}",
                    }

                drop_sql = SQL_COMMON_TEMPLATES["drop_specific_table_workspace"].format(
                    table_name=table_name
                )
                assert self.db
                await self.db.execute(drop_sql, {"workspace": self.workspace})
                return {"status": "success", "message": "data dropped"}
            except Exception as e:
                return {"status": "error", "message": str(e)}


#######################################################################################################################
# YashanDB as Doc Status storage
#######################################################################################################################


@final
@dataclass
class YashanDocStatusStorage(DocStatusStorage):
    db: Optional[YashanDB] = None

    @overload
    def _format_datetime_with_timezone(self, dt: None) -> None: ...

    @overload
    def _format_datetime_with_timezone(self, dt: datetime.datetime) -> str: ...

    def _format_datetime_with_timezone(self, dt: Optional[datetime.datetime]):
        """Convert datetime to ISO format string with timezone info"""
        if dt is None:
            return None
        # If no timezone info, assume it's UTC time (as stored in database)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        # If datetime already has timezone info, keep it as is
        return dt.isoformat()

    async def initialize(self):
        async with get_data_init_lock():
            if self.db is None:
                self.db = await ClientManager.get_client()

            # Implement workspace priority: PostgreSQLDB.workspace > self.workspace > "default"
            if self.db.workspace:
                # Use PostgreSQLDB's workspace (highest priority)
                self.workspace = self.db.workspace
            elif hasattr(self, "workspace") and self.workspace:
                # Use storage class's workspace (medium priority)
                pass
            else:
                # Use "default" for compatibility (lowest priority)
                self.workspace = "default"

    async def finalize(self):
        async with get_storage_lock():
            if self.db is not None:
                await ClientManager.release_client(self.db)
                self.db = None

    async def filter_keys(self, keys: set[str]) -> set[str]:
        """Filter out duplicated content"""
        sql = SQL_COMMON_TEMPLATES["filter_keys"].format(
            table_name=namespace_to_table_name(self.namespace),
            ids=",".join([f"'{id}'" for id in keys]),
        )
        params = {"workspace": self.workspace}
        try:
            assert self.db

            # Only query if keys is not empty
            if keys and (res := await self.db.query(sql, params, multirows=True)):
                exist_keys = [key["id"] for key in res]
            else:
                exist_keys = []
            new_keys = set(s for s in keys if s not in exist_keys)
            return new_keys
        except Exception as e:
            logger.error(
                f"[{self.workspace}] YashanDB database,\nsql:{sql},\nparams:{params},\nerror:{e}"
            )
            raise

    async def get_by_id(self, id: str) -> Optional[dict[str, Any]]:
        sql = "select * from LIGHTRAG_DOC_STATUS where workspace=:workspace and id=:id"
        params = {"workspace": self.workspace, "id": id}
        assert self.db
        result = await self.db.query(sql, params, multirows=False)
        if result is None:
            return None

        # Parse chunks_list JSON string back to list
        chunks_list = result.get("chunks_list", [])
        if isinstance(chunks_list, str):
            try:
                chunks_list = json.loads(chunks_list)
            except json.JSONDecodeError:
                chunks_list = []

        # Parse metadata JSON string back to dict
        metadata = result.get("metadata", {})
        if isinstance(metadata, str):
            try:
                metadata = json.loads(metadata)
            except json.JSONDecodeError:
                metadata = {}

        # Convert datetime objects to ISO format strings with timezone info
        created_at = self._format_datetime_with_timezone(result["created_at"])
        updated_at = self._format_datetime_with_timezone(result["updated_at"])

        return dict(
            content_length=result["content_length"],
            content_summary=result["content_summary"],
            status=result["status"],
            chunks_count=result["chunks_count"],
            created_at=created_at,
            updated_at=updated_at,
            file_path=result["file_path"],
            chunks_list=chunks_list,
            metadata=metadata,
            error_msg=result.get("error_msg"),
            track_id=result.get("track_id"),
        )

    async def get_by_ids(self, ids: list[str]) -> list[dict[str, Any]]:
        """Get doc_chunks data by multiple IDs."""
        if not ids:
            return []

        sql = f"SELECT * FROM LIGHTRAG_DOC_STATUS WHERE workspace=:workspace AND id = {ids_format_query(ids)}"
        params = {"workspace": self.workspace}

        assert self.db
        results = await self.db.query(sql, params, True)

        if not results:
            return []

        processed_results = []
        for row in results:
            # Parse chunks_list JSON string back to list
            chunks_list = row.get("chunks_list", [])
            if isinstance(chunks_list, str):
                try:
                    chunks_list = json.loads(chunks_list)
                except json.JSONDecodeError:
                    chunks_list = []

            # Parse metadata JSON string back to dict
            metadata = row.get("metadata", {})
            if isinstance(metadata, str):
                try:
                    metadata = json.loads(metadata)
                except json.JSONDecodeError:
                    metadata = {}

            # Convert datetime objects to ISO format strings with timezone info
            created_at = self._format_datetime_with_timezone(row["created_at"])
            updated_at = self._format_datetime_with_timezone(row["updated_at"])

            processed_results.append(
                {
                    "content_length": row["content_length"],
                    "content_summary": row["content_summary"],
                    "status": row["status"],
                    "chunks_count": row["chunks_count"],
                    "created_at": created_at,
                    "updated_at": updated_at,
                    "file_path": row["file_path"],
                    "chunks_list": chunks_list,
                    "metadata": metadata,
                    "error_msg": row.get("error_msg"),
                    "track_id": row.get("track_id"),
                }
            )

        return processed_results

    async def get_doc_by_file_path(self, file_path: str) -> Union[dict[str, Any], None]:
        """Get document by file path

        Args:
            file_path: The file path to search for

        Returns:
            Union[dict[str, Any], None]: Document data if found, None otherwise
            Returns the same format as get_by_id method
        """
        sql = "select * from LIGHTRAG_DOC_STATUS where workspace=:workspace and file_path=:file_path"
        params = {"workspace": self.workspace, "file_path": file_path}
        assert self.db
        result = await self.db.query(sql, params, True)

        if result is None or result == []:
            return None
        else:
            # Parse chunks_list JSON string back to list
            chunks_list = result[0].get("chunks_list", [])
            if isinstance(chunks_list, str):
                try:
                    chunks_list = json.loads(chunks_list)
                except json.JSONDecodeError:
                    chunks_list = []

            # Parse metadata JSON string back to dict
            metadata = result[0].get("metadata", {})
            if isinstance(metadata, str):
                try:
                    metadata = json.loads(metadata)
                except json.JSONDecodeError:
                    metadata = {}

            # Convert datetime objects to ISO format strings with timezone info
            created_at = self._format_datetime_with_timezone(result[0]["created_at"])
            updated_at = self._format_datetime_with_timezone(result[0]["updated_at"])

            return dict(
                content_length=result[0]["content_length"],
                content_summary=result[0]["content_summary"],
                status=result[0]["status"],
                chunks_count=result[0]["chunks_count"],
                created_at=created_at,
                updated_at=updated_at,
                file_path=result[0]["file_path"],
                chunks_list=chunks_list,
                metadata=metadata,
                error_msg=result[0].get("error_msg"),
                track_id=result[0].get("track_id"),
            )

    async def get_status_counts(self) -> dict[str, int]:
        """Get counts of documents in each status"""
        sql = """SELECT status, COUNT(1) as count
                   FROM LIGHTRAG_DOC_STATUS
                  where workspace=:workspace GROUP BY STATUS
                 """
        params = {"workspace": self.workspace}
        assert self.db
        result = await self.db.query(sql, params, True)
        counts = {}
        for doc in result:
            counts[doc["status"]] = doc["count"]
        return counts

    async def get_docs_by_status(
        self,
        status: DocStatus,
    ) -> dict[str, DocProcessingStatus]:
        """all documents with a specific status"""
        sql = "select * from LIGHTRAG_DOC_STATUS where workspace=:workspace and status=:status"
        params = {"workspace": self.workspace, "status": status.value}
        assert self.db
        result = await self.db.query(sql, params, True)

        docs_by_status = {}
        for element in result:
            # Parse chunks_list JSON string back to list
            chunks_list = element.get("chunks_list", [])
            if isinstance(chunks_list, str):
                try:
                    chunks_list = json.loads(chunks_list)
                except json.JSONDecodeError:
                    chunks_list = []

            # Parse metadata JSON string back to dict
            metadata = element.get("metadata", {})
            if isinstance(metadata, str):
                try:
                    metadata = json.loads(metadata)
                except json.JSONDecodeError:
                    metadata = {}
            # Ensure metadata is a dict
            if not isinstance(metadata, dict):
                metadata = {}

            # Safe handling for file_path
            file_path = element.get("file_path")
            if file_path is None:
                file_path = "no-file-path"

            # Convert datetime objects to ISO format strings with timezone info
            created_at = self._format_datetime_with_timezone(element["created_at"])
            updated_at = self._format_datetime_with_timezone(element["updated_at"])

            docs_by_status[element["id"]] = DocProcessingStatus(
                content_summary=element["content_summary"],
                content_length=element["content_length"],
                status=element["status"],
                created_at=created_at,
                updated_at=updated_at,
                chunks_count=element["chunks_count"],
                file_path=file_path,
                chunks_list=chunks_list,
                metadata=metadata,
                error_msg=element.get("error_msg"),
                track_id=element.get("track_id"),
            )

        return docs_by_status

    async def get_docs_by_track_id(
        self, track_id: str
    ) -> dict[str, DocProcessingStatus]:
        """Get all documents with a specific track_id"""
        sql = "select * from LIGHTRAG_DOC_STATUS where workspace=:workspace and track_id=:track_id"
        params = {"workspace": self.workspace, "track_id": track_id}
        assert self.db
        result = await self.db.query(sql, params, True)

        docs_by_track_id = {}
        for element in result:
            # Parse chunks_list JSON string back to list
            chunks_list = element.get("chunks_list", [])
            if isinstance(chunks_list, str):
                try:
                    chunks_list = json.loads(chunks_list)
                except json.JSONDecodeError:
                    chunks_list = []

            # Parse metadata JSON string back to dict
            metadata = element.get("metadata", {})
            if isinstance(metadata, str):
                try:
                    metadata = json.loads(metadata)
                except json.JSONDecodeError:
                    metadata = {}
            # Ensure metadata is a dict
            if not isinstance(metadata, dict):
                metadata = {}

            # Safe handling for file_path
            file_path = element.get("file_path")
            if file_path is None:
                file_path = "no-file-path"

            # Convert datetime objects to ISO format strings with timezone info
            created_at = self._format_datetime_with_timezone(element["created_at"])
            updated_at = self._format_datetime_with_timezone(element["updated_at"])

            docs_by_track_id[element["id"]] = DocProcessingStatus(
                content_summary=element["content_summary"],
                content_length=element["content_length"],
                status=element["status"],
                created_at=created_at,
                updated_at=updated_at,
                chunks_count=element["chunks_count"],
                file_path=file_path,
                chunks_list=chunks_list,
                track_id=element.get("track_id"),
                metadata=metadata,
                error_msg=element.get("error_msg"),
            )

        return docs_by_track_id

    async def get_docs_paginated(
        self,
        status_filter: Optional[DocStatus] = None,
        page: int = 1,
        page_size: int = 50,
        sort_field: str = "updated_at",
        sort_direction: str = "desc",
    ) -> tuple[list[tuple[str, DocProcessingStatus]], int]:
        """Get documents with pagination support

        Args:
            status_filter: Filter by document status, None for all statuses
            page: Page number (1-based)
            page_size: Number of documents per page (10-200)
            sort_field: Field to sort by ('created_at', 'updated_at', 'id')
            sort_direction: Sort direction ('asc' or 'desc')

        Returns:
            Tuple of (list of (doc_id, DocProcessingStatus) tuples, total_count)
        """
        # Validate parameters
        if page < 1:
            page = 1
        if page_size < 10:
            page_size = 10
        elif page_size > 200:
            page_size = 200

        if sort_field not in ["created_at", "updated_at", "id", "file_path"]:
            sort_field = "updated_at"

        if sort_direction.lower() not in ["asc", "desc"]:
            sort_direction = "desc"

        # Calculate offset
        offset = (page - 1) * page_size

        # Build WHERE clause
        where_clause = "WHERE workspace=:workspace"
        params: dict[str, Any] = {"workspace": self.workspace}
        # param_count = 1

        if status_filter is not None:
            # param_count += 1
            where_clause += " AND status=:status"
            params["status"] = status_filter.value

        # Build ORDER BY clause
        order_clause = f"ORDER BY {sort_field} {sort_direction.upper()}"

        # Query for total count
        count_sql = f"SELECT COUNT(*) as total FROM LIGHTRAG_DOC_STATUS {where_clause}"
        assert self.db
        count_result = await self.db.query(count_sql, params)
        total_count = count_result["total"] if count_result else 0

        # Query for paginated data
        data_sql = f"""
            SELECT * FROM LIGHTRAG_DOC_STATUS
            {where_clause}
            {order_clause}
            LIMIT :limit OFFSET :offset
        """
        params["limit"] = page_size
        params["offset"] = offset

        result = await self.db.query(data_sql, params, True)

        # Convert to (doc_id, DocProcessingStatus) tuples
        documents = []
        for element in result:
            doc_id = element["id"]

            # Parse chunks_list JSON string back to list
            chunks_list = element.get("chunks_list", [])
            if isinstance(chunks_list, str):
                try:
                    chunks_list = json.loads(chunks_list)
                except json.JSONDecodeError:
                    chunks_list = []

            # Parse metadata JSON string back to dict
            metadata = element.get("metadata", {})
            if isinstance(metadata, str):
                try:
                    metadata = json.loads(metadata)
                except json.JSONDecodeError:
                    metadata = {}

            # Convert datetime objects to ISO format strings with timezone info
            created_at = self._format_datetime_with_timezone(element["created_at"])
            updated_at = self._format_datetime_with_timezone(element["updated_at"])

            doc_status = DocProcessingStatus(
                content_summary=element["content_summary"],
                content_length=element["content_length"],
                status=element["status"],
                created_at=created_at,
                updated_at=updated_at,
                chunks_count=element["chunks_count"],
                file_path=element["file_path"],
                chunks_list=chunks_list,
                track_id=element.get("track_id"),
                metadata=metadata,
                error_msg=element.get("error_msg"),
            )
            documents.append((doc_id, doc_status))

        return documents, total_count

    async def get_all_status_counts(self) -> dict[str, int]:
        """Get counts of documents in each status for all documents

        Returns:
            Dictionary mapping status names to counts, including 'all' field
        """
        sql = """
            SELECT status, COUNT(*) as count
            FROM LIGHTRAG_DOC_STATUS
            WHERE workspace=:workspace
            GROUP BY status
        """
        params = {"workspace": self.workspace}
        assert self.db
        result = await self.db.query(sql, params, True)

        counts = {}
        total_count = 0
        for row in result:
            counts[row["status"]] = row["count"]
            total_count += row["count"]

        # Add 'all' field with total count
        counts["all"] = total_count

        return counts

    async def index_done_callback(self) -> None:
        # PG handles persistence automatically
        pass

    async def delete(self, ids: list[str]) -> None:
        """Delete specific records from storage by their IDs

        Args:
            ids (list[str]): List of document IDs to be deleted from storage

        Returns:
            None
        """
        if not ids:
            return

        table_name = namespace_to_table_name(self.namespace)
        if not table_name:
            logger.error(
                f"[{self.workspace}] Unknown namespace for deletion: {self.namespace}"
            )
            return

        delete_sql = f"DELETE FROM {table_name} WHERE workspace=:workspace AND id = {ids_format_query(ids)}"

        try:
            assert self.db
            await self.db.execute(delete_sql, {"workspace": self.workspace})
            logger.debug(
                f"[{self.workspace}] Successfully deleted {len(ids)} records from {self.namespace}"
            )
        except Exception as e:
            logger.error(
                f"[{self.workspace}] Error while deleting records from {self.namespace}: {e}"
            )

    async def upsert(self, data: dict[str, dict[str, Any]]) -> None:
        """Update or insert document status

        Args:
            data: dictionary of document IDs and their status data
        """
        logger.debug(f"[{self.workspace}] Inserting {len(data)} to {self.namespace}")
        if not data:
            return

        assert self.db

        def parse_datetime(dt_str):
            """Parse datetime and ensure it's stored as UTC time in database"""
            if dt_str is None:
                return None
            if isinstance(dt_str, (datetime.date, datetime.datetime)):
                # If it's a datetime object
                if isinstance(dt_str, datetime.datetime):
                    # If no timezone info, assume it's UTC
                    if dt_str.tzinfo is None:
                        dt_str = dt_str.replace(tzinfo=timezone.utc)
                    # Convert to UTC and remove timezone info for storage
                    return dt_str.astimezone(timezone.utc).replace(tzinfo=None)
                return dt_str
            try:
                # Process ISO format string with timezone
                dt = datetime.datetime.fromisoformat(dt_str)
                # If no timezone info, assume it's UTC
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                # Convert to UTC and remove timezone info for storage
                return dt.astimezone(timezone.utc).replace(tzinfo=None)
            except (ValueError, TypeError):
                logger.warning(
                    f"[{self.workspace}] Unable to parse datetime string: {dt_str}"
                )
                return None

        # Modified SQL to include created_at, updated_at, chunks_list, track_id, metadata, and error_msg in both INSERT and UPDATE operations
        # All fields are updated from the input data in both INSERT and UPDATE cases
        sql = """insert into LIGHTRAG_DOC_STATUS(workspace,id,content_summary,content_length,chunks_count,status,file_path,chunks_list,track_id,metadata,error_msg,created_at,updated_at)
                 values(:workspace, :id, :content_summary, :content_length, :chunks_count, :status, :file_path, :chunks_list, :track_id, :metadata, :error_msg, :created_at, :updated_at)
                  ON DUPLICATE KEY UPDATE
                  content_summary = VALUES(content_summary),
                  content_length = VALUES(content_length),
                  chunks_count = VALUES(chunks_count),
                  status = VALUES(status),
                  file_path = VALUES(file_path),
                  chunks_list = VALUES(chunks_list),
                  track_id = VALUES(track_id),
                  metadata = VALUES(metadata),
                  error_msg = VALUES(error_msg),
                  created_at = VALUES(created_at),
                  updated_at = VALUES(updated_at)"""
        for k, v in data.items():
            # Remove timezone information, store utc time in db
            created_at = parse_datetime(v.get("created_at"))
            updated_at = parse_datetime(v.get("updated_at"))

            # chunks_count, chunks_list, track_id, metadata, and error_msg are optional
            await self.db.execute(
                sql,
                {
                    "workspace": self.workspace,
                    "id": k,
                    "content_summary": v["content_summary"],
                    "content_length": v["content_length"],
                    "chunks_count": v["chunks_count"] if "chunks_count" in v else -1,
                    "status": v["status"],
                    "file_path": v["file_path"],
                    "chunks_list": json.dumps(v.get("chunks_list", [])),
                    "track_id": v.get("track_id"),  # Add track_id support
                    "metadata": json.dumps(
                        v.get("metadata", {})
                    ),  # Add metadata support
                    "error_msg": v.get("error_msg"),  # Add error_msg support
                    "created_at": created_at,  # Use the converted datetime object
                    "updated_at": updated_at,  # Use the converted datetime object
                },
            )

    async def drop(self) -> dict[str, str]:
        """Drop the storage"""
        async with get_storage_lock():
            try:
                assert self.db
                table_name = namespace_to_table_name(self.namespace)
                if not table_name:
                    return {
                        "status": "error",
                        "message": f"Unknown namespace: {self.namespace}",
                    }

                drop_sql = SQL_COMMON_TEMPLATES["drop_specific_table_workspace"].format(
                    table_name=table_name
                )
                await self.db.execute(drop_sql, {"workspace": self.workspace})
                return {"status": "success", "message": "data dropped"}
            except Exception as e:
                return {"status": "error", "message": str(e)}
