"""
SQLiteDB — 线程安全的 SQLite 连接管理器。

每个线程维护独立连接（thread-local），避免多线程竞争。
生产环境可直接替换为 PostgreSQL（修改 db.py，接口不变）。

默认数据库路径：data/shopplan.db（项目根目录下）
"""

from __future__ import annotations

import sqlite3
import threading
from pathlib import Path
from typing import Any, Optional


_DEFAULT_DB_PATH = Path(__file__).parent.parent.parent / "data" / "shopplan.db"


class SQLiteDB:
    """
    线程安全 SQLite 连接管理器。

    用法：
        db = SQLiteDB()
        db.execute("CREATE TABLE IF NOT EXISTS ...")
        rows = db.fetchall("SELECT * FROM users WHERE id=?", (user_id,))
    """

    def __init__(self, db_path: Optional[str] = None):
        self._path = Path(db_path) if db_path else _DEFAULT_DB_PATH
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()

    @property
    def conn(self) -> sqlite3.Connection:
        """获取当前线程的 SQLite 连接（懒初始化）。"""
        if not hasattr(self._local, "conn") or self._local.conn is None:
            conn = sqlite3.connect(
                str(self._path),
                check_same_thread=False,
                timeout=10.0,
            )
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")   # 提升并发写入性能
            conn.execute("PRAGMA synchronous=NORMAL")
            self._local.conn = conn
        return self._local.conn

    def execute(self, sql: str, params: tuple = ()) -> None:
        """执行 DML（INSERT/UPDATE/DELETE/CREATE）并提交。"""
        self.conn.execute(sql, params)
        self.conn.commit()

    def executemany(self, sql: str, params_list: list[tuple]) -> None:
        self.conn.executemany(sql, params_list)
        self.conn.commit()

    def fetchall(self, sql: str, params: tuple = ()) -> list[sqlite3.Row]:
        return self.conn.execute(sql, params).fetchall()

    def fetchone(self, sql: str, params: tuple = ()) -> Optional[sqlite3.Row]:
        return self.conn.execute(sql, params).fetchone()

    def table_exists(self, table_name: str) -> bool:
        row = self.fetchone(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
            (table_name,),
        )
        return row is not None

    def close(self) -> None:
        if hasattr(self._local, "conn") and self._local.conn:
            self._local.conn.close()
            self._local.conn = None


# 全局单例
_db_instance: Optional[SQLiteDB] = None
_db_lock = threading.Lock()


def get_db(db_path: Optional[str] = None) -> SQLiteDB:
    """获取全局 SQLiteDB 单例。"""
    global _db_instance
    if _db_instance is None:
        with _db_lock:
            if _db_instance is None:
                _db_instance = SQLiteDB(db_path)
    return _db_instance
