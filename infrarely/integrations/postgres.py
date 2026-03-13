"""
aos/integrations/postgres.py — PostgreSQL Integration
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from infrarely.integrations import Integration


class PostgresIntegration(Integration):
    """PostgreSQL integration providing database tools."""

    name = "postgres"
    description = "PostgreSQL database integration"
    required_config = ["connection_string"]

    def __init__(
        self, *, connection_string: str = "", connection: Any = None, **config
    ):
        self._connection_string = connection_string or config.get(
            "connection_string", ""
        )
        self._connection = connection
        super().__init__(**config)

    def _setup(self) -> None:
        self._tools = {
            "query": self.query,
            "execute": self.execute,
            "list_tables": self.list_tables,
            "describe_table": self.describe_table,
        }

    def _get_connection(self) -> Any:
        """Get or create a database connection."""
        if self._connection is not None:
            return self._connection
        try:
            import psycopg2

            self._connection = psycopg2.connect(self._connection_string)
            return self._connection
        except ImportError:
            raise ImportError(
                "psycopg2 is required for PostgreSQL integration. "
                "Install it: pip install psycopg2-binary"
            )

    def query(self, sql: str, params: Optional[Tuple] = None) -> List[Dict[str, Any]]:
        """Execute a SELECT query and return results as dicts."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(sql, params or ())
        columns = [desc[0] for desc in cursor.description or []]
        rows = cursor.fetchall()
        return [dict(zip(columns, row)) for row in rows]

    def execute(self, sql: str, params: Optional[Tuple] = None) -> int:
        """Execute a SQL statement and return affected row count."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(sql, params or ())
        conn.commit()
        return cursor.rowcount

    def list_tables(self) -> List[str]:
        """List all tables in the database."""
        results = self.query(
            "SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'"
        )
        return [r["table_name"] for r in results]

    def describe_table(self, table_name: str) -> List[Dict[str, Any]]:
        """Describe a table's columns."""
        return self.query(
            "SELECT column_name, data_type, is_nullable "
            "FROM information_schema.columns WHERE table_name = %s",
            (table_name,),
        )
