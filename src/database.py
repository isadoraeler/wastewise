import sqlite3
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATABASE_PATH = PROJECT_ROOT / "data" / "wastewise.db"


def get_connection():
    DATABASE_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    connection = sqlite3.connect(DATABASE_PATH)
    connection.row_factory = sqlite3.Row

    return connection


def initialize_database():
    with get_connection() as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS analyses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                predicted_category TEXT NOT NULL,
                confidence REAL NOT NULL,
                second_category TEXT NOT NULL,
                second_confidence REAL NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )


def save_analysis(
    predicted_category,
    confidence,
    second_category,
    second_confidence,
):
    created_at = datetime.now().astimezone().isoformat(
        timespec="seconds"
    )

    with get_connection() as connection:
        connection.execute(
            """
            INSERT INTO analyses (
                predicted_category,
                confidence,
                second_category,
                second_confidence,
                created_at
            )
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                predicted_category,
                confidence,
                second_category,
                second_confidence,
                created_at,
            ),
        )


def get_analysis_history():
    with get_connection() as connection:
        rows = connection.execute(
            """
            SELECT
                id,
                predicted_category,
                confidence,
                second_category,
                second_confidence,
                created_at
            FROM analyses
            ORDER BY id DESC
            """
        ).fetchall()

    return [dict(row) for row in rows]


initialize_database()