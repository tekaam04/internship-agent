"""
db.py - the SQLite database the agent reads listings from.

    uv run python db.py              # load the default snapshot
    uv run python db.py 2026-11-07   # load a different snapshot

The listings table holds one row per SimplifyJobs listing from one snapshot.
List fields (locations, terms, degrees) are stored as JSON text.
"""
from __future__ import annotations
import json
import sqlite3
import sys
from pathlib import Path

from sources import DEFAULT_SNAPSHOT, ats_of, load_simplify_snapshot

DB_PATH = Path(__file__).parent / "data" / "agent.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS listings (
    id TEXT PRIMARY KEY,
    company TEXT NOT NULL,
    title TEXT NOT NULL,
    category TEXT,
    locations TEXT,
    terms TEXT,
    degrees TEXT,
    sponsorship TEXT,
    date_posted INTEGER,
    url TEXT,
    ats TEXT,
    active INTEGER,
    snapshot TEXT
);
"""


def connect() -> sqlite3.Connection:
    """connect - open the database and make sure the tables exist.

    Returns:
        sqlite3.Connection: A connection whose rows can be read by column
        name.
    """
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    return conn


def load_listings(snapshot: str = DEFAULT_SNAPSHOT) -> int:
    """load_listings - replace the listings table with one snapshot.

    Parameters:
        snapshot (str): The snapshot date to load.

    Returns:
        int: The number of listings stored.

    Description:
        Listings that are hidden on SimplifyJobs are skipped. Inactive ones
        are kept with ``active = 0`` so a run can explain why a role it found
        earlier is no longer offered.
    """
    rows = [
        (item["id"], item["company_name"], item["title"], item.get("category"),
         json.dumps(item.get("locations") or []), json.dumps(item.get("terms") or []),
         json.dumps(item.get("degrees") or []), item.get("sponsorship"),
         item.get("date_posted"), item.get("url"), ats_of(item.get("url") or ""),
         int(bool(item.get("active"))), snapshot)
        for item in load_simplify_snapshot(snapshot)
        if item.get("is_visible", True)
    ]
    with connect() as conn:
        conn.execute("DELETE FROM listings")
        conn.executemany("INSERT OR REPLACE INTO listings VALUES "
                         "(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", rows)
    return len(rows)


if __name__ == "__main__":
    date = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_SNAPSHOT
    print(f"loaded {load_listings(date)} listings from snapshot {date} into {DB_PATH.name}")
