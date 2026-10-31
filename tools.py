"""
tools.py - the tools the agent can call, and the schemas shown to the model.

    search_listings({"keywords": ..., "category": ..., ...}) -> list[dict]
    get_job_detail({"listing_id": ...})                     -> dict

Each tool takes the model's ``arguments`` dict and returns plain JSON-able
data. Bad arguments raise ValueError and source failures raise SourceError;
graph.py turns both into error records instead of crashing.
"""
from __future__ import annotations
import json
from datetime import date, datetime, timezone

from db import connect
from sources import SourceError, greenhouse_job

DEFAULT_TERM = "Summer 2027"
SEARCH_LIMIT = 20
MAX_SEARCH_LIMIT = 50
# Long postings are cut so a few detail calls still fit in one prompt.
DESCRIPTION_CHARS = 4000
# Full city name -> the short code SimplifyJobs uses for it.
LOCATION_ALIASES = {"new york": "NYC", "san francisco": "SF", "los angeles": "LA"}


def _location_patterns(location: str) -> list[str]:
    """_location_patterns - SQL LIKE patterns that match a location filter.

    Parameters:
        location (str): The location the model asked for, e.g. "New York".

    Returns:
        list[str]: Lowercase patterns; a listing matches if any one does.

    Description:
        SimplifyJobs writes some cities as short codes ("NYC" in 427 active
        listings, "New York, NY" in 5). A code is matched only as a whole
        list element ('"la"'), so "LA" does not match "Atlanta".
    """
    wanted = location.strip().lower()
    codes = {code.lower() for code in LOCATION_ALIASES.values()}
    patterns = [] if wanted in codes else [f"%{wanted}%"]
    for name, code in LOCATION_ALIASES.items():
        if name in wanted or wanted == code.lower():
            patterns += [f"%{name}%", f'%"{code.lower()}"%']
    return list(dict.fromkeys(patterns))


def _posted_date(timestamp: int | None) -> str | None:
    """_posted_date - turn a Unix timestamp into a YYYY-MM-DD string."""
    if not timestamp:
        return None
    return datetime.fromtimestamp(timestamp, tz=timezone.utc).date().isoformat()


def search_listings(arguments: dict) -> list[dict]:
    """search_listings - filter active listings on their structured fields.

    Parameters:
        arguments (dict): Any of ``keywords`` (words that must all appear in
            the title or company), ``category`` (e.g. "Software", matched as
            a substring), ``location`` (substring of any location, e.g.
            "New York" or "Remote"), ``posted_after`` (YYYY-MM-DD,
            inclusive), ``term`` (default "Summer 2027"), and ``limit``
            (default 20, at most 50).

    Returns:
        list[dict]: Matching listings, newest first, each with
        ``listing_id``, ``company``, ``title``, ``category``, ``locations``,
        ``date_posted``, ``url``, and ``ats``.

    Raises:
        ValueError: If ``posted_after`` is not a date or ``limit`` is not a
        positive integer.

    Description:
        Only active listings are searched. Matching is case-insensitive.
        The filters are applied in SQL, so the model never sees the full
        listings table.
    """
    clauses, params = ["active = 1"], []

    term = str(arguments.get("term") or DEFAULT_TERM)
    clauses.append("terms LIKE ?")
    params.append(f'%"{term}"%')

    for word in str(arguments.get("keywords") or "").split():
        clauses.append("(LOWER(title) LIKE ? OR LOWER(company) LIKE ?)")
        params += [f"%{word.lower()}%"] * 2

    if arguments.get("category"):
        clauses.append("LOWER(category) LIKE ?")
        params.append(f"%{str(arguments['category']).lower()}%")

    if arguments.get("location"):
        patterns = _location_patterns(str(arguments["location"]))
        clauses.append("(" + " OR ".join(["LOWER(locations) LIKE ?"] * len(patterns)) + ")")
        params += patterns

    if arguments.get("posted_after"):
        try:
            since = date.fromisoformat(str(arguments["posted_after"]))
        except ValueError as error:
            raise ValueError("posted_after must be a date like 2026-10-26") from error
        clauses.append("date_posted >= ?")
        params.append(int(datetime(since.year, since.month, since.day,
                                   tzinfo=timezone.utc).timestamp()))

    try:
        limit = int(arguments.get("limit") or SEARCH_LIMIT)
    except (TypeError, ValueError) as error:
        raise ValueError("limit must be a positive integer") from error
    if limit <= 0:
        raise ValueError("limit must be a positive integer")
    limit = min(limit, MAX_SEARCH_LIMIT)

    sql = (f"SELECT * FROM listings WHERE {' AND '.join(clauses)} "
           "ORDER BY date_posted DESC, company LIMIT ?")
    with connect() as conn:
        rows = conn.execute(sql, params + [limit]).fetchall()
    return [{"listing_id": row["id"], "company": row["company"], "title": row["title"],
             "category": row["category"], "locations": json.loads(row["locations"]),
             "date_posted": _posted_date(row["date_posted"]), "url": row["url"],
             "ats": row["ats"]}
            for row in rows]


def get_job_detail(arguments: dict) -> dict:
    """get_job_detail - fetch the full posting for one listing.

    Parameters:
        arguments (dict): ``listing_id`` from a ``search_listings`` result.

    Returns:
        dict: ``listing_id``, ``company``, ``title``, ``location``,
        ``deadline`` (as the posting states it, or None), ``url``, and
        ``description`` (plain text, cut to ``DESCRIPTION_CHARS``).

    Raises:
        ValueError: If ``listing_id`` is missing or unknown.
        SourceError: If the posting's job board is not supported yet or the
        board's API fails.

    Description:
        Only Greenhouse boards are supported so far. Lever and Ashby are
        planned; Workday has no public API, so those listings stay
        search-only.
    """
    listing_id = str(arguments.get("listing_id") or "").strip()
    if not listing_id:
        raise ValueError("listing_id is required")
    with connect() as conn:
        row = conn.execute("SELECT * FROM listings WHERE id = ?", (listing_id,)).fetchone()
    if row is None:
        raise ValueError(f"unknown listing_id {listing_id}")
    if row["ats"] != "greenhouse":
        raise SourceError(f"full posting not available: {row['ats']} job boards are not supported yet")

    job = greenhouse_job(row["url"])
    description = job["description"]
    if len(description) > DESCRIPTION_CHARS:
        description = description[:DESCRIPTION_CHARS] + " [...]"
    return {"listing_id": listing_id, "company": row["company"], "title": job["title"],
            "location": job["location"], "deadline": job["deadline"], "url": row["url"],
            "description": description}


TOOL_SCHEMAS = [
    {
        "name": "search_listings",
        "description": ("Search active internship listings from the SimplifyJobs "
                        "list by structured fields. Returns short records with a "
                        "listing_id, newest first. Use it first to build a pool of "
                        "candidate roles. Combine filters to keep the pool small."),
        "parameters": {"type": "object",
                       "properties": {
                           "keywords": {"type": "string",
                                        "description": "Words that must all appear in the title or company, e.g. 'machine learning'."},
                           "category": {"type": "string",
                                        "description": "One of: Software, AI/ML/Data, Hardware, Product, Quant."},
                           "location": {"type": "string",
                                        "description": "Part of a location, e.g. 'New York' or 'Remote'."},
                           "posted_after": {"type": "string",
                                            "description": "Only listings posted on or after this date, YYYY-MM-DD."},
                           "term": {"type": "string", "description": "Default 'Summer 2027'."},
                           "limit": {"type": "integer", "description": "Default 20, at most 50."}},
                       "required": []},
    },
    {
        "name": "get_job_detail",
        "description": ("Fetch the full job description for one listing_id, including "
                        "the stated deadline if any. Only Greenhouse postings "
                        "(ats = 'greenhouse') are supported; other job boards return "
                        "an error."),
        "parameters": {"type": "object",
                       "properties": {"listing_id": {"type": "string"}},
                       "required": ["listing_id"]},
    },
]
# The function run for each tool name.
TOOLS = {"search_listings": search_listings, "get_job_detail": get_job_detail}
