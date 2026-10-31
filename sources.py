"""
sources.py - read the external data sources the agent uses.

    load_simplify_snapshot(date)  # SimplifyJobs listings.json, saved on that date
    greenhouse_job(url)           # one full posting from the Greenhouse job board API

Every HTTP response is saved in data/http_cache/ keyed by URL, so reruns read
the same pages instead of calling the APIs again. The cache is committed with
the snapshot so evaluation runs see the same data.
"""
from __future__ import annotations
import gzip
import hashlib
import html
import json
import re
import urllib.error
import urllib.request
from pathlib import Path

DATA = Path(__file__).parent / "data"
SNAPSHOTS = DATA / "snapshots"
HTTP_CACHE = DATA / "http_cache"
DEFAULT_SNAPSHOT = "2026-10-31"
SIMPLIFY_LIVE_URL = ("https://raw.githubusercontent.com/SimplifyJobs/"
                     "Summer2027-Internships/dev/.github/scripts/listings.json")
HTTP_TIMEOUT_S = 20
# Greenhouse posting URLs look like https://job-boards.greenhouse.io/<board>/jobs/<id>.
_GREENHOUSE_URL = re.compile(r"https?://(?:job-)?boards(\.eu)?\.greenhouse\.io/([^/?#]+)/jobs/(\d+)")


class SourceError(Exception):
    """SourceError - an external source could not be reached or read.

    Description:
        Raised with a message that names the source, so the agent can report
        which source failed and still return partial results.
    """


def http_get_json(url: str, *, use_cache: bool = True) -> object:
    """http_get_json - fetch a URL and parse its JSON body.

    Parameters:
        url (str): The URL to fetch.
        use_cache (bool): Whether to reuse a saved response for this URL.

    Returns:
        object: The parsed JSON value.

    Raises:
        SourceError: If the request fails, times out, or is not JSON.

    Description:
        Responses are saved under HTTP_CACHE as ``<sha256 of url>.json`` with
        the URL alongside the body, so a cached file can be traced back to
        where it came from.
    """
    path = HTTP_CACHE / f"{hashlib.sha256(url.encode()).hexdigest()}.json"
    if use_cache and path.exists():
        return json.loads(path.read_text())["body"]

    request = urllib.request.Request(url, headers={"User-Agent": "cis1990-internship-agent"})
    try:
        with urllib.request.urlopen(request, timeout=HTTP_TIMEOUT_S) as response:
            body = json.loads(response.read().decode())
    except urllib.error.HTTPError as error:
        raise SourceError(f"{url} returned HTTP {error.code}") from error
    except (urllib.error.URLError, TimeoutError) as error:
        raise SourceError(f"{url} could not be reached: {error}") from error
    except json.JSONDecodeError as error:
        raise SourceError(f"{url} did not return JSON") from error

    HTTP_CACHE.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"url": url, "body": body}))
    return body


def load_simplify_snapshot(date: str = DEFAULT_SNAPSHOT) -> list[dict]:
    """load_simplify_snapshot - read a saved SimplifyJobs listings file.

    Parameters:
        date (str): The snapshot folder name, e.g. "2026-10-31".

    Returns:
        list[dict]: Every listing record, active or not, as SimplifyJobs
        wrote it.

    Raises:
        SourceError: If no snapshot exists for that date.
    """
    path = SNAPSHOTS / date / "simplify_listings.json.gz"
    if not path.exists():
        raise SourceError(f"SimplifyJobs snapshot {path} not found")
    with gzip.open(path, "rt") as f:
        return json.load(f)


def save_simplify_snapshot(date: str) -> Path:
    """save_simplify_snapshot - download today's SimplifyJobs list and save it.

    Parameters:
        date (str): The folder name to save under, normally today's date.

    Returns:
        Path: The saved ``simplify_listings.json.gz`` file.

    Description:
        Bypasses the HTTP cache so the download is current. Evaluation uses
        a saved snapshot, never the live list, so results can be reproduced.
    """
    listings = http_get_json(SIMPLIFY_LIVE_URL, use_cache=False)
    path = SNAPSHOTS / date / "simplify_listings.json.gz"
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt") as f:
        json.dump(listings, f)
    return path


def ats_of(url: str) -> str:
    """ats_of - name the applicant tracking system behind a posting URL.

    Parameters:
        url (str): An apply link.

    Returns:
        str: "greenhouse", "lever", "ashby", "workday", or "other".
    """
    for name, marker in (("greenhouse", "greenhouse.io"), ("lever", "lever.co"),
                         ("ashby", "ashbyhq.com"), ("workday", "myworkdayjobs.com")):
        if marker in url:
            return name
    return "other"


def html_to_text(raw: str) -> str:
    """html_to_text - turn a posting's HTML into plain text.

    Parameters:
        raw (str): HTML, possibly escaped twice as Greenhouse returns it.

    Returns:
        str: The text with tags removed, block tags turned into line breaks,
        and runs of spaces collapsed.
    """
    text = html.unescape(html.unescape(raw or ""))
    text = re.sub(r"<(br|/p|/li|/h\d|/div)[^>]*>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"<li[^>]*>", "- ", text, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"[ \t\xa0]+", " ", text)
    return re.sub(r"\n\s*\n+", "\n", text).strip()


def greenhouse_job(url: str) -> dict:
    """greenhouse_job - fetch the full posting behind a Greenhouse apply link.

    Parameters:
        url (str): A Greenhouse posting URL from the listings.

    Returns:
        dict: ``title``, ``company``, ``location``, ``deadline`` (the
        posting's ``application_deadline`` or None), ``updated_at``,
        ``description`` (plain text), and ``api_url``.

    Raises:
        SourceError: If the URL is not a Greenhouse posting or the API call
        fails.
    """
    match = _GREENHOUSE_URL.match(url)
    if not match:
        raise SourceError(f"not a Greenhouse posting URL: {url}")
    eu, board, job_id = match.groups()
    api_url = f"https://boards-api{eu or ''}.greenhouse.io/v1/boards/{board}/jobs/{job_id}?content=true"
    job = http_get_json(api_url)
    if not isinstance(job, dict) or "title" not in job:
        raise SourceError(f"Greenhouse returned no posting for {board}/{job_id}")
    return {
        "title": job.get("title"),
        "company": job.get("company_name") or board,
        "location": (job.get("location") or {}).get("name"),
        "deadline": job.get("application_deadline"),
        "updated_at": job.get("updated_at"),
        "description": html_to_text(job.get("content", "")),
        "api_url": api_url,
    }
