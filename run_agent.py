"""
run_agent.py - run the agent on one request and save the result and trace.

    uv run python run_agent.py "new software or ML internships posted since Monday that fit me"
    uv run python run_agent.py "quant internships in NYC" --profile profiles/sample_student.json

Writes runs/<timestamp>.json (result plus full trace) and runs/<timestamp>.md
(a readable report), and prints the ranked list with run totals.
"""
import argparse
import json
import time
from datetime import datetime
from pathlib import Path

from graph import AS_OF, MAX_STEPS, MODEL, run

RUNS = Path(__file__).parent / "runs"
DEFAULT_PROFILE = Path(__file__).parent / "profiles" / "sample_student.json"


def totals(trace: dict, wall_s: float) -> dict:
    """totals - sum a run's model usage, tool calls, and time.

    Parameters:
        trace (dict): The trace returned by ``graph.run``.
        wall_s (float): Wall-clock seconds for the whole run.

    Returns:
        dict: Counts of model calls (and how many failed), tool calls, input
        and output tokens (None if any successful call did not report them),
        seconds spent in model calls, in tool calls, and waiting on throttles
        or backoff, and wall time.

    Description:
        A failed model call has only an ``error``, so it is counted in
        ``failed_llm_calls`` and left out of the token and time sums.
    """
    llm = [c for c in trace["llm_calls"] if "error" not in c]
    tokens_known = all(c["input_tokens"] is not None for c in llm)
    return {
        "llm_calls": len(llm),
        "failed_llm_calls": len(trace["llm_calls"]) - len(llm),
        "tool_calls": len(trace["tool_calls"]),
        "cached_llm_calls": sum(c["cached"] for c in llm),
        "input_tokens": sum(c["input_tokens"] for c in llm) if tokens_known else None,
        "output_tokens": sum(c["output_tokens"] or 0 for c in llm) if tokens_known else None,
        "llm_s": round(sum(c["latency_s"] for c in llm), 2),
        "tool_s": round(sum(t["latency_s"] for t in trace["tool_calls"]), 2),
        "wait_s": round(sum(c["wait_s"] for c in llm), 2),
        "wall_s": round(wall_s, 2),
    }


def format_report(request: str, result: dict, trace: dict, summary: dict) -> str:
    """format_report - render one run as markdown.

    Parameters:
        request (str): The request text.
        result (dict): The result returned by ``graph.run``.
        trace (dict): The trace returned by ``graph.run``.
        summary (dict): The run totals from ``totals``.

    Returns:
        str: A markdown report with the ranked roles, notes, failed sources,
        totals, and the tool sequence.
    """
    lines = [f"# Run: {request}", "",
             f"Model: `{MODEL}` | As of: {AS_OF} | Step limit: {MAX_STEPS}", "",
             "## Ranked roles", ""]
    if not result["roles"]:
        lines.append("No roles returned.")
    for role in result["roles"]:
        lines += [f"{role['rank']}. **{role['company']} - {role['title']}** "
                  f"({', '.join(role['locations'])}; posted {role['date_posted']}; "
                  f"deadline: {role['deadline']})",
                  f"   - Apply: {role['url']}",
                  f"   - Fit: {role['fit_reason']}"]
        if role["key_requirements"]:
            lines.append(f"   - Requirements: {'; '.join(role['key_requirements'])}")
    lines += ["", "## Notes", "", result["notes"] or "None."]
    if result["failed_sources"]:
        lines += ["", "## Failed sources", ""] + [f"- {f}" for f in result["failed_sources"]]
    if result["dropped"]:
        lines += ["", "## Dropped (not returned by any tool)", "",
                  ", ".join(result["dropped"])]
    lines += ["", "## Totals", "", "```", json.dumps(summary, indent=2), "```",
              "", "## Tool sequence", ""]
    for record in trace["tool_calls"]:
        outcome = f"error: {record['error']}" if record.get("error") else "ok"
        lines.append(f"{record['step']}. `{record['tool']}({json.dumps(record['arguments'])})` "
                     f"- {outcome} ({record['latency_s']}s)"
                     f"{': ' + record['thought'] if record['thought'] else ''}")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the internship agent on one request.")
    parser.add_argument("request")
    parser.add_argument("--profile", type=Path, default=DEFAULT_PROFILE)
    args = parser.parse_args()

    profile = json.loads(args.profile.read_text())
    started = time.perf_counter()
    result, trace = run(args.request, profile)
    summary = totals(trace, time.perf_counter() - started)

    RUNS.mkdir(exist_ok=True)
    stem = RUNS / datetime.now().strftime("%Y%m%d-%H%M%S")
    record = {"request": args.request, "profile": str(args.profile), "model": MODEL,
              "as_of": AS_OF, "result": result, "trace": trace, "totals": summary}
    stem.with_suffix(".json").write_text(json.dumps(record, indent=2, ensure_ascii=False))
    report = format_report(args.request, result, trace, summary)
    stem.with_suffix(".md").write_text(report)
    print(report)
    print(f"Saved {stem.with_suffix('.md').name} and {stem.with_suffix('.json').name} in runs/")


if __name__ == "__main__":
    main()
