"""Gemini client, adapted from the course-provided llm.py used in HW1 and HW2.

``complete`` keeps the HW interface and returns only text. The project adds
``complete_with_usage``, which also returns token counts and timing for the
traces and the cost/latency report. Changes from the HW version:

1. Token counts, API latency, and time spent waiting are returned per call.
2. Every request has a timeout (``TIMEOUT_S``).
3. Server errors (5xx) and timeouts are retried with backoff, not just 429.
4. The cache can be turned off for every call with ``LLM_CACHE=0``, which the
   eval runner uses so it measures real calls, not saved outputs.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
import warnings
from pathlib import Path
from dotenv import load_dotenv

# The Google SDK can emit warnings that are not useful for this project's
# output, so suppress them while leaving actual exceptions visible.
warnings.filterwarnings("ignore")
from google import genai
from google.genai import types

load_dotenv(Path(__file__).parent / ".env")

CACHE_DIR = Path(__file__).parent / ".cache"
# An alias that Google may point at a newer model. Final eval runs should set
# GEMINI_MODEL to a fixed model name and record it with the run date.
DEV_MODEL = "gemini-flash-lite-latest"
# The free Gemini API limits how quickly requests can be sent. An agent may
# make several requests for one task, so waiting 4.2 seconds between real
# requests reduces HTTP 429 ("Too Many Requests") failures.
_MIN_GAP = 4.2
TIMEOUT_S = 60
MAX_ATTEMPTS = 6
# Substrings of error messages that are worth retrying.
_RETRYABLE = ("429", "500", "502", "503", "504", "timed out", "timeout", "deadline")
_last_call = [0.0]
_client = None


def _get_client():
    """_get_client - create and return the shared Gemini client.

    Returns:
        genai.Client: The Gemini SDK client configured with the API key and
        a per-request timeout.

    Description:
        The client is created only when a model request actually needs it.
        This allows the module to be imported before the environment is set
        up, while still reporting a clear error when a request is attempted
        without ``GEMINI_API_KEY``. The same client is reused for later calls.
    """
    global _client
    if _client is None:
        key = os.environ.get("GEMINI_API_KEY")
        if not key:
            raise RuntimeError("Set GEMINI_API_KEY (copy .env.example to .env).")
        _client = genai.Client(api_key=key,
                               http_options=types.HttpOptions(timeout=TIMEOUT_S * 1000))
    return _client


def _cache_key(prompt: str, system: str, temperature: float, model: str) -> str:
    """_cache_key - build an identifier for one model request.

    Parameters:
        prompt (str): The user prompt sent to the model.
        system (str): The model's role or response instructions.
        temperature (float): The model's randomness setting.
        model (str): The Gemini model name.

    Returns:
        str: A SHA-256 hexadecimal string suitable for a cache filename.

    Description:
        Every input that can change the response is included. This prevents a
        response for one prompt, model, or temperature from being reused for a
        different request.
    """
    return hashlib.sha256(f"{model}|{temperature}|{system}|{prompt}".encode()).hexdigest()


def clear_cache() -> None:
    """clear_cache - delete all saved model responses.

    Returns:
        None: The cache directory is removed if it exists.

    Description:
        Use this after changing a prompt or when testing behavior that should
        not reuse an older response. The next call to ``complete`` contacts
        Gemini instead of reading the previous response from disk.
    """
    import shutil
    shutil.rmtree(CACHE_DIR, ignore_errors=True)


def complete_with_usage(prompt: str, *, system: str = "", temperature: float = 0.7,
                        model: str = DEV_MODEL, use_cache: bool = True) -> tuple[str, dict]:
    """complete_with_usage - send a prompt to Gemini and report what it cost.

    Parameters:
        prompt (str): The request sent to the model.
        system (str): Optional role or response rules for the model.
        temperature (float): Controls how varied the response may be.
        model (str): The Gemini model to use.
        use_cache (bool): Whether to reuse an identical saved response.
            Ignored (treated as False) when the ``LLM_CACHE`` environment
            variable is "0".

    Returns:
        tuple[str, dict]: The model's non-empty, whitespace-trimmed response,
        and a usage record with ``model``, ``input_tokens``,
        ``output_tokens``, ``latency_s`` (time inside successful and failed
        API calls), ``wait_s`` (throttle and backoff sleeps), ``attempts``,
        and ``cached``.

    Description:
        With caching enabled, an identical request is read from disk with its
        saved token counts, and ``cached`` is True so cost reports can tell
        it apart from a real call. New requests are spaced apart, empty
        responses are retried, and rate-limit, server, and timeout errors
        receive progressively longer waits before another attempt.
    """
    use_cache = use_cache and os.environ.get("LLM_CACHE", "1") != "0"
    key = _cache_key(prompt, system, temperature, model)
    path = CACHE_DIR / f"{key}.json"
    usage = {"model": model, "input_tokens": None, "output_tokens": None,
             "latency_s": 0.0, "wait_s": 0.0, "attempts": 0, "cached": False}
    if use_cache and path.exists():
        # A cached response avoids both an API request and the throttle delay.
        saved = json.loads(path.read_text())
        if saved.get("v"):
            usage.update(input_tokens=saved.get("in"), output_tokens=saved.get("out"),
                         cached=True)
            return saved["v"], usage

    config = types.GenerateContentConfig(
        temperature=temperature,
        system_instruction=system or None,
    )
    for attempt in range(MAX_ATTEMPTS):
        # This timestamp is shared by calls in the current process. It prevents
        # consecutive agent steps from exceeding the API's request rate.
        wait = _MIN_GAP - (time.time() - _last_call[0])
        if wait > 0:
            time.sleep(wait)
            usage["wait_s"] += wait
        _last_call[0] = time.time()
        usage["attempts"] += 1
        started = time.perf_counter()
        try:
            response = _get_client().models.generate_content(
                model=model, contents=prompt, config=config)
            usage["latency_s"] += time.perf_counter() - started
            output = (response.text or "").strip()
            meta = response.usage_metadata
            usage["input_tokens"] = getattr(meta, "prompt_token_count", None)
            usage["output_tokens"] = getattr(meta, "candidates_token_count", None)
            if output:
                # Save successful responses so rerunning the same prompt is
                # fast, deterministic, and does not spend additional quota.
                CACHE_DIR.mkdir(exist_ok=True)
                path.write_text(json.dumps({"v": output, "in": usage["input_tokens"],
                                            "out": usage["output_tokens"]}))
                return output, usage
        except Exception as error:
            usage["latency_s"] += time.perf_counter() - started
            message = str(error)
            retryable = any(marker in message.lower() for marker in _RETRYABLE)
            if retryable and attempt < MAX_ATTEMPTS - 1:
                # Exponential backoff gives the service time to recover.
                backoff = min(40, 5 * (2 ** attempt))
                time.sleep(backoff)
                usage["wait_s"] += backoff
                continue
            if "429" in message:
                raise RuntimeError("Rate limit: retried with backoff and still throttled.")
            raise RuntimeError(f"Gemini call failed: {message[:160]}")
    raise RuntimeError("Model returned empty text repeatedly.")


def complete(prompt: str, *, system: str = "", temperature: float = 0.7,
             model: str = DEV_MODEL, use_cache: bool = True) -> str:
    """complete - send a prompt to Gemini and return the generated text.

    Parameters:
        prompt (str): The request sent to the model.
        system (str): Optional role or response rules for the model.
        temperature (float): Controls how varied the response may be.
        model (str): The Gemini model to use.
        use_cache (bool): Whether to reuse an identical saved response.

    Returns:
        str: The model's non-empty, whitespace-trimmed response.

    Description:
        The HW interface, kept for code that does not need usage numbers.
        See ``complete_with_usage`` for caching, throttling, and retries.
    """
    return complete_with_usage(prompt, system=system, temperature=temperature,
                               model=model, use_cache=use_cache)[0]
