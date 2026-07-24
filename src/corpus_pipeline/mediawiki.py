"""Revision-pinned MediaWiki source retrieval."""

from datetime import datetime, timezone
import hashlib

import httpx

from corpus_pipeline.catalogue import CorpusSpec, WIKISOURCE_API_URL
from corpus_pipeline.models import CorpusValidationError, SourcePage


DEFAULT_USER_AGENT = (
    "AutoAnki/0.1 corpus builder "
    "(local desktop source-text retrieval)"
)
DEFAULT_BATCH_SIZE = 50
DEFAULT_TIMEOUT_SECONDS = 30.0


class MediaWikiFetchError(RuntimeError):
    """A MediaWiki request failed or returned an unusable response."""


def _retrieval_timestamp():
    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z")
    )


def _validate_fetch_options(user_agent, maxlag, batch_size):
    if not isinstance(user_agent, str) or not user_agent.strip():
        raise ValueError("A non-empty MediaWiki User-Agent is required.")
    if (
            isinstance(maxlag, bool)
            or not isinstance(maxlag, int)
            or maxlag < 0):
        raise ValueError("MediaWiki maxlag must be a non-negative integer.")
    if (
            isinstance(batch_size, bool)
            or not isinstance(batch_size, int)
            or not 1 <= batch_size <= 50):
        raise ValueError(
            "MediaWiki revision batches must contain between 1 and 50 titles.")


def _response_payload(response):
    try:
        response.raise_for_status()
    except Exception as exc:
        raise MediaWikiFetchError(
            f"MediaWiki returned an HTTP error: {exc}") from exc
    try:
        payload = response.json()
    except Exception as exc:
        raise MediaWikiFetchError(
            "MediaWiki returned a response that was not valid JSON.") from exc
    if not isinstance(payload, dict):
        raise MediaWikiFetchError(
            "MediaWiki returned an unexpected JSON response.")
    if "error" in payload:
        error = payload["error"]
        if isinstance(error, dict):
            code = error.get("code", "unknown")
            detail = error.get("info", "No detail supplied.")
        else:
            code = "unknown"
            detail = str(error)
        raise MediaWikiFetchError(
            f"MediaWiki API error {code}: {detail}")
    return payload


def _revision_content(revision, title):
    try:
        content = revision["slots"]["main"]["content"]
    except (KeyError, TypeError) as exc:
        raise MediaWikiFetchError(
            f"MediaWiki did not return raw main-slot content for {title!r}."
        ) from exc
    if not isinstance(content, str):
        raise MediaWikiFetchError(
            f"MediaWiki returned non-text content for {title!r}.")
    return content


def _request_batch(
        client,
        api_url,
        titles,
        *,
        user_agent,
        maxlag,
        timeout):
    response = client.get(
        api_url,
        params={
            "action": "query",
            "prop": "info|revisions",
            "inprop": "url",
            "rvprop": "ids|timestamp|sha1|content",
            "rvslots": "main",
            "titles": "|".join(titles),
            "format": "json",
            "formatversion": "2",
            "maxlag": str(maxlag),
        },
        headers={
            "User-Agent": user_agent,
            "Accept": "application/json",
        },
        timeout=timeout,
    )
    payload = _response_payload(response)
    try:
        pages = payload["query"]["pages"]
    except (KeyError, TypeError) as exc:
        raise MediaWikiFetchError(
            "MediaWiki response did not contain query pages.") from exc
    if not isinstance(pages, list):
        raise MediaWikiFetchError(
            "MediaWiki query pages were not returned as a list.")
    return pages


def fetch_pages(
        page_titles,
        *,
        page_keys=None,
        api_url=WIKISOURCE_API_URL,
        client=None,
        user_agent=DEFAULT_USER_AGENT,
        maxlag=5,
        batch_size=DEFAULT_BATCH_SIZE,
        timeout=DEFAULT_TIMEOUT_SECONDS,
        retrieved_at=None,
        progress_callback=None):
    """Fetch exact current revisions while retaining the requested order.

    ``client`` may be an ``httpx.Client`` or a compatible test double.  No
    retries are hidden here: a maxlag or transport failure is surfaced to the
    caller so a future GUI can ask before trying again.
    """
    titles = tuple(page_titles)
    if not titles:
        raise ValueError("At least one MediaWiki page title is required.")
    if len(set(titles)) != len(titles):
        raise ValueError("MediaWiki page titles must be unique.")
    if page_keys is None:
        keys = tuple(
            f"page-{order:03d}"
            for order in range(1, len(titles) + 1)
        )
    else:
        keys = tuple(page_keys)
        if len(keys) != len(titles):
            raise ValueError(
                "page_keys must contain one key for every page title.")
        if len(set(keys)) != len(keys):
            raise ValueError("MediaWiki page keys must be unique.")
    _validate_fetch_options(user_agent, maxlag, batch_size)

    retrieval_time = retrieved_at or _retrieval_timestamp()
    owned_client = client is None
    request_client = client or httpx.Client()
    returned_by_title = {}
    try:
        for start in range(0, len(titles), batch_size):
            batch = titles[start:start + batch_size]
            pages = _request_batch(
                request_client,
                api_url,
                batch,
                user_agent=user_agent,
                maxlag=maxlag,
                timeout=timeout,
            )
            for page in pages:
                if not isinstance(page, dict):
                    raise MediaWikiFetchError(
                        "MediaWiki returned an invalid page record.")
                title = page.get("title")
                if page.get("missing") is not None:
                    raise MediaWikiFetchError(
                        f"Required MediaWiki page is missing: {title!r}.")
                if title not in batch:
                    raise MediaWikiFetchError(
                        "MediaWiki returned an unexpected or redirected page: "
                        f"{title!r}.")
                if title in returned_by_title:
                    raise MediaWikiFetchError(
                        f"MediaWiki returned duplicate page data for {title!r}.")
                revisions = page.get("revisions")
                if not isinstance(revisions, list) or len(revisions) != 1:
                    raise MediaWikiFetchError(
                        "MediaWiki did not return exactly one current revision "
                        f"for {title!r}.")
                revision = revisions[0]
                content = _revision_content(revision, title)
                try:
                    revision_id = int(revision["revid"])
                    revision_timestamp = str(revision["timestamp"])
                    revision_sha1 = str(revision["sha1"])
                except (KeyError, TypeError, ValueError) as exc:
                    raise MediaWikiFetchError(
                        "MediaWiki revision metadata is incomplete for "
                        f"{title!r}.") from exc
                url = page.get("fullurl")
                if not isinstance(url, str) or not url:
                    raise MediaWikiFetchError(
                        f"MediaWiki did not provide a source URL for {title!r}.")
                returned_by_title[title] = (
                    url,
                    revision_id,
                    revision_timestamp,
                    revision_sha1,
                    content,
                )
            missing = set(batch).difference(returned_by_title)
            if missing:
                names = ", ".join(sorted(repr(title) for title in missing))
                raise MediaWikiFetchError(
                    f"MediaWiki omitted required page(s): {names}.")
            if progress_callback is not None:
                progress_callback(
                    min(start + len(batch), len(titles)),
                    len(titles))
    except httpx.HTTPError as exc:
        raise MediaWikiFetchError(
            f"Could not retrieve MediaWiki source text: {exc}") from exc
    finally:
        if owned_client:
            request_client.close()

    result = []
    for order, (key, title) in enumerate(
            zip(keys, titles, strict=True),
            start=1):
        (
            url,
            revision_id,
            revision_timestamp,
            revision_sha1,
            content,
        ) = returned_by_title[title]
        result.append(SourcePage(
            page_key=key,
            order=order,
            title=title,
            url=url,
            revision_id=revision_id,
            revision_timestamp=revision_timestamp,
            revision_sha1=revision_sha1,
            retrieved_at=retrieval_time,
            raw_wikitext=content,
            raw_sha256=hashlib.sha256(
                content.encode("utf-8")).hexdigest(),
        ))
    return tuple(result)


def fetch_corpus_pages(
        spec: CorpusSpec,
        **fetch_options):
    """Fetch the index and content pages defined by an exact corpus spec."""
    titles = spec.requested_titles
    return fetch_pages(
        titles,
        page_keys=tuple(spec.page_key(title) for title in titles),
        api_url=spec.api_url,
        **fetch_options,
    )


def validate_page_provenance(page):
    """Check hashes before a stored source page is cleaned or rebuilt."""
    actual_hash = hashlib.sha256(
        page.raw_wikitext.encode("utf-8")).hexdigest()
    if actual_hash != page.raw_sha256:
        raise CorpusValidationError(
            f"Raw source hash does not match for {page.title!r}.")
