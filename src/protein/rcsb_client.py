from __future__ import annotations

from collections.abc import Iterator

import requests

RCSB_SEARCH_URL = "https://search.rcsb.org/rcsbsearch/v2/query"


def iter_rcsb_entry_identifiers(
    query: dict,
    page_size: int,
    session: requests.Session | None = None,
    max_ids: int | None = None,
) -> Iterator[str]:
    if page_size <= 0:
        raise ValueError("page_size must be positive")
    if max_ids is not None and max_ids <= 0:
        raise ValueError("max_ids must be positive when set")

    own_session = session is None
    sess = session or requests.Session()

    start = 0
    yielded = 0
    try:
        while True:
            payload = {
                "query": query,
                "return_type": "entry",
                "request_options": {"paginate": {"start": start, "rows": page_size}},
            }
            response = sess.post(RCSB_SEARCH_URL, json=payload, timeout=120)
            response.raise_for_status()
            body = response.json()

            result_set = body.get("result_set")
            if not result_set:
                return

            for item in result_set:
                identifier = item.get("identifier")
                if not identifier:
                    raise ValueError("RCSB result missing identifier")
                yield str(identifier)
                yielded += 1
                if max_ids is not None and yielded >= max_ids:
                    return

            if len(result_set) < page_size:
                return

            start += len(result_set)
    finally:
        if own_session:
            sess.close()
