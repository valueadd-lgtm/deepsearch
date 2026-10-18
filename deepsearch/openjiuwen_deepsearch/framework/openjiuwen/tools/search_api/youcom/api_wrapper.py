# -*- coding: UTF-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
"""You.com web search provider.

Adds the You.com Agents Search API (``POST https://api.you.com/v1/agents/search``)
as an optional web search engine, selectable with ``--web_search_engine_name youcom``.
The API key is read from the standard ``search_api_key`` config field (populated from
``YDC_API_KEY``). The engine is opt-in: the default engine and every other code path
are unchanged when ``youcom`` is not selected.
"""

import json
import logging
from functools import cached_property
from typing import Any, Generic, TypeVar, List, Dict, Union, Optional

import httpx
import requests
from pydantic import BaseModel, ConfigDict, SecretStr

from openjiuwen.core.common.security.ssl_utils import SslUtils
from openjiuwen_deepsearch.common.common_constants import (
    MAX_URL_LENGTH,
    MAX_SEARCH_CONTENT_LENGTH,
)
from openjiuwen_deepsearch.utils.common_utils.url_utils import validate_search_service_url

T = TypeVar("T")

logger = logging.getLogger(__name__)

DEFAULT_YOUCOM_SEARCH_URL = "https://api.you.com"
# Attribution token required by the You.com integration program; identifies the
# source project on every call to api.you.com. Append-only (space-separated).
YOUCOM_USER_AGENT = "youdotcom-integration/openjiuwen-ai-deepsearch"


class YoucomSearchAPIWrapper(BaseModel, Generic[T]):
    """Wrapper class for the You.com Agents Search API.

    Uses the keyed Agents endpoint (``POST /v1/agents/search`` with ``X-API-Key``).
    The response carries a flat ``results`` array where each item exposes ``title``,
    ``url`` and a ``snippet``/``description`` text field, which we normalize into the
    project's common ``{title, url, content, score}``` shape.
    """

    search_api_key: bytearray = None
    search_url: SecretStr = None
    max_web_search_results: int = 5
    extension: Optional[dict] = None

    # MINIMAL: country/language filters are not part of the You.com Agents API
    # request body today; upgrade path is to add ``country``/``mkt`` when You.com
    # exposes them. ``extension`` is still honored for future options below.
    country: Optional[str] = None

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    def model_post_init(self, __context: Any) -> None:
        """Apply engine-specific options from ``extension``."""
        ext = self.extension
        if not ext:
            return
        if "country" in ext:
            self.country = ext["country"]

        # 预解析 search_url 以避免在 async 路径中首次触发同步 DNS 解析
        _ = self._resolved_search_url

    def raw_search_results(self, query: str) -> Dict:
        """Run query through the You.com Agents Search API and return raw result."""
        api_url = f"{self._resolved_search_url}/v1/agents/search"
        headers = self._build_headers()
        body = self._build_search_body(query=query)
        verify = self._get_ssl_verify_config()

        response = requests.post(api_url, json=body, headers=headers, verify=verify, timeout=30)
        response.raise_for_status()
        return response.json()

    async def raw_search_results_async(self, query: str) -> Dict:
        """Run query through the You.com Agents Search API asynchronously."""
        request_url = f"{self._resolved_search_url}/v1/agents/search"
        headers = self._build_headers()
        body = self._build_search_body(query=query)
        ssl_verify_flag = self._get_ssl_verify_config()

        async with httpx.AsyncClient(verify=ssl_verify_flag, timeout=30) as http_client:
            api_response = await http_client.post(request_url, json=body, headers=headers)
            if api_response.status_code not in (200, 201):
                raise Exception(
                    f"Error {api_response.status_code}: {api_response.reason_phrase}"
                )
            return json.loads(api_response.text)

    def results(self, query: str) -> List[Dict]:
        """Run query through the You.com Agents Search API and return cleaned result."""
        return self.clean_results(self._extract_search_results(self.raw_search_results(query=query)))

    async def aresults(self, query: str) -> List[Dict]:
        """Run query through the You.com Agents Search API asynchronously and return cleaned result."""
        raw_data = await self.raw_search_results_async(query=query)
        return self.clean_results(self._extract_search_results(raw_data))

    @staticmethod
    def _extract_search_results(raw_data: Any) -> List[Dict]:
        """校验并提取 You.com Agents 响应中的结果列表。

        The keyed Agents API returns ``{"results": [ ... ]}`` as a flat array. We
        also tolerate the nested ``{"results": {"web": [...]}}`` shape returned by
        the keyless/Search endpoints for robustness, so callers can swap endpoints
        without changing the parser.

        Raises:
            ValueError: 响应不是字典、或 results 既不是列表也不是字典时抛出。
        """
        if not isinstance(raw_data, dict):
            raise ValueError("You.com response must be a dict")
        results = raw_data.get("results", [])
        if isinstance(results, list):
            if any(not isinstance(item, dict) for item in results):
                raise ValueError("You.com response results must be a list of dicts")
            return results
        if isinstance(results, dict):
            # Nested shape (keyless/Search API): flatten web + news.
            flattened: List[Dict] = []
            for key in ("web", "news"):
                items = results.get(key)
                if isinstance(items, list):
                    flattened.extend(item for item in items if isinstance(item, dict))
            return flattened
        raise ValueError("You.com response results must be a list or a dict")

    def clean_results(self, results: List[Dict]) -> List[Dict]:
        """Clean results from the You.com Agents Search API into the common shape."""
        cleaned_results: List[Dict] = []
        for result in results:
            title = result.get("title", "") or ""
            url = result.get("url", "") or ""
            # Agents API (keyed) exposes ``snippet`` (singular); fall back to
            # ``description`` then ``content`` for the nested/Search-API shape.
            content = (
                result.get("snippet")
                or result.get("description")
                or result.get("content")
                or ""
            )
            cleaned_result = {
                "title": title[:MAX_SEARCH_CONTENT_LENGTH],
                "url": url[:MAX_URL_LENGTH],
                "content": content[:MAX_SEARCH_CONTENT_LENGTH],
                "score": result.get("score", 0.0),
            }
            cleaned_results.append(cleaned_result)
        return cleaned_results

    def _build_headers(self) -> Dict[str, str]:
        """Build request headers, including You.com attribution."""
        api_key = self.search_api_key.decode("utf-8") if self.search_api_key else ""
        return {
            "X-API-Key": api_key,
            "Content-Type": "application/json",
            "User-Agent": YOUCOM_USER_AGENT,
        }

    def _build_search_body(self, query: str) -> Dict[str, Any]:
        """Build the JSON body for the You.com Agents Search API request."""
        body: Dict[str, Any] = {
            "query": query,
            "max_results": self.max_web_search_results,
        }
        if self.country:
            body["country"] = self.country
        return body

    def _get_ssl_verify_config(self) -> Union[str, bool]:
        """Get SSL verification configuration."""
        ssl_verify, ssl_cert = SslUtils.get_ssl_config(
            "TOOL_SSL_VERIFY", "TOOL_SSL_CERT", ["false"]
        )
        return ssl_cert if ssl_verify else False

    @cached_property
    def _resolved_search_url(self) -> str:
        """Return configured URL or You.com's public default URL."""
        if self.search_url is None:
            return DEFAULT_YOUCOM_SEARCH_URL
        if hasattr(self.search_url, "get_secret_value"):
            configured = self.search_url.get_secret_value()
        else:
            configured = str(self.search_url)
        configured = (configured or "").strip().rstrip("/")
        if not configured:
            return DEFAULT_YOUCOM_SEARCH_URL
        validate_search_service_url(configured)
        return configured
