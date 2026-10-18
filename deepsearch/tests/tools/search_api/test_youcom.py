import os

os.environ["LLM_SSL_VERIFY"] = "false"
os.environ["TOOL_SSL_VERIFY"] = "false"

import json
from unittest.mock import Mock, patch, AsyncMock

import pytest
from pydantic import SecretStr

from openjiuwen_deepsearch.framework.openjiuwen.tools.search_api.youcom.api_wrapper import (
    YoucomSearchAPIWrapper,
    DEFAULT_YOUCOM_SEARCH_URL,
    YOUCOM_USER_AGENT,
)


class TestYoucomSearchAPIWrapper:
    """YoucomSearchAPIWrapper 单元测试 — mocked HTTP, no network."""

    @pytest.fixture
    def wrapper(self):
        return YoucomSearchAPIWrapper[str](
            search_api_key=bytearray(b"fake_ydc_key"),
            search_url=SecretStr("https://api.example.com"),
            max_web_search_results=4,
        )

    def test_init_defaults(self):
        w = YoucomSearchAPIWrapper(
            search_api_key=bytearray(b"k"),
            search_url=SecretStr(""),
            max_web_search_results=5,
        )
        assert w.max_web_search_results == 5
        assert w.country is None

    def test_extension_applies_country(self):
        w = YoucomSearchAPIWrapper(
            search_api_key=bytearray(b"k"),
            search_url=SecretStr(""),
            extension={"country": "us"},
        )
        assert w.country == "us"

    @patch("openjiuwen_deepsearch.framework.openjiuwen.tools.search_api.youcom.api_wrapper.requests.post")
    @patch("openjiuwen_deepsearch.framework.openjiuwen.tools.search_api.youcom.api_wrapper.SslUtils.get_ssl_config")
    def test_raw_search_results_flat_array(self, mock_ssl, mock_post):
        """Keyed Agents API returns a flat results array."""
        mock_ssl.return_value = (False, None)
        mock_response = Mock()
        mock_response.json.return_value = {
            "results": [
                {"title": "You.com Docs", "url": "https://you.com/docs", "snippet": "search api"},
                {"title": "News hit", "url": "https://example.com/n", "description": "desc fallback"},
            ]
        }
        mock_response.raise_for_status = Mock()
        mock_post.return_value = mock_response

        w = YoucomSearchAPIWrapper(
            search_api_key=bytearray(b"fake_ydc_key"),
            search_url=SecretStr("https://api.example.com"),
            max_web_search_results=2,
        )
        raw = w.raw_search_results(query="hello")

        assert raw == mock_response.json.return_value
        mock_post.assert_called_once()
        call_args = mock_post.call_args
        assert call_args[0][0] == "https://api.example.com/v1/agents/search"
        headers = call_args[1]["headers"]
        assert headers["X-API-Key"] == "fake_ydc_key"
        assert headers["User-Agent"] == YOUCOM_USER_AGENT
        assert call_args[1]["json"] == {"query": "hello", "max_results": 2}

    @patch("openjiuwen_deepsearch.framework.openjiuwen.tools.search_api.youcom.api_wrapper.requests.post")
    @patch("openjiuwen_deepsearch.framework.openjiuwen.tools.search_api.youcom.api_wrapper.SslUtils.get_ssl_config")
    def test_results_cleans_and_falls_back_to_description(self, mock_ssl, mock_post):
        mock_ssl.return_value = (False, None)
        mock_response = Mock()
        mock_response.json.return_value = {
            "results": [
                {"title": "With snippet", "url": "https://a.example", "snippet": "s"},
                {"title": "Desc only", "url": "https://b.example", "description": "d"},
            ]
        }
        mock_response.raise_for_status = Mock()
        mock_post.return_value = mock_response

        w = YoucomSearchAPIWrapper(
            search_api_key=bytearray(b"fake_ydc_key"),
            search_url=SecretStr(""),
            max_web_search_results=2,
        )
        cleaned = w.results("q")

        assert cleaned == [
            {"title": "With snippet", "url": "https://a.example", "content": "s", "score": 0.0},
            {"title": "Desc only", "url": "https://b.example", "content": "d", "score": 0.0},
        ]

    def test_extract_search_results_handles_nested_shape(self):
        """Keyless/Search API returns results nested under web/news — flatten."""
        nested = {
            "results": {
                "web": [{"title": "w", "url": "https://w.example", "snippets": ["s"]}],
                "news": [{"title": "n", "url": "https://n.example", "description": "d"}],
            }
        }
        flat = YoucomSearchAPIWrapper._extract_search_results(nested)
        assert [r["title"] for r in flat] == ["w", "n"]

    def test_extract_search_results_rejects_non_dict(self):
        with pytest.raises(ValueError):
            YoucomSearchAPIWrapper._extract_search_results(["not", "a", "dict"])
        with pytest.raises(ValueError):
            YoucomSearchAPIWrapper._extract_search_results({"results": "bad"})

    @patch("openjiuwen_deepsearch.framework.openjiuwen.tools.search_api.youcom.api_wrapper.httpx.AsyncClient")
    @patch("openjiuwen_deepsearch.framework.openjiuwen.tools.search_api.youcom.api_wrapper.SslUtils.get_ssl_config")
    def test_aresults_async(self, mock_ssl, mock_async_client_cls):
        mock_ssl.return_value = (False, None)
        mock_resp = AsyncMock()
        mock_resp.status_code = 200
        mock_resp.reason_phrase = "OK"
        mock_resp.text = json.dumps(
            {"results": [{"title": "T", "url": "https://x.example", "snippet": "c"}]}
        )
        mock_client = AsyncMock()
        mock_client.post.return_value = mock_resp
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = None
        mock_async_client_cls.return_value = mock_client

        import asyncio

        w = YoucomSearchAPIWrapper(
            search_api_key=bytearray(b"fake_ydc_key"),
            search_url=SecretStr("https://api.example.com"),
            max_web_search_results=1,
        )
        out = asyncio.run(w.aresults("q"))
        assert out == [
            {"title": "T", "url": "https://x.example", "content": "c", "score": 0.0},
        ]
        # Verify the request used the Agents endpoint and attribution header.
        mock_client.post.assert_called_once()
        cargs = mock_client.post.call_args
        assert cargs[0][0] == "https://api.example.com/v1/agents/search"
        assert cargs[1]["headers"]["X-API-Key"] == "fake_ydc_key"
        assert cargs[1]["headers"]["User-Agent"] == YOUCOM_USER_AGENT
        assert cargs[1]["json"] == {"query": "q", "max_results": 1}

    def test_resolved_search_url_defaults_when_empty(self):
        w = YoucomSearchAPIWrapper(
            search_api_key=bytearray(b"k"),
            search_url=SecretStr(""),
        )
        assert w._resolved_search_url == DEFAULT_YOUCOM_SEARCH_URL

    def test_resolved_search_url_strips_trailing_slash(self):
        w = YoucomSearchAPIWrapper(
            search_api_key=bytearray(b"k"),
            search_url=SecretStr("https://api.example.com/"),
        )
        assert w._resolved_search_url == "https://api.example.com"
