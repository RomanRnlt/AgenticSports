"""Tests for src.services.product_search.

Covers:
- ProductResult: frozen immutability, field access
- AmazonProductSearch: success, error, empty, partial data
- get_product_search_provider: configured vs unconfigured
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from src.services.product_search import (
    AmazonProductSearch,
    ProductResult,
    get_product_search_provider,
)


# ---------------------------------------------------------------------------
# Sample PA-API 5.0 response data
# ---------------------------------------------------------------------------

SAMPLE_ITEM = {
    "ASIN": "B0EXAMPLE",
    "DetailPageURL": "https://www.amazon.de/dp/B0EXAMPLE",
    "ItemInfo": {
        "Title": {"DisplayValue": "Nike Pegasus 41"},
    },
    "Images": {
        "Primary": {
            "Large": {"URL": "https://m.media-amazon.com/images/I/example.jpg"},
        },
    },
    "Offers": {
        "Listings": [
            {
                "Price": {"Amount": 129.99, "Currency": "EUR"},
            },
        ],
    },
}

SAMPLE_RESPONSE_JSON = {
    "SearchResult": {
        "Items": [SAMPLE_ITEM],
    },
}


# ---------------------------------------------------------------------------
# ProductResult
# ---------------------------------------------------------------------------


class TestProductResult:
    def test_frozen(self) -> None:
        result = ProductResult(
            name="Test",
            image_url="https://example.com/img.jpg",
            price=49.99,
            currency="EUR",
            product_url="https://example.com/product",
            source="amazon_api",
        )
        with pytest.raises(FrozenInstanceError):
            result.name = "Modified"  # type: ignore[misc]

    def test_fields(self) -> None:
        result = ProductResult(
            name="Nike Pegasus 41",
            image_url="https://m.media-amazon.com/images/I/example.jpg",
            price=129.99,
            currency="EUR",
            product_url="https://www.amazon.de/dp/B0EXAMPLE",
            source="amazon_api",
        )
        assert result.name == "Nike Pegasus 41"
        assert result.image_url == "https://m.media-amazon.com/images/I/example.jpg"
        assert result.price == 129.99
        assert result.currency == "EUR"
        assert result.product_url == "https://www.amazon.de/dp/B0EXAMPLE"
        assert result.source == "amazon_api"


# ---------------------------------------------------------------------------
# AmazonProductSearch
# ---------------------------------------------------------------------------


def _make_provider() -> AmazonProductSearch:
    return AmazonProductSearch(
        access_key="AKIAIOSFODNN7EXAMPLE",
        secret_key="wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
        partner_tag="athletly-21",
    )


def _mock_httpx_response(json_data: dict, status_code: int = 200) -> MagicMock:
    """Create a mock httpx.Response with the given JSON data."""
    response = MagicMock(spec=httpx.Response)
    response.status_code = status_code
    response.json.return_value = json_data
    response.raise_for_status.return_value = None
    return response


class TestAmazonProductSearch:
    def test_name_is_amazon_api(self) -> None:
        provider = _make_provider()
        assert provider.name == "amazon_api"

    @pytest.mark.asyncio
    async def test_search_success(self) -> None:
        provider = _make_provider()
        mock_response = _mock_httpx_response(SAMPLE_RESPONSE_JSON)

        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.post.return_value = mock_response
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("src.services.product_search.httpx.AsyncClient", return_value=mock_client):
            results = await provider.search("running shoes")

        assert len(results) == 1
        result = results[0]
        assert result.name == "Nike Pegasus 41"
        assert result.image_url == "https://m.media-amazon.com/images/I/example.jpg"
        assert result.price == 129.99
        assert result.currency == "EUR"
        assert result.product_url == "https://www.amazon.de/dp/B0EXAMPLE"
        assert result.source == "amazon_api"

    @pytest.mark.asyncio
    async def test_search_error_returns_empty(self) -> None:
        provider = _make_provider()

        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.post.side_effect = httpx.ConnectError("Connection refused")
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("src.services.product_search.httpx.AsyncClient", return_value=mock_client):
            results = await provider.search("running shoes")

        assert results == []

    @pytest.mark.asyncio
    async def test_search_empty_results(self) -> None:
        provider = _make_provider()
        empty_response = {"SearchResult": {"Items": []}}
        mock_response = _mock_httpx_response(empty_response)

        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.post.return_value = mock_response
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("src.services.product_search.httpx.AsyncClient", return_value=mock_client):
            results = await provider.search("nonexistent product xyz")

        assert results == []

    @pytest.mark.asyncio
    async def test_search_partial_data(self) -> None:
        """Items with missing price or image should still parse successfully."""
        provider = _make_provider()
        partial_item = {
            "ASIN": "B0PARTIAL",
            "DetailPageURL": "https://www.amazon.de/dp/B0PARTIAL",
            "ItemInfo": {
                "Title": {"DisplayValue": "Budget Running Shoe"},
            },
            # No Images key
            # No Offers key
        }
        partial_response = {"SearchResult": {"Items": [partial_item]}}
        mock_response = _mock_httpx_response(partial_response)

        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.post.return_value = mock_response
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("src.services.product_search.httpx.AsyncClient", return_value=mock_client):
            results = await provider.search("budget shoes")

        assert len(results) == 1
        result = results[0]
        assert result.name == "Budget Running Shoe"
        assert result.image_url is None
        assert result.price is None
        assert result.currency == "EUR"
        assert result.product_url == "https://www.amazon.de/dp/B0PARTIAL"
        assert result.source == "amazon_api"


# ---------------------------------------------------------------------------
# get_product_search_provider
# ---------------------------------------------------------------------------


def _mock_settings(**overrides: str) -> MagicMock:
    """Create a mock Settings object with Amazon PA-API fields."""
    defaults = {
        "amazon_pa_api_access_key": "",
        "amazon_pa_api_secret_key": "",
        "amazon_affiliate_tag": "",
    }
    defaults.update(overrides)
    settings = MagicMock()
    for key, value in defaults.items():
        setattr(settings, key, value)
    return settings


class TestGetProductSearchProvider:
    def test_returns_none_when_no_keys(self) -> None:
        settings = _mock_settings()
        with patch("src.config.get_settings", return_value=settings):
            provider = get_product_search_provider()
        assert provider is None

    def test_returns_provider_when_configured(self) -> None:
        settings = _mock_settings(
            amazon_pa_api_access_key="AKIAIOSFODNN7EXAMPLE",
            amazon_pa_api_secret_key="wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
            amazon_affiliate_tag="athletly-21",
        )
        with patch("src.config.get_settings", return_value=settings):
            provider = get_product_search_provider()
        assert provider is not None
        assert isinstance(provider, AmazonProductSearch)

    def test_provider_has_correct_name(self) -> None:
        settings = _mock_settings(
            amazon_pa_api_access_key="AKIAIOSFODNN7EXAMPLE",
            amazon_pa_api_secret_key="wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
            amazon_affiliate_tag="athletly-21",
        )
        with patch("src.config.get_settings", return_value=settings):
            provider = get_product_search_provider()
        assert provider is not None
        assert provider.name == "amazon_api"
