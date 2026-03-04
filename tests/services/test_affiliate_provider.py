"""Tests for src.services.affiliate_provider.

Covers:
- LinkStrategy enum values
- AffiliateProvider immutability and tuple domain_patterns
- build_providers: env-driven registration
- find_provider: domain matching, priority, subdomains
- affiliatize: PARAM_INJECT, REDIRECT_WRAP, API_LOOKUP, no-match passthrough
"""

from __future__ import annotations

import urllib.parse
from dataclasses import FrozenInstanceError
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from src.services.affiliate_provider import (
    AffiliateProvider,
    LinkStrategy,
    affiliatize,
    build_providers,
    find_provider,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_settings(
    amazon_tag: str = "",
    awin_id: str = "",
    awin_merchant: str = "",
) -> SimpleNamespace:
    """Build a minimal settings-like object for testing."""
    return SimpleNamespace(
        amazon_affiliate_tag=amazon_tag,
        awin_affiliate_id=awin_id,
        awin_adidas_merchant_id=awin_merchant,
    )


def _amazon_provider(tag: str = "mystore-21") -> AffiliateProvider:
    return AffiliateProvider(
        name="Amazon Associates",
        strategy=LinkStrategy.PARAM_INJECT,
        enabled=True,
        priority=10,
        domain_patterns=("amazon.de", "amazon.com", "amazon.co.uk", "amzn.to"),
        param_name="tag",
        param_value=tag,
    )


def _awin_provider(
    affiliate_id: str = "123",
    merchant_id: str = "456",
) -> AffiliateProvider:
    return AffiliateProvider(
        name="Awin Adidas",
        strategy=LinkStrategy.REDIRECT_WRAP,
        enabled=True,
        priority=20,
        domain_patterns=("adidas.de", "adidas.com"),
        redirect_template=(
            "https://www.awin1.com/cread.php"
            "?awinmid={merchant_id}"
            "&awinaffid={affiliate_id}"
            "&ued={url}"
        ),
        merchant_id=merchant_id,
        affiliate_id=affiliate_id,
    )


def _api_provider() -> AffiliateProvider:
    return AffiliateProvider(
        name="Future API",
        strategy=LinkStrategy.API_LOOKUP,
        enabled=True,
        priority=30,
        domain_patterns=("futurestore.com",),
        api_endpoint="https://api.futurestore.com/affiliate",
    )


# ---------------------------------------------------------------------------
# TestLinkStrategy
# ---------------------------------------------------------------------------


class TestLinkStrategy:
    def test_values_exist(self) -> None:
        assert LinkStrategy.PARAM_INJECT.value == "param_inject"
        assert LinkStrategy.REDIRECT_WRAP.value == "redirect_wrap"
        assert LinkStrategy.API_LOOKUP.value == "api_lookup"


# ---------------------------------------------------------------------------
# TestAffiliateProvider
# ---------------------------------------------------------------------------


class TestAffiliateProvider:
    def test_frozen_immutable(self) -> None:
        provider = _amazon_provider()
        with pytest.raises(FrozenInstanceError):
            provider.name = "changed"  # type: ignore[misc]

    def test_domain_patterns_is_tuple(self) -> None:
        provider = _amazon_provider()
        assert isinstance(provider.domain_patterns, tuple)


# ---------------------------------------------------------------------------
# TestBuildProviders
# ---------------------------------------------------------------------------


class TestBuildProviders:
    def test_no_env_vars_returns_empty(self) -> None:
        settings = _make_settings()
        result = build_providers(settings)
        assert result == []

    def test_amazon_registered_when_tag_set(self) -> None:
        settings = _make_settings(amazon_tag="mystore-21")
        result = build_providers(settings)
        assert len(result) == 1
        assert result[0].name == "Amazon Associates"
        assert result[0].strategy == LinkStrategy.PARAM_INJECT
        assert result[0].param_value == "mystore-21"

    def test_awin_registered_when_ids_set(self) -> None:
        settings = _make_settings(awin_id="123", awin_merchant="456")
        result = build_providers(settings)
        assert len(result) == 1
        assert result[0].name == "Awin Adidas"
        assert result[0].strategy == LinkStrategy.REDIRECT_WRAP

    def test_both_providers_registered(self) -> None:
        settings = _make_settings(
            amazon_tag="mystore-21",
            awin_id="123",
            awin_merchant="456",
        )
        result = build_providers(settings)
        assert len(result) == 2
        # Sorted by priority — Amazon (10) before Awin (20)
        assert result[0].name == "Amazon Associates"
        assert result[1].name == "Awin Adidas"


# ---------------------------------------------------------------------------
# TestFindProvider
# ---------------------------------------------------------------------------


class TestFindProvider:
    def test_amazon_url_matches_amazon_provider(self) -> None:
        providers = [_amazon_provider(), _awin_provider()]
        result = find_provider("https://www.amazon.de/dp/B0123", providers=providers)
        assert result is not None
        assert result.name == "Amazon Associates"

    def test_adidas_url_matches_awin_provider(self) -> None:
        providers = [_amazon_provider(), _awin_provider()]
        result = find_provider("https://www.adidas.de/shoes/abc", providers=providers)
        assert result is not None
        assert result.name == "Awin Adidas"

    def test_unknown_domain_returns_none(self) -> None:
        providers = [_amazon_provider(), _awin_provider()]
        result = find_provider("https://www.nike.com/shoes", providers=providers)
        assert result is None

    def test_priority_ordering(self) -> None:
        """When two providers match the same domain, the lower priority wins."""
        low_prio = AffiliateProvider(
            name="LowPrio",
            strategy=LinkStrategy.PARAM_INJECT,
            enabled=True,
            priority=50,
            domain_patterns=("example.com",),
            param_name="ref",
            param_value="low",
        )
        high_prio = AffiliateProvider(
            name="HighPrio",
            strategy=LinkStrategy.PARAM_INJECT,
            enabled=True,
            priority=5,
            domain_patterns=("example.com",),
            param_name="ref",
            param_value="high",
        )
        # Pass in "wrong" order to prove sorting matters
        providers = sorted([low_prio, high_prio], key=lambda p: p.priority)
        result = find_provider("https://www.example.com/page", providers=providers)
        assert result is not None
        assert result.name == "HighPrio"

    def test_subdomain_matching(self) -> None:
        """www.amazon.de should match the pattern 'amazon.de'."""
        providers = [_amazon_provider()]
        result = find_provider("https://www.amazon.de/dp/B999", providers=providers)
        assert result is not None
        assert result.name == "Amazon Associates"


# ---------------------------------------------------------------------------
# TestAffiliatize
# ---------------------------------------------------------------------------


class TestAffiliatize:
    @pytest.mark.asyncio
    async def test_amazon_param_inject(self) -> None:
        providers = [_amazon_provider(tag="teststore-21")]
        url, name = await affiliatize(
            "https://www.amazon.de/dp/B0123",
            providers=providers,
        )
        parsed = urllib.parse.urlparse(url)
        params = urllib.parse.parse_qs(parsed.query)
        assert params["tag"] == ["teststore-21"]
        assert name == "Amazon Associates"

    @pytest.mark.asyncio
    async def test_amazon_existing_params(self) -> None:
        providers = [_amazon_provider(tag="teststore-21")]
        url, name = await affiliatize(
            "https://www.amazon.de/dp/B0123?ref=search",
            providers=providers,
        )
        parsed = urllib.parse.urlparse(url)
        params = urllib.parse.parse_qs(parsed.query)
        assert params["tag"] == ["teststore-21"]
        assert params["ref"] == ["search"]
        assert name == "Amazon Associates"

    @pytest.mark.asyncio
    async def test_awin_redirect_wrap(self) -> None:
        providers = [_awin_provider(affiliate_id="111", merchant_id="222")]
        url, name = await affiliatize(
            "https://www.adidas.de/shoes/ultraboost",
            providers=providers,
        )
        assert url.startswith("https://www.awin1.com/cread.php")
        assert "awinmid=222" in url
        assert "awinaffid=111" in url
        # The original URL should be URL-encoded in the ued parameter
        assert urllib.parse.quote("https://www.adidas.de/shoes/ultraboost", safe="") in url
        assert name == "Awin Adidas"

    @pytest.mark.asyncio
    async def test_no_provider_returns_original_url(self) -> None:
        providers = [_amazon_provider()]
        original = "https://www.nike.com/shoes"
        url, name = await affiliatize(original, providers=providers)
        assert url == original
        assert name is None

    @pytest.mark.asyncio
    async def test_returns_provider_name(self) -> None:
        providers = [_amazon_provider()]
        _, name = await affiliatize(
            "https://www.amazon.com/dp/X",
            providers=providers,
        )
        assert name == "Amazon Associates"

    @pytest.mark.asyncio
    async def test_api_lookup_passthrough(self) -> None:
        providers = [_api_provider()]
        original = "https://www.futurestore.com/product/123"
        url, name = await affiliatize(original, providers=providers)
        assert url == original
        assert name == "Future API"
