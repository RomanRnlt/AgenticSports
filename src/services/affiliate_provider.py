"""Config-driven affiliate provider system.

Matches product URLs against registered providers and decorates them with
affiliate tracking parameters.  Provider list is built once at import time
from environment variables via ``src.config.get_settings()``.

Usage::

    from src.services.affiliate_provider import affiliatize, find_provider

    url, provider_name = await affiliatize("https://www.amazon.de/dp/B0123")
    # url = "https://www.amazon.de/dp/B0123?tag=mystore-21"
    # provider_name = "Amazon Associates"
"""

from __future__ import annotations

import logging
import urllib.parse
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Strategy enum
# ---------------------------------------------------------------------------


class LinkStrategy(Enum):
    """How an affiliate provider decorates outbound links."""

    PARAM_INJECT = "param_inject"
    REDIRECT_WRAP = "redirect_wrap"
    API_LOOKUP = "api_lookup"


# ---------------------------------------------------------------------------
# Provider dataclass (frozen / immutable)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AffiliateProvider:
    """Immutable descriptor for a single affiliate network configuration."""

    name: str
    strategy: LinkStrategy
    enabled: bool
    priority: int  # lower = preferred
    domain_patterns: tuple[str, ...]

    # PARAM_INJECT fields
    param_name: str = ""
    param_value: str = ""

    # REDIRECT_WRAP fields
    redirect_template: str = ""
    merchant_id: str = ""
    affiliate_id: str = ""

    # API_LOOKUP fields
    api_endpoint: str = ""


# ---------------------------------------------------------------------------
# Provider construction
# ---------------------------------------------------------------------------


def build_providers(settings: object) -> list[AffiliateProvider]:
    """Build the provider list from application settings.

    Only registers a provider when its required credentials are present.
    Returns a new list sorted by priority (ascending).
    """
    providers: list[AffiliateProvider] = []

    amazon_tag = getattr(settings, "amazon_affiliate_tag", "")
    if amazon_tag:
        providers.append(
            AffiliateProvider(
                name="Amazon Associates",
                strategy=LinkStrategy.PARAM_INJECT,
                enabled=True,
                priority=10,
                domain_patterns=("amazon.de", "amazon.com", "amazon.co.uk", "amzn.to"),
                param_name="tag",
                param_value=amazon_tag,
            )
        )

    awin_id = getattr(settings, "awin_affiliate_id", "")
    awin_merchant = getattr(settings, "awin_adidas_merchant_id", "")
    if awin_id and awin_merchant:
        providers.append(
            AffiliateProvider(
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
                merchant_id=awin_merchant,
                affiliate_id=awin_id,
            )
        )

    return sorted(providers, key=lambda p: p.priority)


# ---------------------------------------------------------------------------
# Module-level provider list (populated once at import)
# ---------------------------------------------------------------------------


def _load_providers() -> list[AffiliateProvider]:
    """Load providers from current settings.  Swallowed errors return []."""
    try:
        from src.config import get_settings

        return build_providers(get_settings())
    except Exception:  # noqa: BLE001
        logger.warning("Failed to load affiliate providers from settings", exc_info=True)
        return []


_PROVIDERS: list[AffiliateProvider] = _load_providers()


# ---------------------------------------------------------------------------
# Lookup
# ---------------------------------------------------------------------------


def find_provider(url: str, providers: list[AffiliateProvider] | None = None) -> AffiliateProvider | None:
    """Return the highest-priority enabled provider whose domain matches *url*.

    Checks whether any ``domain_patterns`` entry appears in the URL's hostname.
    If *providers* is ``None``, uses the module-level ``_PROVIDERS`` list.
    """
    pool = providers if providers is not None else _PROVIDERS

    parsed = urllib.parse.urlparse(url)
    hostname = parsed.hostname or ""

    for provider in pool:
        if not provider.enabled:
            continue
        for pattern in provider.domain_patterns:
            if hostname == pattern or hostname.endswith(f".{pattern}"):
                return provider

    return None


# ---------------------------------------------------------------------------
# Strategy implementations
# ---------------------------------------------------------------------------


def _apply_param_inject(url: str, provider: AffiliateProvider) -> str:
    """Add an affiliate query parameter to the URL."""
    parsed = urllib.parse.urlparse(url)
    params = urllib.parse.parse_qs(parsed.query)
    params[provider.param_name] = [provider.param_value]

    new_query = urllib.parse.urlencode(params, doseq=True)
    new_parsed = parsed._replace(query=new_query)
    return urllib.parse.urlunparse(new_parsed)


def _apply_redirect_wrap(url: str, provider: AffiliateProvider) -> str:
    """Wrap the URL in a redirect through the affiliate network."""
    encoded_url = urllib.parse.quote(url, safe="")
    return provider.redirect_template.format(
        merchant_id=provider.merchant_id,
        affiliate_id=provider.affiliate_id,
        url=encoded_url,
    )


def _apply_api_lookup(url: str, _provider: AffiliateProvider) -> str:
    """Placeholder for API-based affiliate link resolution."""
    return url


_STRATEGY_MAP = {
    LinkStrategy.PARAM_INJECT: _apply_param_inject,
    LinkStrategy.REDIRECT_WRAP: _apply_redirect_wrap,
    LinkStrategy.API_LOOKUP: _apply_api_lookup,
}


# ---------------------------------------------------------------------------
# Public async API
# ---------------------------------------------------------------------------


async def affiliatize(
    url: str,
    providers: list[AffiliateProvider] | None = None,
) -> tuple[str, str | None]:
    """Decorate *url* with affiliate tracking if a provider matches.

    Returns ``(decorated_url, provider_name)``.  When no provider matches,
    returns ``(url, None)`` unchanged.
    """
    provider = find_provider(url, providers=providers)
    if provider is None:
        return url, None

    apply_fn = _STRATEGY_MAP.get(provider.strategy)
    if apply_fn is None:
        logger.warning("No strategy handler for %s", provider.strategy)
        return url, provider.name

    decorated = apply_fn(url, provider)
    return decorated, provider.name
