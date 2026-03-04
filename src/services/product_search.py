"""Product search service — abstract provider with Amazon PA-API 5.0 implementation.

Provides a protocol-based interface for searching product catalogs and
an ``AmazonProductSearch`` implementation using the Product Advertising
API 5.0 (PA-API).

Usage::

    from src.services.product_search import get_product_search_provider

    provider = get_product_search_provider()
    if provider:
        results = await provider.search("running shoes", category="Shoes")
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Protocol

import httpx

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ProductResult:
    """Immutable representation of a single product search result."""

    name: str
    image_url: str | None
    price: float | None
    currency: str
    product_url: str
    source: str  # "amazon_api", "llm", etc.


# ---------------------------------------------------------------------------
# Provider protocol
# ---------------------------------------------------------------------------


class ProductSearchProvider(Protocol):
    """Abstract interface for product search backends."""

    name: str

    async def search(
        self,
        query: str,
        category: str | None = None,
        max_results: int = 3,
    ) -> list[ProductResult]: ...


# ---------------------------------------------------------------------------
# AWS Signature V4 helpers
# ---------------------------------------------------------------------------


def _hmac_sha256(key: bytes, message: str) -> bytes:
    """Compute HMAC-SHA256 of *message* with *key*."""
    return hmac.new(key, message.encode("utf-8"), hashlib.sha256).digest()


def _sha256_hex(payload: str) -> str:
    """Return the hex-encoded SHA-256 digest of *payload*."""
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _build_signing_key(
    secret_key: str,
    date_stamp: str,
    region: str,
    service: str,
) -> bytes:
    """Derive the AWS Signature V4 signing key."""
    k_date = _hmac_sha256(f"AWS4{secret_key}".encode("utf-8"), date_stamp)
    k_region = _hmac_sha256(k_date, region)
    k_service = _hmac_sha256(k_region, service)
    return _hmac_sha256(k_service, "aws4_request")


def _build_authorization_header(
    *,
    access_key: str,
    secret_key: str,
    host: str,
    region: str,
    path: str,
    payload: str,
    amz_date: str,
    date_stamp: str,
) -> dict[str, str]:
    """Build signed headers for an AWS Signature V4 request.

    Returns a dict of headers including Authorization, x-amz-date,
    host, and content-type.
    """
    service = "ProductAdvertisingAPI"
    content_type = "application/json; charset=UTF-8"
    target = "com.amazon.paapi5.v1.ProductAdvertisingAPIv1.SearchItems"

    # Canonical headers (must be sorted)
    canonical_headers = (
        f"content-encoding:amz-1.0\n"
        f"content-type:{content_type}\n"
        f"host:{host}\n"
        f"x-amz-date:{amz_date}\n"
        f"x-amz-target:{target}\n"
    )
    signed_headers = "content-encoding;content-type;host;x-amz-date;x-amz-target"

    payload_hash = _sha256_hex(payload)

    canonical_request = "\n".join(
        [
            "POST",
            path,
            "",  # empty query string
            canonical_headers,
            signed_headers,
            payload_hash,
        ]
    )

    credential_scope = f"{date_stamp}/{region}/{service}/aws4_request"
    string_to_sign = "\n".join(
        [
            "AWS4-HMAC-SHA256",
            amz_date,
            credential_scope,
            _sha256_hex(canonical_request),
        ]
    )

    signing_key = _build_signing_key(secret_key, date_stamp, region, service)
    signature = hmac.new(
        signing_key, string_to_sign.encode("utf-8"), hashlib.sha256
    ).hexdigest()

    authorization = (
        f"AWS4-HMAC-SHA256 "
        f"Credential={access_key}/{credential_scope}, "
        f"SignedHeaders={signed_headers}, "
        f"Signature={signature}"
    )

    return {
        "Authorization": authorization,
        "content-encoding": "amz-1.0",
        "content-type": content_type,
        "host": host,
        "x-amz-date": amz_date,
        "x-amz-target": target,
    }


# ---------------------------------------------------------------------------
# Amazon PA-API 5.0 implementation
# ---------------------------------------------------------------------------


def _extract_product(item: dict, partner_tag: str) -> ProductResult:
    """Parse a single PA-API SearchResult item into a ``ProductResult``.

    Gracefully handles missing nested fields.
    """
    title_info = item.get("ItemInfo", {}).get("Title", {})
    name = title_info.get("DisplayValue", "Unknown Product")

    image_url = (
        item.get("Images", {})
        .get("Primary", {})
        .get("Large", {})
        .get("URL")
    )

    # Price may be absent for out-of-stock or marketplace items
    listings = item.get("Offers", {}).get("Listings", [])
    price: float | None = None
    currency = "EUR"
    if listings:
        price_info = listings[0].get("Price", {})
        price = price_info.get("Amount")
        currency = price_info.get("Currency", "EUR")

    product_url = item.get(
        "DetailPageURL",
        f"https://www.amazon.de/dp/{item.get('ASIN', '')}?tag={partner_tag}",
    )

    return ProductResult(
        name=name,
        image_url=image_url,
        price=price,
        currency=currency,
        product_url=product_url,
        source="amazon_api",
    )


class AmazonProductSearch:
    """Product search via Amazon Product Advertising API 5.0.

    Uses AWS Signature V4 signing and ``httpx.AsyncClient`` for HTTP.
    """

    name: str = "amazon_api"

    def __init__(
        self,
        *,
        access_key: str,
        secret_key: str,
        partner_tag: str,
        marketplace: str = "www.amazon.de",
        region: str = "eu-west-1",
    ) -> None:
        self._access_key = access_key
        self._secret_key = secret_key
        self._partner_tag = partner_tag
        self._marketplace = marketplace
        self._region = region
        self._host = f"webservices.{marketplace.removeprefix('www.')}"
        self._path = "/paapi5/searchitems"

    async def search(
        self,
        query: str,
        category: str | None = None,
        max_results: int = 3,
    ) -> list[ProductResult]:
        """Search Amazon for products matching *query*.

        Returns up to *max_results* items. On any error, logs a warning
        and returns an empty list — never raises.
        """
        try:
            return await self._do_search(query, category, max_results)
        except Exception:
            logger.warning(
                "Amazon product search failed for query=%r",
                query,
                exc_info=True,
            )
            return []

    async def _do_search(
        self,
        query: str,
        category: str | None,
        max_results: int,
    ) -> list[ProductResult]:
        """Execute the PA-API SearchItems request."""
        payload = self._build_request_body(query, category, max_results)
        payload_str = json.dumps(payload)

        now = datetime.now(timezone.utc)
        amz_date = now.strftime("%Y%m%dT%H%M%SZ")
        date_stamp = now.strftime("%Y%m%d")

        headers = _build_authorization_header(
            access_key=self._access_key,
            secret_key=self._secret_key,
            host=self._host,
            region=self._region,
            path=self._path,
            payload=payload_str,
            amz_date=amz_date,
            date_stamp=date_stamp,
        )

        url = f"https://{self._host}{self._path}"

        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(url, content=payload_str, headers=headers)
            response.raise_for_status()

        data = response.json()
        items = data.get("SearchResult", {}).get("Items", [])

        return [_extract_product(item, self._partner_tag) for item in items]

    def _build_request_body(
        self,
        query: str,
        category: str | None,
        max_results: int,
    ) -> dict:
        """Build the PA-API 5.0 SearchItems request body."""
        body: dict = {
            "Keywords": query,
            "Resources": [
                "Images.Primary.Large",
                "ItemInfo.Title",
                "Offers.Listings.Price",
            ],
            "ItemCount": min(max_results, 10),  # PA-API max is 10
            "PartnerTag": self._partner_tag,
            "PartnerType": "Associates",
            "Marketplace": self._marketplace,
        }
        if category:
            body["SearchIndex"] = category
        return body


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def get_product_search_provider() -> ProductSearchProvider | None:
    """Return a configured product search provider, or ``None`` if unconfigured.

    Reads Amazon PA-API credentials from application settings.  All three
    values (access key, secret key, affiliate tag) must be set.
    """
    from src.config import get_settings

    settings = get_settings()

    access_key = settings.amazon_pa_api_access_key
    secret_key = settings.amazon_pa_api_secret_key
    partner_tag = settings.amazon_affiliate_tag

    if not (access_key and secret_key and partner_tag):
        logger.debug("Amazon PA-API credentials not configured — provider disabled")
        return None

    return AmazonProductSearch(
        access_key=access_key,
        secret_key=secret_key,
        partner_tag=partner_tag,
    )
