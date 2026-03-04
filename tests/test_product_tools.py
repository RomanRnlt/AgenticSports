"""Unit tests for src.agent.tools.product_tools.

Covers:
- Tool registration (name, category, parameters)
- recommend_products: enrichment pipeline, fallback, edge cases
- _build_base_rec helper
- _fallback_products helper

All DB, search, and affiliate dependencies are mocked.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.agent.tools.registry import ToolRegistry


USER_ID = "user-product-tools-test"
SESSION_ID = "session-001"
PLAN_ID = "plan-001"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_settings(user_id: str = USER_ID) -> MagicMock:
    s = MagicMock()
    s.agenticsports_user_id = user_id
    return s


def _build_registry(settings: MagicMock | None = None) -> ToolRegistry:
    from src.agent.tools.product_tools import register_product_tools

    registry = ToolRegistry()
    user_model = MagicMock()
    with patch("src.agent.tools.product_tools.get_settings", return_value=settings or _mock_settings()):
        register_product_tools(registry, user_model)
    return registry


def _sample_products() -> list[dict]:
    return [
        {
            "name": "Nike Pegasus 41",
            "category": "shoes",
            "reason": "Great daily trainer",
            "search_query": "Nike Pegasus 41 running shoes",
        },
        {
            "name": "Garmin Forerunner 265",
            "category": "watch",
            "reason": "GPS watch for marathon training",
            "search_query": "Garmin Forerunner 265",
        },
        {
            "name": "Theragun Mini",
            "category": "recovery",
            "reason": "Compact massage gun for recovery",
            "search_query": "Theragun Mini massage gun",
        },
    ]


# ---------------------------------------------------------------------------
# Tool Registration
# ---------------------------------------------------------------------------


class TestToolRegistration:
    def test_recommend_products_registered(self) -> None:
        registry = _build_registry()
        names = [t["name"] for t in registry.list_tools()]
        assert "recommend_products" in names

    def test_tool_category_is_planning(self) -> None:
        registry = _build_registry()
        tools = {t["name"]: t for t in registry.list_tools()}
        assert tools["recommend_products"]["category"] == "planning"

    def test_tool_source_is_native(self) -> None:
        registry = _build_registry()
        tools = {t["name"]: t for t in registry.list_tools()}
        assert tools["recommend_products"]["source"] == "native"


# ---------------------------------------------------------------------------
# recommend_products — empty user
# ---------------------------------------------------------------------------


class TestRecommendProductsNoUser:
    def test_no_user_returns_error(self) -> None:
        settings = _mock_settings(user_id="")
        registry = _build_registry(settings)
        result = registry.execute("recommend_products", {"products": _sample_products()})
        assert "error" in result
        assert result["saved_count"] == 0

    def test_empty_products_returns_error(self) -> None:
        registry = _build_registry()
        result = registry.execute("recommend_products", {"products": []})
        assert "error" in result
        assert result["saved_count"] == 0


# ---------------------------------------------------------------------------
# recommend_products — fallback (no search provider)
# ---------------------------------------------------------------------------


class TestRecommendProductsFallback:
    def test_saves_without_enrichment(self) -> None:
        registry = _build_registry()
        saved_data = [
            {"product_name": "Nike Pegasus 41", "category": "shoes"},
            {"product_name": "Garmin Forerunner 265", "category": "watch"},
            {"product_name": "Theragun Mini", "category": "recovery"},
        ]

        with (
            patch("src.services.product_search.get_product_search_provider", return_value=None),
            patch("src.services.affiliate_provider.affiliatize", new_callable=AsyncMock, return_value=("url", None)),
            patch("src.db.product_recommendations_db.save_recommendations", return_value=saved_data),
        ):
            result = registry.execute("recommend_products", {
                "products": _sample_products(),
                "plan_id": PLAN_ID,
            })

        assert result["saved_count"] == 3
        assert len(result["products"]) == 3

    def test_product_summary_fields(self) -> None:
        registry = _build_registry()
        saved_data = [
            {
                "product_name": "Nike Pegasus 41",
                "category": "shoes",
                "image_url": "https://img.example.com/shoe.jpg",
                "price": 129.99,
                "affiliate_url": "https://aff.example.com/shoe",
            },
        ]

        with (
            patch("src.services.product_search.get_product_search_provider", return_value=None),
            patch("src.services.affiliate_provider.affiliatize", new_callable=AsyncMock, return_value=("url", None)),
            patch("src.db.product_recommendations_db.save_recommendations", return_value=saved_data),
        ):
            result = registry.execute("recommend_products", {
                "products": [_sample_products()[0]],
            })

        product = result["products"][0]
        assert product["name"] == "Nike Pegasus 41"
        assert product["has_image"] is True
        assert product["has_price"] is True
        assert product["has_affiliate"] is True


# ---------------------------------------------------------------------------
# recommend_products — with search provider
# ---------------------------------------------------------------------------


class TestRecommendProductsWithSearch:
    def test_enriches_with_search_data(self) -> None:
        from src.services.product_search import ProductResult

        registry = _build_registry()

        mock_provider = MagicMock()
        mock_provider.name = "amazon_api"
        mock_provider.search = AsyncMock(return_value=[
            ProductResult(
                name="Nike Air Zoom Pegasus 41",
                image_url="https://m.media-amazon.com/img.jpg",
                price=129.99,
                currency="EUR",
                product_url="https://www.amazon.de/dp/B0EXAMPLE",
                source="amazon_api",
            ),
        ])

        saved_data = [
            {
                "product_name": "Nike Air Zoom Pegasus 41",
                "image_url": "https://m.media-amazon.com/img.jpg",
                "price": 129.99,
                "affiliate_url": "https://www.amazon.de/dp/B0EXAMPLE?tag=athletly-21",
                "category": "shoes",
            },
        ]

        with (
            patch("src.services.product_search.get_product_search_provider", return_value=mock_provider),
            patch("src.services.affiliate_provider.affiliatize", new_callable=AsyncMock, return_value=(
                "https://www.amazon.de/dp/B0EXAMPLE?tag=athletly-21", "amazon_associates",
            )),
            patch("src.db.product_recommendations_db.save_recommendations", return_value=saved_data),
        ):
            result = registry.execute("recommend_products", {
                "products": [_sample_products()[0]],
            })

        assert result["saved_count"] == 1
        assert result["products"][0]["has_image"] is True
        assert result["products"][0]["has_price"] is True

    def test_search_failure_falls_back_gracefully(self) -> None:
        registry = _build_registry()

        mock_provider = MagicMock()
        mock_provider.name = "amazon_api"
        mock_provider.search = AsyncMock(side_effect=Exception("API timeout"))

        saved_data = [{"product_name": "Nike Pegasus 41", "category": "shoes"}]

        with (
            patch("src.services.product_search.get_product_search_provider", return_value=mock_provider),
            patch("src.services.affiliate_provider.affiliatize", new_callable=AsyncMock, return_value=("url", None)),
            patch("src.db.product_recommendations_db.save_recommendations", return_value=saved_data),
        ):
            result = registry.execute("recommend_products", {
                "products": [_sample_products()[0]],
            })

        # Should still save without crashing
        assert result["saved_count"] == 1


# ---------------------------------------------------------------------------
# recommend_products — with session_id and plan_id
# ---------------------------------------------------------------------------


class TestRecommendProductsLinking:
    def test_session_id_passed_through(self) -> None:
        registry = _build_registry()
        saved_data = [{"product_name": "Shoes", "category": "shoes"}]

        with (
            patch("src.services.product_search.get_product_search_provider", return_value=None),
            patch("src.services.affiliate_provider.affiliatize", new_callable=AsyncMock, return_value=("url", None)),
            patch("src.db.product_recommendations_db.save_recommendations", return_value=saved_data) as mock_save,
        ):
            registry.execute("recommend_products", {
                "products": [{"name": "Shoes", "reason": "Good"}],
                "session_id": SESSION_ID,
            })

        call_args = mock_save.call_args[0]
        recs = call_args[1]
        assert recs[0]["session_id"] == SESSION_ID

    def test_plan_id_passed_through(self) -> None:
        registry = _build_registry()
        saved_data = [{"product_name": "Watch", "category": "watch"}]

        with (
            patch("src.services.product_search.get_product_search_provider", return_value=None),
            patch("src.services.affiliate_provider.affiliatize", new_callable=AsyncMock, return_value=("url", None)),
            patch("src.db.product_recommendations_db.save_recommendations", return_value=saved_data) as mock_save,
        ):
            registry.execute("recommend_products", {
                "products": [{"name": "Watch", "reason": "Track runs"}],
                "plan_id": PLAN_ID,
            })

        call_args = mock_save.call_args[0]
        recs = call_args[1]
        assert recs[0]["plan_id"] == PLAN_ID


# ---------------------------------------------------------------------------
# _build_base_rec helper
# ---------------------------------------------------------------------------


class TestBuildBaseRec:
    def test_builds_from_product_dict(self) -> None:
        from src.agent.tools.product_tools import _build_base_rec

        product = {
            "name": "Garmin 265",
            "reason": "GPS watch",
            "category": "watch",
            "sport": "running",
            "search_query": "Garmin Forerunner 265",
        }
        rec = _build_base_rec(product, USER_ID, SESSION_ID, PLAN_ID)

        assert rec["product_name"] == "Garmin 265"
        assert rec["reason"] == "GPS watch"
        assert rec["category"] == "watch"
        assert rec["sport"] == "running"
        assert rec["search_query"] == "Garmin Forerunner 265"
        assert rec["user_id"] == USER_ID
        assert rec["session_id"] == SESSION_ID
        assert rec["plan_id"] == PLAN_ID
        assert rec["source"] == "llm"

    def test_missing_search_query_falls_back_to_name(self) -> None:
        from src.agent.tools.product_tools import _build_base_rec

        product = {"name": "Foam Roller", "reason": "Recovery"}
        rec = _build_base_rec(product, USER_ID, None, None)

        assert rec["search_query"] == "Foam Roller"


# ---------------------------------------------------------------------------
# _fallback_products helper
# ---------------------------------------------------------------------------


class TestFallbackProducts:
    def test_builds_list_of_base_recs(self) -> None:
        from src.agent.tools.product_tools import _fallback_products

        products = [
            {"name": "A", "reason": "R1"},
            {"name": "B", "reason": "R2"},
        ]
        result = _fallback_products(products, USER_ID, None, None)

        assert len(result) == 2
        assert result[0]["product_name"] == "A"
        assert result[1]["product_name"] == "B"
        assert all(r["source"] == "llm" for r in result)
