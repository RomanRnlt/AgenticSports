"""LLM backend for ReAgt using Gemini 2.5 Flash via google-genai SDK."""

import os

from google import genai

MODEL = "gemini-2.5-flash"


def get_client() -> genai.Client:
    """Create a Gemini client using the API key from environment."""
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not set in environment")
    return genai.Client(api_key=api_key)


def test_connection() -> str:
    """Send a test prompt to Gemini and return the response text."""
    client = get_client()
    response = client.models.generate_content(
        model=MODEL,
        contents="Say 'ReAgt connected successfully' and nothing else.",
    )
    return response.text
