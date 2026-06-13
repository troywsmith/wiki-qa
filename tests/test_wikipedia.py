import httpx
import pytest
import respx

from wikiqa.wikipedia import WikipediaClient, _strip_html

API = "https://en.wikipedia.org/w/api.php"


def test_strip_html_removes_tags_and_entities():
    assert _strip_html('a <span class="x">b</span> c &quot;d&quot;') == 'a b c "d"'


@respx.mock
@pytest.mark.asyncio
async def test_search_parses_hits():
    respx.get(API).mock(
        return_value=httpx.Response(
            200,
            json={
                "query": {
                    "search": [
                        {"title": "Ada Lovelace", "snippet": "first <b>programmer</b>"},
                    ]
                }
            },
        )
    )
    results = await WikipediaClient().search("ada lovelace")
    assert results == [
        {
            "title": "Ada Lovelace",
            "snippet": "first programmer",
            "url": "https://en.wikipedia.org/wiki/Ada_Lovelace",
        }
    ]


@respx.mock
@pytest.mark.asyncio
async def test_get_article_returns_extract():
    respx.get(API).mock(
        return_value=httpx.Response(
            200,
            json={"query": {"pages": {"99": {"title": "Python", "extract": "A language.\n"}}}},
        )
    )
    article = await WikipediaClient().get_article("Python")
    assert article["title"] == "Python"
    assert article["extract"] == "A language."
    assert article["url"] == "https://en.wikipedia.org/wiki/Python"


@respx.mock
@pytest.mark.asyncio
async def test_get_article_handles_missing_page():
    respx.get(API).mock(
        return_value=httpx.Response(200, json={"query": {"pages": {"-1": {"missing": ""}}}})
    )
    article = await WikipediaClient().get_article("Nonexistent Title 12345")
    assert article["extract"] == ""
