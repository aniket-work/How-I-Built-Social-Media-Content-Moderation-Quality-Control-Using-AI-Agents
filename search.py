from typing import List
from langchain_community.tools.tavily_search import TavilySearchResults


def search_web(query: str, k: int = 3) -> List[str]:
    """Perform web search using Tavily."""
    web_search_tool = TavilySearchResults(k=k)
    search_results = web_search_tool.invoke(query)
    return [result["content"] for result in search_results]