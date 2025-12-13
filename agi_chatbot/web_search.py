"""Shim for `agi_chatbot.web_search`.

Provides a minimal `web_search_tool` object compatible with both
callable-style usage and the `.search()` method used across the codebase.
"""

from typing import Any, Dict, List


class WebSearchTool:
    def search(
        self, query: str, max_results: int = 5, **kwargs
    ) -> List[Dict[str, Any]]:
        return []

    def __call__(
        self, query: str, max_results: int = 5, **kwargs
    ) -> List[Dict[str, Any]]:
        return self.search(query, max_results=max_results, **kwargs)


# Instance compatible with older callable usage and new `.search()` member access
web_search_tool = WebSearchTool()
