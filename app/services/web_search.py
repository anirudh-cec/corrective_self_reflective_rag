from tavily import TavilyClient
from app.config import get_settings
from loguru import logger


class WebSearchService:
    def __init__(self):
        self.settings = get_settings()
        self.client = TavilyClient(api_key=self.settings.tavily_api_key)
    
    def search(
        self,
        query: str,
        max_results: int = 3
    ) -> list[dict]:
        """Search web using Tavily"""
        try:
            response = self.client.search(
                query=query,
                max_results=max_results,
                search_depth="basic",
                include_answer=True
            )
            
            results = []
            for result in response.get('results', []):
                results.append({
                    "title": result.get('title'),
                    "url": result.get('url'),
                    "content": result.get('content'),
                    "score": result.get('score', 0.0)
                })
            
            logger.info(f"Web search returned {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"Web search error: {e}")
            return []
