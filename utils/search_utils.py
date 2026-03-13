import requests
from config.config import SERPER_API_KEY, SERPER_NUM_RESULTS

SERPER_ENDPOINT = "https://google.serper.dev/search"


def web_search(query, num_results=SERPER_NUM_RESULTS):
    """Calls Serper API and returns formatted web search results as a string."""
    if not SERPER_API_KEY:
        return "Web search unavailable: no Serper API key configured."

    try:
        response = requests.post(
            SERPER_ENDPOINT,
            headers={"X-API-KEY": SERPER_API_KEY, "Content-Type": "application/json"},
            json={"q": query, "num": num_results},
            timeout=10,
        )
        response.raise_for_status()

        organic = response.json().get("organic", [])
        if not organic:
            return f"No web results found for: {query}"

        lines = [f'Web search results for: "{query}"\n']
        for i, item in enumerate(organic[:num_results], 1):
            lines.append(
                f"{i}. {item.get('title', 'No title')}\n"
                f"   {item.get('snippet', '')}\n"
                f"   Source: {item.get('link', '')}\n"
            )
        return "\n".join(lines)

    except requests.exceptions.Timeout:
        return "Web search timed out. Please try again."
    except requests.exceptions.HTTPError as e:
        return f"Web search failed (HTTP {e.response.status_code}). Check your Serper API key."
    except Exception as e:
        return f"Web search error: {e}"
