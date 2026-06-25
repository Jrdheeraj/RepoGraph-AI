"""Router logic for classifying intents."""

import logging

logger = logging.getLogger(__name__)


def route_query(query: str) -> str:
    """Classify the user's query intent.

    Intent rules:
    - architecture, design, flow -> architecture
    - docs, documentation, readme -> documentation
    - test, pytest -> testing
    - refactor, optimize, improve -> refactoring
    - else -> default

    Args:
        query: The user query.

    Returns:
        The classified intent string.
    """
    logger.info("Routing query to determine intent", extra={"query_preview": query[:50]})
    q_lower = query.lower()

    if any(kw in q_lower for kw in ("architecture", "design", "flow")):
        intent = "architecture"
    elif any(kw in q_lower for kw in ("docs", "documentation", "readme")):
        intent = "documentation"
    elif any(kw in q_lower for kw in ("test", "pytest")):
        intent = "testing"
    elif any(kw in q_lower for kw in ("refactor", "optimize", "improve")):
        intent = "refactoring"
    else:
        intent = "default"

    logger.info("Query intent determined", extra={"intent": intent})
    return intent
