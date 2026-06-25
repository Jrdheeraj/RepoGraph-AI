"""Agent state definitions for RepoGraph AI."""

from typing import TypedDict


class RepoGraphState(TypedDict, total=False):
    """The state dictionary for the LangGraph workflow."""
    
    messages: list
    query: str
    intent: str
    context: list
    citations: list
    answer: str
    agent_used: str
