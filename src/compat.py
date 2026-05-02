"""Couche de compatibilite pour les differentes versions de LangChain."""

try:
    from langchain_classic.agents import create_react_agent, AgentExecutor
except ImportError:
    from langchain.agents import create_react_agent, AgentExecutor

try:
    from langchain_community.chat_models import ChatOllama
except ImportError:
    try:
        from langchain_ollama import ChatOllama
    except ImportError:
        ChatOllama = None

__all__ = ["create_react_agent", "AgentExecutor", "ChatOllama"]
