"""Couche de compatibilité pour les différentes versions de LangChain."""

try:
    # LangChain >= 1.0
    from langchain_classic.agents import create_react_agent, AgentExecutor
except ImportError:
    # LangChain 0.2.x
    from langchain.agents import create_react_agent, AgentExecutor

try:
    from langchain_community.chat_models import ChatOllama
except ImportError:
    try:
        from langchain_ollama import ChatOllama
    except ImportError:
        ChatOllama = None

__all__ = ["create_react_agent", "AgentExecutor", "ChatOllama"]
