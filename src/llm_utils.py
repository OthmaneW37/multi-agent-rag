"""Utilitaires pour la configuration des LLMs."""

from langchain_core.language_models import BaseChatModel
from langchain_community.chat_models import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq

from src.config import config


def get_llm(temperature: float = 0.7) -> BaseChatModel:
    """
    Retourne une instance de LLM configuree selon le provider choisi.

    Args:
        temperature: Temperature pour la generation

    Returns:
        Instance de BaseChatModel
    """
    provider = config.llm_provider.lower()

    if provider == "ollama":
        return ChatOllama(
            model=config.ollama_model,
            base_url=config.ollama_base_url,
            temperature=temperature,
            timeout=300,
        )
    elif provider == "openai":
        if not config.openai_api_key:
            raise ValueError("OPENAI_API_KEY non configuree")
        return ChatOpenAI(
            model=config.openai_model,
            api_key=config.openai_api_key,
            temperature=temperature,
        )
    elif provider == "groq":
        if not config.groq_api_key:
            raise ValueError("GROQ_API_KEY non configuree")
        return ChatGroq(
            model=config.groq_model,
            api_key=config.groq_api_key,
            temperature=temperature,
        )
    else:
        raise ValueError(f"Provider LLM non supporte : {provider}")
