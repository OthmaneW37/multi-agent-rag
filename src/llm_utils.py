"""Utilitaires pour la configuration des LLMs."""

from typing import Union

from langchain_core.language_models import BaseChatModel
from langchain_community.chat_models import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_groq import ChatGroq

from src.config import config


def get_llm(temperature: float = 0.7) -> BaseChatModel:
    """
    Retourne une instance de LLM configurée selon le provider choisi.
    
    Args:
        temperature: Température pour la génération
    
    Returns:
        Instance de BaseChatModel
    """
    provider = config.llm_provider.lower()

    if provider == "ollama":
        return ChatOllama(
            model=config.ollama_model,
            base_url=config.ollama_base_url,
            temperature=temperature,
        )
    elif provider == "openai":
        if not config.openai_api_key:
            raise ValueError("OPENAI_API_KEY non configurée")
        return ChatOpenAI(
            model=config.openai_model,
            api_key=config.openai_api_key,
            temperature=temperature,
        )
    elif provider == "anthropic":
        if not config.anthropic_api_key:
            raise ValueError("ANTHROPIC_API_KEY non configurée")
        return ChatAnthropic(
            model=config.anthropic_model,
            api_key=config.anthropic_api_key,
            temperature=temperature,
        )
    elif provider == "groq":
        if not config.groq_api_key:
            raise ValueError("GROQ_API_KEY non configurée")
        return ChatGroq(
            model=config.groq_model,
            api_key=config.groq_api_key,
            temperature=temperature,
        )
    else:
        raise ValueError(f"Provider LLM non supporté : {provider}")
