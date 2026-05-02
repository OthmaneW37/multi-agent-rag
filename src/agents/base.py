"""Classe de base pour les agents du systeme WAC Sport Analytics."""

import time
from abc import ABC
from typing import Any, Dict, List, Optional

from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.tools import BaseTool

from src.compat import create_react_agent, AgentExecutor
from src.llm_utils import get_llm


# Instructions anti-hallucination globales injectees dans TOUS les agents
ANTI_HALLUCINATION_INSTRUCTIONS = """
REGLES STRICTES ANTI-HALLUCINATION :
1. Tu ne connais RIEN sur le football en dehors des DONNEES fournies dans ce prompt.
2. Tu ne dois JAMAIS utiliser tes connaissances generales sur le football, les championnats, les equipes ou les joueurs.
3. Si les donnees fournies sont insuffisantes, tu DOIS repondre explicitement que l'information manque.
4. Tu ne dois JAMAIS inventer des noms d'equipes, de joueurs, de scores, de statistiques ou de classements.
5. Tu ne dois JAMAIS parler d'autres championnats (Ligue 1, Premier League, La Liga, Serie A, Bundesliga, etc.).
6. Tu ne parles que de la BOTOLA PRO marocaine et des equipes qui y figurent.
7. Chaque fait que tu enonces doit etre directement issu des donnees fournies dans ce prompt.
8. Si tu ne trouves pas une information dans les donnees fournies, dis-le clairement au lieu de deviner.
"""


class BaseAgent(ABC):
    """Classe de base pour tous les agents du systeme multi-agents."""

    def __init__(
        self,
        name: str,
        role: str,
        tools: List[BaseTool],
        llm: Optional[BaseChatModel] = None,
        temperature: float = 0.7,
    ):
        self.name = name
        self.role = role
        self.tools = tools
        self.llm = llm or get_llm(temperature=temperature)
        self.agent_executor = self._build_agent()

    def _build_agent(self) -> AgentExecutor:
        """Construit l'agent ReAct avec les outils configures."""
        # Template ReAct en francais adapte au football avec guardrails
        template = """{system_prompt}

{anti_hallucination}

Tu as acces aux outils suivants :
{tools}

Utilise le format suivant :

Question : la question d'input que tu dois repondre
Thought : tu dois toujours reflechir a ce que tu dois faire
Action : l'action a entreprendre, doit etre l'une de [{tool_names}]
Action Input : l'entree pour l'action
Observation : le resultat de l'action
... (ce cycle Thought/Action/Action Input/Observation peut se repeter N fois)
Thought : Je sais maintenant la reponse finale
Final Answer : la reponse finale a la question originale

Commence !

Question : {input}
Thought : {agent_scratchpad}"""

        prompt = PromptTemplate.from_template(template).partial(
            system_prompt=self.role,
            anti_hallucination=ANTI_HALLUCINATION_INSTRUCTIONS,
            tools="\n\n".join(
                [f"{tool.name}: {tool.description}" for tool in self.tools]
            ),
            tool_names=", ".join([tool.name for tool in self.tools]),
        )

        agent = create_react_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=prompt,
        )

        return AgentExecutor(
            agent=agent,
            tools=self.tools,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=10,
        )

    def run(
        self, input_text: str, chat_history: Optional[List] = None
    ) -> Dict[str, Any]:
        """
        Execute l'agent avec l'entree donnee.

        Args:
            input_text: Texte d'entree pour l'agent
            chat_history: Historique de conversation optionnel

        Returns:
            Resultat de l'execution de l'agent
        """
        inputs = {"input": input_text}
        if chat_history:
            inputs["chat_history"] = chat_history

        return self.agent_executor.invoke(inputs)


class DirectLLMAgent(ABC):
    """
    Agent qui fonctionne par invocation LLM directe SANS boucle ReAct.

    Ideal pour les agents qui n'ont pas besoin d'outils (analyse de texte,
    redaction, validation). Cela evite les problemes de parsing ReAct
    et reduit les hallucinations.
    """

    def __init__(
        self,
        name: str,
        role: str,
        llm: Optional[BaseChatModel] = None,
        temperature: float = 0.3,
    ):
        self.name = name
        self.role = role
        self.llm = llm or get_llm(temperature=temperature)

    def run(self, input_text: str, **kwargs) -> Dict[str, Any]:
        """
        Execute l'agent en invoquant directement le LLM.
        Implemente un retry avec backoff exponentiel pour les rate limits.

        Args:
            input_text: Le prompt complet (instructions + donnees)

        Returns:
            Resultat sous forme de dict avec cle 'output'
        """
        messages = [
            SystemMessage(content=self.role + "\n\n" + ANTI_HALLUCINATION_INSTRUCTIONS),
            HumanMessage(content=input_text),
        ]

        max_retries = 3
        base_delay = 1.0

        for attempt in range(max_retries):
            try:
                response = self.llm.invoke(messages)
                return {"output": response.content}
            except Exception as exc:
                error_str = str(exc).lower()
                is_rate_limit = any(k in error_str for k in ["rate limit", "429", "rate_limit_exceeded", "too many requests"])
                if is_rate_limit and attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt)
                    print(f"[{self.name}] Rate limit detecte, retry dans {delay}s... (tentative {attempt + 1}/{max_retries})")
                    time.sleep(delay)
                else:
                    # Derniere tentative ou erreur non-recoverable
                    print(f"[{self.name}] Erreur LLM apres {attempt + 1} tentatives: {exc}")
                    raise

        # Ne devrait jamais arriver
        return {"output": "[ERREUR] Impossible d'obtenir une reponse du LLM apres plusieurs tentatives."}
