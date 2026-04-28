"""Classe de base pour les agents du système."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import BaseTool

from src.compat import create_react_agent, AgentExecutor
from src.llm_utils import get_llm


class BaseAgent(ABC):
    """Classe de base pour tous les agents du système multi-agents."""

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
        from langchain_core.prompts import PromptTemplate

        # Template ReAct compatible avec create_react_agent
        template = """{system_prompt}

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
            tools="\n\n".join([f"{tool.name}: {tool.description}" for tool in self.tools]),
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

        return AgentExecutor(
            agent=agent,
            tools=self.tools,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=10,
        )

    def run(self, input_text: str, chat_history: Optional[List] = None) -> Dict[str, Any]:
        """
        Exécute l'agent avec l'entrée donnée.
        
        Args:
            input_text: Texte d'entrée pour l'agent
            chat_history: Historique de conversation optionnel
        
        Returns:
            Résultat de l'exécution de l'agent
        """
        inputs = {"input": input_text}
        if chat_history:
            inputs["chat_history"] = chat_history

        return self.agent_executor.invoke(inputs)

    def get_system_prompt(self) -> str:
        """Retourne le system prompt specifique a l'agent."""
        return self.role
