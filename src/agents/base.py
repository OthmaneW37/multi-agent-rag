"""Classe de base pour les agents du systeme WAC Sport Analytics."""

import re
import time
from abc import ABC
from typing import Any, Dict, List, Optional

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.tools import BaseTool

from src.llm_utils import get_llm


# Instructions anti-hallucination globales injectees dans TOUS les agents
ANTI_HALLUCINATION_INSTRUCTIONS = """
REGLES ANTI-HALLUCINATION :
1. Tu ne connais que les DONNEES fournies ici. Pas de connaissances generales.
2. Si une info manque, dis "Non disponible". N'invente JAMAIS de noms, scores ou stats.
3. Tu ne parles que de la BOTOLA PRO marocaine. Pas d'autres championnats.
4. Chaque fait doit etre issu directement des donnees fournies.
"""


class BaseAgent(ABC):
    """
    Agent ReAct maison robuste.

    Ne passe PAS par AgentExecutor (qui cree des threads internes incompatibles
    avec sentence-transformers/PyTorch sous Streamlit). Le raisonnement ReAct
    est implemente manuellement : on appelle le LLM, on parse Thought/Action,
    on execute l'outil directement via tool.func(), on reboucle.

    Fallback sur DirectLLM si le ReAct echoue apres max_iterations.
    """

    MAX_ITERATIONS = 5

    def __init__(
        self,
        name: str,
        role: str,
        tools: List[BaseTool],
        llm: Optional[BaseChatModel] = None,
        temperature: float = 0.2,
    ):
        self.name = name
        self.role = role
        self.tools = tools
        self.llm = llm or get_llm(temperature=temperature)
        self._tool_map = {tool.name: tool for tool in tools}

    def _build_react_prompt(self, question: str, scratchpad: str) -> str:
        tools_desc = "\n".join(
            [f"- {tool.name}: {tool.description}" for tool in self.tools]
        )
        return f"""{self.role}

{ANTI_HALLUCINATION_INSTRUCTIONS}

Tu as acces aux outils suivants :
{tools_desc}

Utilise le format suivant (STRICTEMENT) :

Thought : reflechis a l'etape suivante
Action : l'action a entreprendre (DOIT etre l'une de : {', '.join(self._tool_map.keys())})
Action Input : l'entree pour l'action
Observation : le resultat de l'action
... (ce cycle peut se repeter jusqu'a 5 fois)
Thought : Je sais maintenant la reponse finale
Final Answer : la reponse finale a la question

Commence !

Question : {question}
{scratchpad}"""

    def _parse_thought_action(self, text: str) -> Optional[Dict[str, str]]:
        """Parse la reponse LLM pour extraire Thought, Action, Action Input."""
        # Cherche Final Answer en priorite
        fa_match = re.search(
            r"Final Answer\s*[:\-]?\s*(.+)", text, re.DOTALL | re.IGNORECASE
        )
        if fa_match:
            return {"type": "final", "answer": fa_match.group(1).strip()}

        # Cherche Thought
        thought_match = re.search(
            r"Thought\s*[:\-]?\s*(.+?)(?=Action|Observation|Final Answer|$)",
            text, re.DOTALL | re.IGNORECASE,
        )
        thought = thought_match.group(1).strip() if thought_match else ""

        # Cherche Action
        action_match = re.search(
            r"Action\s*[:\-]?\s*(\w+)", text, re.IGNORECASE
        )
        if not action_match:
            return None
        action_name = action_match.group(1).strip()

        # Cherche Action Input
        input_match = re.search(
            r"Action Input\s*[:\-]?\s*(.+?)(?=Observation|Thought|Action|Final Answer|$)",
            text, re.DOTALL | re.IGNORECASE,
        )
        action_input = input_match.group(1).strip() if input_match else ""

        return {
            "type": "action",
            "thought": thought,
            "action": action_name,
            "action_input": action_input,
        }

    def _execute_tool(self, action_name: str, action_input: str) -> str:
        """Execute un outil directement (pas via invoke() pour eviter les threads)."""
        tool = self._tool_map.get(action_name)
        if not tool:
            return f"[ERREUR] Outil '{action_name}' inconnu. Outils disponibles : {list(self._tool_map.keys())}"

        try:
            # Appel direct de la fonction Python sous-jacente
            result = tool.func(action_input)
            if result is None:
                return "[Observation] Aucun resultat retourne."
            return str(result)
        except Exception as exc:
            return f"[ERREUR] {exc}"

    def _direct_llm_fallback(self, prompt: str) -> Dict[str, Any]:
        """Fallback : appel LLM direct sans ReAct."""
        messages = [
            SystemMessage(content=self.role + "\n\n" + ANTI_HALLUCINATION_INSTRUCTIONS),
            HumanMessage(content=prompt),
        ]
        try:
            response = self.llm.invoke(messages)
            return {"output": response.content}
        except Exception as exc:
            return {"output": f"[ERREUR LLM] {exc}"}

    def run(self, input_text: str, **kwargs) -> Dict[str, Any]:
        """
        Execute l'agent en mode ReAct avec fallback DirectLLM.

        Args:
            input_text: La question / tache a accomplir

        Returns:
            Dict avec cle 'output' contenant la reponse finale
        """
        scratchpad = ""
        question = input_text

        for iteration in range(self.MAX_ITERATIONS):
            prompt = self._build_react_prompt(question, scratchpad)
            messages = [
                SystemMessage(content="Tu es un assistant qui raisonne etape par etape."),
                HumanMessage(content=prompt),
            ]

            try:
                response = self.llm.invoke(messages)
                text = response.content
            except Exception as exc:
                print(f"[{self.name}] Erreur LLM iteration {iteration}: {exc}")
                return self._direct_llm_fallback(input_text)

            parsed = self._parse_thought_action(text)

            if parsed is None:
                # Parsing rate : fallback si derniere iteration
                if iteration >= self.MAX_ITERATIONS - 1:
                    print(f"[{self.name}] Parsing ReAct echoue, fallback DirectLLM.")
                    return self._direct_llm_fallback(input_text)
                scratchpad += f"\nThought : Je dois reformuler ma reponse.\n"
                continue

            if parsed["type"] == "final":
                return {"output": parsed["answer"]}

            if parsed["type"] == "action":
                thought = parsed["thought"]
                action_name = parsed["action"]
                action_input = parsed["action_input"]

                print(f"[{self.name}] Thought: {thought[:80]}...")
                print(f"[{self.name}] Action: {action_name}({action_input[:60]}...)")

                observation = self._execute_tool(action_name, action_input)
                print(f"[{self.name}] Observation: {observation[:100]}...")

                scratchpad += (
                    f"\nThought : {thought}\n"
                    f"Action : {action_name}\n"
                    f"Action Input : {action_input}\n"
                    f"Observation : {observation}\n"
                )

        # Max iterations atteint : fallback
        print(f"[{self.name}] Max iterations atteint, fallback DirectLLM.")
        return self._direct_llm_fallback(input_text)


class DirectLLMAgent(ABC):
    """
    Agent qui fonctionne par invocation LLM directe SANS boucle ReAct.
    Ideal pour les agents qui n'ont pas besoin d'outils (analyse, redaction, validation).
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
                    print(f"[{self.name}] Rate limit detecte, retry dans {delay}s...")
                    time.sleep(delay)
                else:
                    print(f"[{self.name}] Erreur LLM apres {attempt + 1} tentatives: {exc}")
                    raise

        return {"output": "[ERREUR] Impossible d'obtenir une reponse du LLM apres plusieurs tentatives."}
