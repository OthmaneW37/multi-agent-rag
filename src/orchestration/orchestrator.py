"""Orchestrateur du systeme multi-agents."""

from typing import Dict, Any, List
from dataclasses import dataclass, field

from src.agents.collecteur import CollecteurAgent
from src.agents.analyste import AnalysteAgent
from src.agents.redacteur import RedacteurAgent
from src.agents.verificateur import VerificateurAgent


@dataclass
class WorkflowState:
    """Etat partage entre les agents lors du workflow."""
    sujet_recherche: str = ""
    passages_collectes: str = ""
    analyse: str = ""
    document: str = ""
    rapport_verification: str = ""
    iterations: int = 0
    max_iterations: int = 3
    status: str = "initialized"  # initialized, collecting, analyzing, writing, verifying, completed, failed
    history: List[Dict[str, Any]] = field(default_factory=list)


class Orchestrator:
    """
    Orchestrateur LangChain qui coordonne les 4 agents dans un flux de travail sequentiel.
    
    Flux de travail :
    1. Collecteur -> extraction des passages pertinents
    2. Analyste -> analyse des tendances et lacunes
    3. Redacteur -> redaction de la synthese structuree
    4. Verificateur -> validation et controle qualite
    5. (Optionnel) Boucle de correction si le document est rejete
    """

    def __init__(self):
        self.collecteur = CollecteurAgent()
        self.analyste = AnalysteAgent()
        self.redacteur = RedacteurAgent()
        self.verificateur = VerificateurAgent()

    def run(self, sujet_recherche: str) -> WorkflowState:
        """
        Execute le workflow complet d'orchestration.
        
        Args:
            sujet_recherche: Le sujet ou la question de recherche
        
        Returns:
            L'etat final du workflow avec tous les resultats
        """
        state = WorkflowState(sujet_recherche=sujet_recherche)
        print(f"\n{'='*60}")
        print(f"DEMARRAGE DU WORKFLOW MULTI-AGENTS")
        print(f"Sujet : {sujet_recherche}")
        print(f"{'='*60}\n")

        # Etape 1 : Collecte
        print("ETAPE 1/4 : COLLECTE DES PASSAGES...")
        state.status = "collecting"
        try:
            result_collecte = self.collecteur.collecter(sujet_recherche)
            state.passages_collectes = result_collecte.get("output", "")
            state.history.append({"agent": "Collecteur", "status": "success"})
            print(f"[OK] Collecte terminee ({len(state.passages_collectes)} caracteres extraits)\n")
        except Exception as e:
            state.status = "failed"
            state.history.append({"agent": "Collecteur", "status": "error", "error": str(e)})
            print(f"[ERREUR] Erreur lors de la collecte : {e}\n")
            return state

        # Etape 2 : Analyse
        print("ETAPE 2/4 : ANALYSE DES PASSAGES...")
        state.status = "analyzing"
        try:
            result_analyse = self.analyste.analyser(state.passages_collectes, sujet_recherche)
            state.analyse = result_analyse.get("output", "")
            state.history.append({"agent": "Analyste", "status": "success"})
            print(f"[OK] Analyse terminee ({len(state.analyse)} caracteres generes)\n")
        except Exception as e:
            state.status = "failed"
            state.history.append({"agent": "Analyste", "status": "error", "error": str(e)})
            print(f"[ERREUR] Erreur lors de l'analyse : {e}\n")
            return state

        # Etape 3-4 : Redaction et Verification (avec boucle de correction)
        while state.iterations < state.max_iterations:
            state.iterations += 1
            print(f"ETAPE 3/4 : REDACTION (iteration {state.iterations}/{state.max_iterations})...")
            state.status = "writing"
            try:
                result_redaction = self.redacteur.rediger(
                    state.analyse,
                    state.passages_collectes,
                    sujet_recherche
                )
                state.document = result_redaction.get("output", "")
                state.history.append({"agent": "Redacteur", "status": "success", "iteration": state.iterations})
                print(f"[OK] Redaction terminee ({len(state.document)} caracteres generes)\n")
            except Exception as e:
                state.status = "failed"
                state.history.append({"agent": "Redacteur", "status": "error", "error": str(e)})
                print(f"[ERREUR] Erreur lors de la redaction : {e}\n")
                return state

            # Etape 4 : Verification
            print(f"ETAPE 4/4 : VERIFICATION (iteration {state.iterations}/{state.max_iterations})...")
            state.status = "verifying"
            try:
                result_verif = self.verificateur.verifier(
                    state.document,
                    state.passages_collectes,
                    sujet_recherche
                )
                state.rapport_verification = result_verif.get("output", "")
                state.history.append({"agent": "Verificateur", "status": "success", "iteration": state.iterations})
                print(f"[OK] Verification terminee ({len(state.rapport_verification)} caracteres generes)\n")
            except Exception as e:
                state.status = "failed"
                state.history.append({"agent": "Verificateur", "status": "error", "error": str(e)})
                print(f"[ERREUR] Erreur lors de la verification : {e}\n")
                return state

            # Decision : valider ou iterer
            rapport_lower = state.rapport_verification.lower()
            if "valide" in rapport_lower or "valide" in rapport_lower:
                print("DOCUMENT VALIDE !")
                state.status = "completed"
                break
            elif state.iterations < state.max_iterations:
                print(f"[ATTENTION] Document a revoir. Lancement d'une correction (iteration {state.iterations + 1})...\n")
                # On garde le rapport de verification pour informer la prochaine redaction
                state.analyse = state.analyse + "\n\n--- RAPPORT DE VERIFICATION PRECEDENT (a prendre en compte) ---\n" + state.rapport_verification
            else:
                print("[ATTENTION] Nombre maximum d'iterations atteint. Document livre avec reserves.")
                state.status = "completed"
                break

        print(f"\n{'='*60}")
        print(f"WORKFLOW TERMINE - Statut : {state.status.upper()}")
        print(f"{'='*60}\n")
        return state

    def run_agent_only(self, agent_name: str, input_text: str) -> Dict[str, Any]:
        """
        Execute un agent individuellement (pour demonstration).
        
        Args:
            agent_name: Nom de l'agent ('collecteur', 'analyste', 'redacteur', 'verificateur')
            input_text: Texte d'entree pour l'agent
        
        Returns:
            Resultat de l'execution de l'agent
        """
        agents = {
            "collecteur": self.collecteur,
            "analyste": self.analyste,
            "redacteur": self.redacteur,
            "verificateur": self.verificateur,
        }

        agent = agents.get(agent_name.lower())
        if not agent:
            raise ValueError(f"Agent inconnu : {agent_name}. Agents disponibles : {list(agents.keys())}")

        print(f"\nExecution de l'agent : {agent.name}")
        print(f"{'-'*40}")
        return agent.run(input_text)
