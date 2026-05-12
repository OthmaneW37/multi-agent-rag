"""
WAC Sport Analytics - Interface Streamlit

Lancez avec :
    streamlit run app.py
"""

import os
import sys
import traceback

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import streamlit as st
from dotenv import load_dotenv

from src.orchestration.orchestrator import WACOrchestrator
from src.rag.indexing import build_or_load_index
from src.llm_utils import get_llm
from src.rag.retrieval import query_index

load_dotenv()

ADVERSAIRES = [
    "Raja CA", "FAR Rabat", "FUS Rabat", "Ittihad Tanger",
    "Olympic Safi", "Hassania Agadir", "Maghreb Fes",
    "Kawkab Marrakech", "RSB Berkane", "DH El Jadida",
    "CODM Meknes", "Yacoub El Mansour", "UTS Rabat",
    "Olympique Dcheira", "CR Khemis Zemamra",
]

st.set_page_config(
    page_title="WAC Sport Analytics",
    page_icon="⚽",
    layout="wide",
)

st.title("⚽ WAC Sport Analytics - Systeme Multi-Agents RAG")
st.markdown(
    "Analyse tactique et strategique pour le Wydad AC via un systeme "
    "multi-agents (Scout, Modelisateur, Tacticien, Validateur) avec RAG."
)

# Etat session
if "results" not in st.session_state:
    st.session_state.results = None
if "analyse_en_cours" not in st.session_state:
    st.session_state.analyse_en_cours = False
if "error_msg" not in st.session_state:
    st.session_state.error_msg = None


def run_analysis(adversaire: str, contexte: str):
    """Execute l'analyse et stocke le resultat dans session_state."""
    st.session_state.analyse_en_cours = True
    st.session_state.error_msg = None
    st.session_state.results = None
    try:
        orchestrator = WACOrchestrator()
        results = orchestrator.analyser_match(
            adversaire=adversaire,
            contexte=contexte,
            verbose=False,
        )
        st.session_state.results = results
    except Exception as exc:
        st.session_state.error_msg = f"{exc}\n\n{traceback.format_exc()}"
    finally:
        st.session_state.analyse_en_cours = False


# Onglets
tab_analyse, tab_demo = st.tabs(["Analyse de Match", "Demo RAG vs No-RAG"])

# ---------------------------------------------------------------------------
# ONGLET 1 : Analyse de Match
# ---------------------------------------------------------------------------
with tab_analyse:
    st.header("Analyse d'un match WAC")

    col1, col2 = st.columns([1, 2])

    with col1:
        adversaire = st.selectbox("Adversaire", ADVERSAIRES, key="adv_select")
        contexte = st.text_input(
            "Contexte (optionnel)",
            placeholder="Ex: Derby a domicile, 21e journee",
            key="ctx_input",
        )
        lancer = st.button(
            "Lancer l'analyse",
            type="primary",
            disabled=st.session_state.analyse_en_cours,
        )

    with col2:
        if lancer and not st.session_state.analyse_en_cours:
            run_analysis(adversaire, contexte)
            st.rerun()

        if st.session_state.analyse_en_cours:
            st.info("Analyse en cours... Cela peut prendre 1 a 2 minutes.")
            st.spinner("Orchestrateur en cours d'execution...")

        elif st.session_state.error_msg:
            st.error("Erreur pendant l'analyse")
            with st.expander("Voir le detail de l'erreur"):
                st.code(st.session_state.error_msg)

        elif st.session_state.results is not None:
            results = st.session_state.results
            if "erreur" in results:
                st.error(f"Pipeline arrete : {results['erreur']}")
            else:
                st.success("Analyse terminee !")

                rapport = results.get("rapport_final", "")
                st.markdown("---")
                st.subheader(f"Rapport Final - WAC vs {adversaire}")
                st.markdown(rapport)

                with st.expander("Voir les etapes intermediaires"):
                    etapes = results.get("etapes", {})
                    for nom, data in etapes.items():
                        output = data.get("output", str(data))
                        st.markdown(f"#### {nom.upper()}")
                        st.markdown(output[:2000] + ("\n..." if len(output) > 2000 else ""))

                st.download_button(
                    label="Telecharger le rapport (.md)",
                    data=rapport,
                    file_name=f"rapport_wac_{adversaire.replace(' ', '_').lower()}.md",
                    mime="text/markdown",
                )
        else:
            st.info("Selectionnez un adversaire et cliquez sur 'Lancer l'analyse'.")

# ---------------------------------------------------------------------------
# ONGLET 2 : Demo RAG vs No-RAG
# ---------------------------------------------------------------------------
with tab_demo:
    st.header("Demo comparative : RAG vs No-RAG")
    st.markdown(
        "Cette demo montre l'interet du RAG en comparant une reponse "
        "**avec** retrieval (donnees FootyStats) et **sans** retrieval (LLM seul)."
    )

    question = st.text_input(
        "Votre question",
        value="Quel est le bilan du WAC a domicile cette saison?",
    )
    comparer = st.button("Comparer", type="primary")

    if comparer:
        col_rag, col_no_rag = st.columns(2)

        with col_rag:
            st.subheader("Avec RAG")
            with st.spinner("Recherche dans la base..."):
                try:
                    idx = build_or_load_index()
                    rep_rag = query_index(idx, question)
                except Exception as exc:
                    rep_rag = f"[ERREUR] {exc}"
            st.markdown(rep_rag)

        with col_no_rag:
            st.subheader("Sans RAG (LLM seul)")
            with st.spinner("Appel LLM..."):
                try:
                    llm = get_llm(temperature=0.3)
                    prompt = (
                        "Tu es un assistant football specialise dans la Botola Pro.\n"
                        "Reponds a la question UNIQUEMENT avec tes connaissances internes.\n"
                        f"Question : {question}\n\nReponse :"
                    )
                    rep_no_rag = llm.invoke(prompt).content
                except Exception as exc:
                    rep_no_rag = f"[ERREUR] {exc}"
            st.markdown(rep_no_rag)

        st.markdown("---")
        st.info(
            "**Observation attendue** : La reponse RAG cite des sources concretes "
            "issues de FootyStats, tandis que la reponse sans RAG est generique "
            "ou incertaine car le LLM n'a pas acces aux donnees privees."
        )
