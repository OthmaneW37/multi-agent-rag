"""Backend API FastAPI pour WAC Sport Analytics.

Lancez avec :
    uvicorn api:app --reload --port 8000
"""

import asyncio
import json
import os
import sys
import traceback
from typing import Any, AsyncGenerator, Dict, List

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

# S'assurer que src/ est dans le PYTHONPATH
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv

load_dotenv()

from src.orchestration.orchestrator import WACOrchestrator
from src.rag.indexing import build_or_load_index
from src.rag.retrieval import query_index
from src.llm_utils import get_llm

app = FastAPI(
    title="WAC Sport Analytics API",
    description="API pour le systeme multi-agents RAG d'analyse sportive du Wydad AC",
    version="1.0.0",
)

# CORS pour permettre au frontend React de communiquer
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000", "http://localhost:5500", "http://127.0.0.1:5500"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Liste des adversaires (doit matcher ScoutAgent.ADVERSAIRE_SLUG_MAP)
ADVERSAIRES = [
    "Raja CA", "FAR Rabat", "FUS Rabat", "Ittihad Tanger",
    "Olympic Safi", "Hassania Agadir", "Maghreb Fes",
    "Kawkab Marrakech", "RSB Berkane", "DH El Jadida",
    "CODM Meknes", "Yacoub El Mansour", "UTS Rabat",
    "Olympique Dcheira", "CR Khemis Zemamra",
]


class AnalyseRequest(BaseModel):
    adversaire: str
    contexte: str = ""


class CompareRequest(BaseModel):
    question: str


class AnalyseResponse(BaseModel):
    adversaire: str
    contexte: str
    rapport_final: str
    etapes: Dict[str, Any]
    erreur: str = ""


class CompareResponse(BaseModel):
    question: str
    rag: str
    no_rag: str


@app.get("/")
def root():
    return {"message": "WAC Sport Analytics API", "docs": "/docs"}


@app.get("/adversaires")
def get_adversaires() -> List[str]:
    return ADVERSAIRES


@app.post("/analyse", response_model=AnalyseResponse)
async def analyse_match(req: AnalyseRequest):
    """Execute le pipeline complet d'analyse multi-agents."""
    try:
        orchestrator = WACOrchestrator()
        # Execution dans un thread separe pour ne pas blocher l'event loop
        results = await asyncio.to_thread(
            orchestrator.analyser_match,
            adversaire=req.adversaire,
            contexte=req.contexte,
            verbose=False,
        )
        return AnalyseResponse(
            adversaire=req.adversaire,
            contexte=req.contexte,
            rapport_final=results.get("rapport_final", ""),
            etapes=results.get("etapes", {}),
            erreur=results.get("erreur", ""),
        )
    except Exception as exc:
        return AnalyseResponse(
            adversaire=req.adversaire,
            contexte=req.contexte,
            rapport_final="",
            etapes={},
            erreur=f"{exc}\n\n{traceback.format_exc()}",
        )


async def analyse_stream_generator(req: AnalyseRequest) -> AsyncGenerator[str, None]:
    """Genere un stream NDJSON des evenements du pipeline multi-agents."""
    queue = asyncio.Queue()
    loop = asyncio.get_running_loop()

    def on_event(event_type: str, agent_id: str, data=None):
        asyncio.run_coroutine_threadsafe(
            queue.put({"event": event_type, "agent": agent_id, "data": data}),
            loop,
        )

    async def run_pipeline():
        try:
            orchestrator = WACOrchestrator()
            await asyncio.to_thread(
                orchestrator.analyser_match,
                adversaire=req.adversaire,
                contexte=req.contexte,
                verbose=False,
                on_event=on_event,
            )
        except Exception as exc:
            await queue.put({
                "event": "error",
                "agent": "orchestrateur",
                "data": {"message": str(exc), "traceback": traceback.format_exc()},
            })
        finally:
            await queue.put({"event": "done", "agent": "orchestrateur", "data": None})

    # Lancer le pipeline en background
    task = asyncio.create_task(run_pipeline())

    try:
        while True:
            event = await queue.get()
            yield json.dumps(event, ensure_ascii=False) + "\n"
            if event["event"] == "done":
                break
    finally:
        if not task.done():
            task.cancel()


@app.post("/analyse/stream")
async def analyse_match_stream(req: AnalyseRequest):
    """Execute le pipeline complet en streaming (NDJSON ligne par ligne)."""
    return StreamingResponse(
        analyse_stream_generator(req),
        media_type="application/x-ndjson",
    )


@app.post("/compare", response_model=CompareResponse)
async def compare_rag(req: CompareRequest):
    """Compare une reponse avec RAG vs sans RAG."""
    # RAG
    try:
        idx = await asyncio.to_thread(build_or_load_index)
        rag_text = await asyncio.to_thread(query_index, idx, req.question)
    except Exception as exc:
        rag_text = f"[ERREUR RAG] {exc}"

    # No-RAG
    try:
        llm = get_llm(temperature=0.3)
        prompt = (
            "Tu es un assistant football specialise dans la Botola Pro.\n"
            "Reponds a la question UNIQUEMENT avec tes connaissances internes.\n"
            f"Question : {req.question}\n\nReponse :"
        )
        no_rag_text = await asyncio.to_thread(lambda: llm.invoke(prompt).content)
    except Exception as exc:
        no_rag_text = f"[ERREUR LLM] {exc}"

    return CompareResponse(
        question=req.question,
        rag=rag_text,
        no_rag=no_rag_text,
    )


# ============================================================================
# ENDPOINTS DONNEES BRUTES (lecture directe CSV, pas de LLM)
# ============================================================================

import csv
from pathlib import Path

SCRAP_PATH = Path("ScrappingDataBotola/data")
TEAMS_PATH = SCRAP_PATH / "teams"

ADVERSAIRE_SLUG_MAP = {
    "Raja CA": "raja-club-athletic-de-casablanca-2536",
    "FAR Rabat": "as-forces-armees-royales-de-rabat-2532",
    "FUS Rabat": "fath-union-sport-de-rabat-2543",
    "Ittihad Tanger": "ittihad-riadhi-de-tanger-2529",
    "Olympic Safi": "olympique-club-de-safi-2534",
    "Hassania Agadir": "hassania-union-sport-dagadir-2531",
    "Maghreb Fes": "maghreb-as-de-fes-2547",
    "Kawkab Marrakech": "kawkab-athletique-club-de-marrakech-2537",
    "RSB Berkane": "renaissance-sportive-de-berkane-2541",
    "DH El Jadida": "difaa-hassani-del-jadida-2538",
    "CODM Meknes": "club-omnisports-de-meknes-2563",
    "Yacoub El Mansour": "us-yacoub-el-mansour-1322394",
    "UTS Rabat": "uts-rabat-17956",
    "Olympique Dcheira": "olympique-dcheira-2555",
    "CR Khemis Zemamra": "club-renaissance-khemis-zemamra-10556",
    "Wydad AC": "wydad-athletic-club-2530",
}


def _slug_for(name: str) -> str:
    return ADVERSAIRE_SLUG_MAP.get(name, "")


def _read_team_stats(slug: str) -> List[Dict[str, str]]:
    filepath = TEAMS_PATH / f"{slug}.csv"
    if not filepath.exists():
        return []
    rows = []
    with open(filepath, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


CLUB_ALIASES = {
    "Wydad AC": ["wydad casablanca", "wydad athletic club"],
    "Raja CA": ["raja casablanca", "raja club athletic"],
    "FAR Rabat": ["far rabat", "forces armees royales"],
    "FUS Rabat": ["fus rabat", "fath union sport"],
    "Ittihad Tanger": ["ittihad tanger", "al-ittihad"],
    "Olympic Safi": ["olympic safi", "olympique safi"],
    "Hassania Agadir": ["hassania agadir", "hus agadir"],
    "Maghreb Fes": ["maghreb fes", "mas fes", "maghreb fez"],
    "Kawkab Marrakech": ["kawkab marrakech", "kac marrakech"],
    "RSB Berkane": ["rsb berkane", "rs berkane", "renaissance sportive"],
    "DH El Jadida": ["difaa hassani", "dh el jadida", "difaâ el jadida"],
    "CODM Meknes": ["codm meknes", "club omnisports de meknes"],
    "Yacoub El Mansour": ["yacoub el mansour", "us yacoub"],
    "UTS Rabat": ["uts rabat"],
    "Olympique Dcheira": ["olympique dcheira", "olympique dcheïra"],
    "CR Khemis Zemamra": ["cr khemis zemamra", "club renaissance khemis"],
}


def _club_matches_query(club_raw: str, team_name: str) -> bool:
    raw_lower = club_raw.lower()
    aliases = CLUB_ALIASES.get(team_name, [team_name.lower()])
    return any(alias in raw_lower for alias in aliases)


def _read_players_for_team(team_name: str) -> List[Dict[str, Any]]:
    filepath = SCRAP_PATH / "player_stats.csv"
    if not filepath.exists():
        return []
    players = []
    with open(filepath, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            club_raw = row.get("club", "")
            if _club_matches_query(club_raw, team_name):
                players.append({
                    "name": row.get("player_name", ""),
                    "position": row.get("position", "").split("Nationality")[0].strip(),
                    "nationality": row.get("nationality", "").split("Kit")[0].strip(),
                    "matches": row.get("matches_played", ""),
                    "goals": row.get("goals_scored", ""),
                    "minutes": row.get("minutes", ""),
                    "rating": row.get("rating", ""),
                })
    return players


def _read_fixtures() -> List[Dict[str, str]]:
    filepath = SCRAP_PATH / "fixtures.csv"
    if not filepath.exists():
        return []
    rows = []
    with open(filepath, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def _find_matches_for_team(team_name: str) -> List[Dict[str, str]]:
    fixtures = _read_fixtures()
    matches = []
    aliases = [team_name]
    if team_name == "Wydad AC":
        aliases = ["Wydad Casablanca", "Wydad AC"]
    for fx in fixtures:
        home = fx.get("Home", "")
        away = fx.get("Away", "")
        score = fx.get("Score", "")
        status = fx.get("Status", "")
        date = fx.get("Date", "")
        if any(a.lower() in home.lower() or a.lower() in away.lower() for a in aliases):
            matches.append({
                "date": date,
                "home": home,
                "away": away,
                "score": score,
                "status": status,
            })
    return matches


@app.get("/stats/{club}")
def get_stats(club: str):
    """Retourne les statistiques collectives brutes d'un club depuis FootyStats."""
    slug = _slug_for(club)
    if not slug:
        return {"erreur": "Club inconnu", "club": club}
    stats = _read_team_stats(slug)
    return {"club": club, "slug": slug, "stats": stats}


@app.get("/squad/{club}")
def get_squad(club: str):
    """Retourne l'effectif d'un club depuis player_stats.csv."""
    players = _read_players_for_team(club)
    return {"club": club, "joueurs": players}


@app.get("/fixtures/{club}")
def get_fixtures(club: str):
    """Retourne les matchs (passés et à venir) d'un club."""
    matches = _find_matches_for_team(club)
    return {"club": club, "matchs": matches}
