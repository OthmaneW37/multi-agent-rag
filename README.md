# WAC Sport Analytics - Systeme Multi-Agents RAG

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/)
[![LangChain](https://img.shields.io/badge/LangChain-0.1%2B-green)](https://python.langchain.com/)
[![LlamaIndex](https://img.shields.io/badge/LlamaIndex-0.9%2B-orange)](https://www.llamaindex.ai/)

> **Projet de Controle Continu - IA Distribuee & Systemes Multi-Agents**
>
> Analyse tactique et strategique pour le Wydad Athletic Club (WAC) via un systeme multi-agents avec RAG, alimente par des donnees reelles de FootyStats.

---

## Table des matieres

- [Contexte et Objectifs](#contexte-et-objectifs)
- [Architecture du Systeme](#architecture-du-systeme)
- [Stack Technologique](#stack-technologique)
- [Installation](#installation)
- [Utilisation](#utilisation)
- [Pipeline RAG](#pipeline-rag)
- [Integration FootyStats](#integration-footystats)
- [Les Agents](#les-agents)
- [Orchestration](#orchestration)
- [Structure du Projet](#structure-du-projet)
- [Auteurs](#auteurs)

---

## Contexte et Objectifs

Ce projet implemente un **Assistant d'Analyse Sportive** pour le Wydad AC (WAC) sous forme de systeme multi-agents intelligent. Il aide le staff technique a preparer les matchs en analysant un corpus de statistiques reelles scrappees depuis FootyStats.org pour la Botola Pro 2025/2026.

### Objectifs principaux

1. Concevoir un systeme multi-agents avec un cas d'usage concret (Analyse sportive - Botola Pro)
2. Implementer un pipeline RAG complet via **LlamaIndex** pour contextualiser les agents
3. Developper 4 agents IA avec **LangChain**, leurs outils et leur raisonnement
4. Integrer un **scraper FootyStats** pour obtenir des donnees reelles et fraiches
5. Demontrer la collaboration inter-agents via un orchestrateur intelligent

---

## Architecture du Systeme

```
+------------------------------------------------------------------+
|                    ORCHESTRATEUR LANGCHAIN                         |
|              (Gestion du flux sequentiel & Etat)                   |
+------------------------------------------------------------------+
                                |
        +-----------------------+-----------------------+
        |                       |                       |
        v                       v                       v
   +---------+          +-----------+          +-------------+
   |  SCOUT  |          |MODELISATEUR|          |  TACTICIEN  |
   |  (RAG)  |--------->| (Analyse) |--------->>| (Strategie) |
   +---------+          +-----------+          +-------------+
                                |                       |
                                +-----------------------+
                                                        |
                                                        v
                                               +----------------+
                                               |  VALIDATEUR    |
                                               | (Rapport QA)   |
                                               +----------------+
                                                        |
                                                        v
                                               +----------------+
                                               | RAPPORT FINAL  |
                                               |   (Markdown)   |
                                               +----------------+
```

### Flux de travail

1. **ScoutAgent** -> Parcourt la base RAG, extrait les donnees brutes (matchs, stats, joueurs)
2. **ModelisateurAgent** -> Analyse les donnees, identifie tendances et points faibles
3. **TacticienAgent** -> Formule des recommandations strategiques concretes
4. **ValidateurAgent** -> Verifie la coherence et compile le rapport final en Markdown

---

## Stack Technologique

| Technologie | Role | Version |
|---|---|---|
| **LangChain** | Agents, outils, orchestration, raisonnement ReAct | >=0.1.0 |
| **LlamaIndex** | Indexation, pipeline RAG, query engine | >=0.9.0 |
| **ChromaDB** | Vector store pour embeddings | >=0.4.0 |
| **Sentence-Transformers** | Modele d'embedding (all-MiniLM-L6-v2) | >=2.2.0 |
| **Ollama** | LLM local (mistral/llama3.2) - gratuit | >=0.1.0 |
| **Selenium / undetected-chromedriver** | Scraping FootyStats avec bypass Cloudflare | >=3.0 |
| **Python** | Langage de developpement | >=3.9 |

### Providers LLM supportes

- **Ollama** (par defaut) : Modeles open source locaux
- **OpenAI** : GPT-3.5-turbo, GPT-4
- **Anthropic** : Claude 3
- **Groq** : Inference rapide (Llama3, Mixtral)

---

## Installation

### Prerequis

- Python 3.9+
- **Groq API Key** (recommande, gratuit et rapide) : https://console.groq.com/
- Google Chrome (pour le scraping)
- Git

> **Note** : Ollama est supporte mais **deconseille** pour ce projet car trop lent (15-20 min par analyse) et sujet aux hallucinations. Utilisez **Groq** pour des resultats instantanes et fiables.

### Etapes

```bash
# 1. Cloner le repository
git clone <url-du-repo>
cd ProjetAgenticAI

# 2. Creer un environnement virtuel
python -m venv venv

# Activer l'environnement :
# Windows :
venv\Scripts\activate
# macOS/Linux :
source venv/bin/activate

# 3. Installer les dependances
pip install -r requirements.txt

# 4. Configuration des variables d'environnement
copy .env.example .env
# Editer .env avec votre cle Groq (recommande) ou autre provider

# 5. Pipeline complet (scraping -> ingestion -> analyse)
python -m src.main full-pipeline --adversaire "Raja CA"
```

### Configuration (.env) - RECOMMANDE : Groq

```env
# RECOMMANDE : Groq (gratuit, rapide, stable)
LLM_PROVIDER=groq
GROQ_API_KEY=gsk_votre_cle_groq
GROQ_MODEL=llama3-8b-8192

# Alternative : Ollama (local, lent, sujet aux hallucinations)
# LLM_PROVIDER=ollama
# OLLAMA_BASE_URL=http://localhost:11434
# OLLAMA_MODEL=mistral

# Alternative : OpenAI (payant)
# LLM_PROVIDER=openai
# OPENAI_API_KEY=sk-...
# OPENAI_MODEL=gpt-3.5-turbo

# Configuration RAG
VECTOR_STORE_PATH=./data/processed/vector_store
DATA_PATH=./data/raw
CHUNK_SIZE=512
CHUNK_OVERLAP=50
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
```

---

## Utilisation

### Pipeline complet ( recommande )

```bash
python -m src.main full-pipeline --adversaire "Raja CA" --contexte "Derby a domicile"
```

Cette commande execute automatiquement :
1. **Conversion** des CSV FootyStats en texte (`bridge`)
2. **Ingestion** des documents dans le vector store (`ingest --rebuild`)
3. **Analyse** multi-agents du match (`analyse`)

### Commandes individuelles

#### 1. Convertir les donnees FootyStats

```bash
python -m src.main bridge
```

Convertit les CSV produits par le scraper (`ScrappingDataBotola/data/`) en fichiers texte exploitables par le RAG dans `data/raw/`.

#### 2. Ingestion des documents

```bash
# Indexation incrementale
python -m src.main ingest

# Reconstruction complete de l'index
python -m src.main ingest --rebuild
```

#### 3. Analyse d'un match

```bash
python -m src.main analyse --adversaire "Raja CA" --contexte "Match a domicile, 21e journee"
```

Execute le pipeline complet : Scout -> Modelisateur -> Tacticien -> Validateur.

#### 4. Pipeline ingestion + analyse

```bash
python -m src.main pipeline --adversaire "FUS Rabat"
```

#### 5. Sauvegarder le rapport

```bash
python -m src.main full-pipeline --adversaire "RS Berkane" --output rapport_wac_berkane.md
```

---

## Pipeline RAG

### 1. Ingestion (`src/rag/ingestion.py`)

- **Formats supportes** : PDF, TXT, CSV, JSON
- Chargement des documents avec `SimpleDirectoryReader` de LlamaIndex
- Extraction des metadonnees (source, type)

### 2. Indexation (`src/rag/indexing.py`)

- **Chunking** : `SentenceSplitter` avec taille de chunk = 512 tokens et overlap = 50 tokens
- **Embeddings** : `sentence-transformers/all-MiniLM-L6-v2` (gratuit, 384 dimensions)
- **Vector Store** : ChromaDB persistant dans `./data/processed/vector_store`
- **Index** : `VectorStoreIndex` de LlamaIndex

### 3. Retrieval (`src/rag/retrieval.py`)

- **Recherche semantique** : `VectorIndexRetriever` avec top-k = 5 resultats
- **Post-traitement** : Filtre par similarite minimale (cutoff = 0.3)
- **Modes de reponse** : `compact` pour synthese concise

---

## Integration FootyStats

Le projet integre un **scraper FootyStats** (`ScrappingDataBotola/`) qui collecte automatiquement les statistiques de la Botola Pro.

### Donnees collectees

1. **Stats par equipe** (`data/teams/{club}.csv`) :
   - Wins, Draws, Losses
   - xG For / Against
   - Scored / Conceded per match
   - Clean Sheets %, Possession AVG
   - Shots, Fouls, Corners, etc.
   - Segmente par **Global / Domicile / Exterieur**

2. **Calendrier & Resultats** (`data/fixtures.csv`) :
   - Tous les matchs de la saison
   - Date, equipes, score, statut (FT / a venir)

3. **Classements Joueurs** (`data/players.csv`) :
   - Top Scorers, Assists
   - Most Clean Sheets
   - Apps, Goals, Cards

### Bridge RAG (`src/rag/footystats_bridge.py`)

Le module `footystats_bridge.py` convertit automatiquement les CSV en documents texte structures pour le pipeline RAG :

```
CSV FootyStats  ->  Texte structure  ->  Indexation LlamaIndex  ->  Agents
```

### Mise a jour des donnees

Pour rafraichir les donnees avec les derniers resultats :

1. **Lancer le scraper** (manuellement via Chrome) :
   ```bash
   cd ScrappingDataBotola/footystats_scraper
   python scraper.py
   ```

2. **Relancer le pipeline complet** :
   ```bash
   python -m src.main full-pipeline --adversaire "Raja CA"
   ```

---

## Les Agents

### Agent 1 : Scout (`src/agents/scout.py`) - **ReAct + Outils LangChain**

**Role** : Parcourir la base vectorielle indexee via RAG pour extraire les donnees brutes.

**Type** : Agent **ReAct** heritant de `BaseAgent`. Il utilise une boucle de raisonnement Thought/Action/Observation/Final Answer avec les outils LangChain definis dans `tools.py`.

**Implementation ReAct maison** (`src/agents/base.py`) :
- Boucle ReAct manuelle (pas d'`AgentExecutor` LangChain) pour eviter les callbacks/threads internes qui entrent en conflit avec `sentence-transformers`/PyTorch.
- Parsing tolerant du format ReAct avec fallback sur DirectLLM si le parsing echoue.
- Max 5 iterations pour eviter les boucles infinies.

**Outils LangChain** (`src/agents/tools.py`) :
- `search_match_reports` : Recherche semantique dans les stats FootyStats
- `query_sport_database` : Interrogation synthetique de la base
- `get_player_stats` : Statistiques individuelles des joueurs

**Raisonnement ReAct** :
1. **Thought** : "Je dois d'abord rechercher la forme recente du WAC"
2. **Action** : `search_match_reports("forme WAC stats collectives domicile")`
3. **Observation** : Resultats tronques de l'index RAG
4. **Thought** : "Maintenant je dois chercher la forme de l'adversaire"
5. ... (cycle repeté pour chaque dimension d'analyse)
6. **Final Answer** : Rapport de scouting structure

**Donnees complementaires** : Chargement programmatique des effectifs depuis `player_stats.csv` pour enrichir le prompt.

**Temperature** : 0.2 (precis et factuel)

**Sortie** : Donnees brutes structurees (forme, stats, confrontations, joueurs cles)

---

### Agent 2 : Modelisateur (`src/agents/modelisateur.py`)

**Role** : Traiter les donnees du Scout pour identifier tendances et points faibles.

**Type** : Agent **DirectLLM** avec **Chain-of-Thought explicite**.

**Prompt systeme** : Analyste statistique football, rigoureux et objectif. Le prompt inclut une section "RAISONNEMENT (Chain-of-Thought)" qui oblige le LLM a penser etape par etape avant chaque conclusion.

**Raisonnement CoT** :
1. Examiner les donnees brutes et les classer par categorie
2. Identifier les tendances et patterns dans chaque categorie
3. Croiser les categories pour trouver des correlations
4. Formuler des conclusions justifiees avec references aux chiffres
5. Verifier que chaque conclusion est soutenue par les donnees

**Temperature** : 0.2 (rigoureux et factuel)

**Sortie** : Analyse comparative, prediction qualitative, facteurs cles du match

---

### Agent 3 : Tacticien (`src/agents/tacticien.py`)

**Role** : Formuler des recommandations strategiques concretes pour le staff technique.

**Type** : Agent **DirectLLM** avec **Chain-of-Thought explicite**.

**Prompt systeme** : Strategiste football experimente, pragmatique et concret. Le prompt inclut une section "RAISONNEMENT (Chain-of-Thought)" avec 6 etapes de reflexion tactique.

**Raisonnement CoT** :
1. Analyser les forces/faiblesses identifiees par le Modelisateur
2. Examiner l'effectif disponible et les profils des joueurs
3. Choisir un schema tactique adapte
4. Definir les roles individuels en fonction des stats
5. Anticiper les reactions adverses
6. Verifier la realisabilite avec l'effectif disponible

**Temperature** : 0.3 (creatif mais base sur les donnees)

**Sortie** : Schema tactique, composition, consignes individuelles, plans alternatifs

---

### Agent 4 : Validateur (`src/agents/validateur.py`)

**Role** : Controle qualite et compilation du rapport final en Markdown.

**Type** : Agent **DirectLLM** avec **Chain-of-Thought explicite**.

**Prompt systeme** : Controleur qualite de rapports sportifs, exigeant et constructif. Le prompt inclut une section "RAISONNEMENT (Chain-of-Thought)" avec 6 etapes de validation.

**Raisonnement CoT** :
1. Lire les donnees brutes du Scout et identifier les faits etablis
2. Comparer chaque affirmation du Modelisateur avec les faits du Scout
3. Verifier la coherence entre la strategie du Tacticien et l'analyse
4. Detecter les hallucinations
5. Evaluer la qualite globale (ratio faits verifies / faits totaux)
6. Compiler le rapport final structure

**Temperature** : 0.1 (tres strict et factuel)

**Sortie** : Rapport final Markdown avec verdict [VALIDE / A REVOIR] et score /20

---

## Orchestration

### Orchestrateur (`src/orchestration/orchestrator.py`)

L'orchestrateur coordonne les agents via un **flux de travail sequentiel** :

```python
class WACOrchestrator:
    def analyser_match(self, adversaire, contexte):
        # 1. Scout -> Collecte des donnees
        scout_result = self.scout.scout_match(adversaire, contexte)
        
        # 2. Modelisateur -> Analyse
        analyse = self.modelisateur.analyser(scout_result, adversaire)
        
        # 3. Tacticien -> Strategie
        strategie = self.tacticien.formuler_strategie(analyse, adversaire)
        
        # 4. Validateur -> Rapport final
        rapport = self.validateur.valider_et_compiler(
            scout_result, analyse, strategie, adversaire
        )
        
        return rapport
```

### Mecanismes d'orchestration

1. **Flux sequentiel** : Scout -> Modelisateur -> Tacticien -> Validateur
2. **Etat partage** : Les resultats de chaque agent sont passes au suivant
3. **Routage conditionnel** : Verification automatique de la qualite du scouting avant de poursuivre. Si le Scout ne trouve pas assez de donnees, le pipeline s'arrete avec un message explicite.
4. **Gestion des erreurs** : Chaque etape est isolee pour eviter la cascade d'erreurs
5. **Traçabilite** : Tous les resultats intermediaires sont sauvegardes en JSON

---

## Tests Unitaires

Le projet inclut une suite de tests `pytest` dans le dossier `tests/` :

```bash
pytest tests/
```

### Couverture

| Fichier de test | Composant teste |
|---|---|
| `test_ingestion.py` | Chargement de documents (PDF, TXT, CSV) |
| `test_bridge.py` | Conversion CSV FootyStats -> texte |
| `test_retrieval.py` | Retrieval semantique avec mocks |
| `test_orchestrator.py` | Orchestrateur avec agents mockes + routage conditionnel |

---

## Demo RAG vs No-RAG

Un script de demonstration compare les reponses **avec** et **sans** RAG :

```bash
python demo_rag_vs_no_rag.py --question "Quel est le bilan du WAC a domicile?"
```

### Objectif

Montrer que le LLM seul (sans RAG) est incapable de repondre de maniere fiable sur des donnees privees (FootyStats), tandis que le RAG fournit des reponses contextualisees avec citations verifiables.

---

## Interface Web Immersive (HTML/CSS/JS pur)

Le projet dispose d'une interface **SPA immersive** en HTML/CSS/JS vanilla, avec un design premium inspiré des clubs européens de haut niveau. L'identité visuelle du Wydad Athletic Club (rouge #CC0000 et blanc sur fond sombre #111) est appliquée sur l'ensemble de l'interface.

### Architecture

```
Frontend SPA (HTML/CSS/JS)  <--HTTP-->  Backend FastAPI  <--->  Pipeline Python
```

### Lancer le backend

```bash
# Installer les dependances (inclut fastapi et uvicorn)
pip install -r requirements.txt

# Lancer l'API
python -m uvicorn api:app --reload --port 8000
```

L'API est disponible sur `http://localhost:8000` avec la documentation interactive Swagger sur `/docs`.

### Lancer le frontend

```bash
cd frontend
# Serveur HTTP simple (Python)
python -m http.server 5500
```

L'interface est disponible sur `http://localhost:5500`.

### Commandes disponibles dans le chat

| Commande | Description | Endpoint utilisé |
|---|---|---|
| `Analyse WAC vs [adversaire]` | Pipeline complet multi-agents (1-2 min) | `/analyse` |
| `Stats [club]` | Stats collectives brutes depuis FootyStats | `/stats/{club}` |
| `Squad [club]` ou `Composition [club]` | Effectif réel avec buts, notes, minutes | `/squad/{club}` |
| `Calendrier [club]` | Matchs passés et à venir | `/fixtures/{club}` |
| `RAG vs No-RAG` | Demo comparative avec/sans retrieval | `/compare` |

### Fonctionnalites immersives

- **Chat interactif** : conversation type messagerie avec bulles stylisees par agent (couleur + icone propres)
- **Sidebar agents** : liste des 4 agents avec etat en temps reel (Inactif / En cours / Termine) et barre de progression
- **Animations immersives** : a chaque agent actif, une banniere animee s'affiche avec une animation CSS unique :
  - **Scout** 🔍 : Radar/sonar avec ondes circulaires, sweep rotatif et particules qui apparaissent
  - **Modelisateur** 📊 : Barres de graphiques qui grandissent, sparkline qui se dessine
  - **Tacticien** 📋 : Terrain de football 4-3-3 qui se construit, joueurs qui se placent, fleches tactiques
  - **Validateur** ✓ : Cachet officiel qui s'imprime, coche verte, lignes de document qui se remplissent
- **Indicateur navbar** : badge en haut a droite indiquant l'agent actuellement en cours
- **Donnees reelles** : le frontend est connecte au backend FastAPI. Les commandes rapides lisent directement les CSV FootyStats (`ScrappingDataBotola/data/`). L'analyse complete execute le vrai pipeline multi-agents Python.

---

## Structure du Projet

```
ProjetAgenticAI/
├── data/
│   ├── raw/                          # Documents sources pour le RAG (TXT genere par bridge)
│   │   ├── wydad-athletic-club-2530_stats.txt
│   │   ├── raja-club-athletic-de-casablanca-2536_stats.txt
│   │   ├── fixtures_all.txt
│   │   ├── fixtures_wac.txt
│   │   └── players_leaderboard.txt
│   └── processed/                    # Vector store ChromaDB + rapports JSON
├── ScrappingDataBotola/              # Projet de scraping FootyStats
│   ├── footystats_scraper/
│   │   ├── scraper.py                # Scraper principal (Selenium)
│   │   ├── requirements.txt
│   │   └── README.md
│   └── data/                         # CSV bruts scrapes depuis FootyStats
│       ├── teams/                    # 16 fichiers CSV (1 par club)
│       ├── fixtures.csv              # Calendrier & resultats
│       └── players.csv               # Classements joueurs
├── src/
│   ├── __init__.py
│   ├── config.py                     # Configuration globale
│   ├── llm_utils.py                  # Factory LLM (Ollama, OpenAI, etc.)
│   ├── compat.py                     # Compatibilite LangChain
│   ├── main.py                       # Point d'entree CLI
│   ├── rag/
│   │   ├── __init__.py
│   │   ├── ingestion.py              # Chargement des documents
│   │   ├── indexing.py               # Indexation LlamaIndex + ChromaDB
│   │   ├── retrieval.py              # Retrieval semantique
│   │   └── footystats_bridge.py      # Conversion CSV -> Texte
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── base.py                   # Classe de base Agent ReAct + DirectLLM
│   │   ├── tools.py                  # Outils RAG pour les agents
│   │   ├── scout.py                  # Agent 1 : Scout (DirectLLM + outils programmatiques)
│   │   ├── modelisateur.py           # Agent 2 : Modelisateur
│   │   ├── tacticien.py              # Agent 3 : Tacticien
│   │   └── validateur.py             # Agent 4 : Validateur
│   └── orchestration/
│       ├── __init__.py
│       └── orchestrator.py           # Orchestrateur sequentiel + routage
├── tests/                            # Tests unitaires pytest
│   ├── __init__.py
│   ├── conftest.py
│   ├── test_ingestion.py
│   ├── test_bridge.py
│   ├── test_retrieval.py
│   └── test_orchestrator.py
├── api.py                            # Backend FastAPI (endpoints REST)
├── app.py                            # Interface Streamlit (deprecated)
├── demo_rag_vs_no_rag.py             # Script demo comparative RAG
├── frontend/                         # Interface immersive SPA (HTML/CSS/JS)
│   ├── index.html                    # Structure HTML complete
│   ├── css/
│   │   └── style.css                 # Styles + animations immersives
│   └── js/
│       └── app.js                    # Logique chat + demo multi-agents
├── .env.example                      # Template de configuration
├── requirements.txt                  # Dependances Python
└── README.md                         # Ce fichier
```

---

## Points forts du systeme

| Critere d'evaluation | Implementation |
|---|---|---|
| **Pipeline RAG** | Ingestion multi-formats, chunking configurable, embeddings HuggingFace, ChromaDB, retrieval avec filtre de similarite |
| **Agents IA** | 4 agents avec roles distincts : Scout (ReAct + outils LangChain), Modelisateur (DirectLLM + CoT), Tacticien (DirectLLM + CoT), Validateur (DirectLLM + CoT). Prompts optimises, temperatures adaptees. |
| **Orchestration** | Flux sequentiel, etat partage entre agents, gestion d'erreurs, traçabilite complete |
| **Donnees reelles** | Integration FootyStats avec scraping automatique + bridge CSV->Texte |
| **Cas d'usage** | Assistant d'analyse sportive concret et pertinent pour le WAC |
| **Interface** | SPA immersive HTML/CSS/JS avec animations par agent (radar, terrain, data viz, validation) + backend FastAPI |
| **Qualite du code** | Architecture modulaire, typage, docstrings, separation des responsabilites, configuration centralisee |

---

## Ameliorations possibles

- [x] Ajouter une interface web immersive (HTML/CSS/JS + animations agents) pour la demonstration
- [ ] Implementer le parallelisme pour le Scout (recherches simultanees)
- [ ] Ajouter un agent Routeur pour classifier les types de requetes
- [ ] Automatiser le scraping via CI/CD (cron job hebdomadaire)
- [x] Support de l'indexation incrementale (ajout de documents sans re-indexation totale)
- [x] Ajout de tests unitaires avec pytest
- [x] Demo comparative RAG vs No-RAG
- [x] Routage conditionnel dans l'orchestrateur

---

## Problemes connus et solutions

### 1. Hallucinations (LLM qui invente des donnees)

**Symptome** : L'agent parle de Ligue 1, de joueurs inexistants, ou de stats inventees.

**Causes** :
- LLM local (Ollama/Mistral) instable avec le format ReAct
- Prompts pas assez restrictifs
- Seuils de similarite RAG mal ajustes

**Solutions appliquees** :
- ✅ Prompts anti-hallucination injectes dans TOUS les agents
- ✅ Agent Scout : **ReAct maison** (`BaseAgent`) avec outils LangChain (`search_match_reports`, `query_sport_database`, `get_player_stats`). La boucle ReAct est implementee manuellement (pas d'`AgentExecutor`) pour eviter les callbacks/threads qui font crasher l'application. Fallback DirectLLM automatique si le parsing ReAct echoue.
- ✅ Agents Modelisateur/Tacticien/Validateur : **Chain-of-Thought explicite** dans les prompts (6 etapes de raisonnement documentees). Appels LLM directs sans outils.
- ✅ Seuils de similarite ajustes pour le modele all-MiniLM-L6-v2
- ✅ `query_index` et `_format_results` tronquent les textes a 300 caracteres pour limiter la consommation de tokens
- ✅ Recommandation : utiliser **Groq** (Llama3-8b) au lieu d'Ollama

### 2. Lenteur extreme / Erreur 413 (trop de tokens)

**Symptome** : L'analyse prend plus de 15 minutes, timeout, ou erreur Groq `Request too large for model` (limite 6000 TPM).

**Causes** :
- Ollama + Mistral = 4 appels LLM sequentiels, chacun tres lent
- Le ReAct textuel accumule les observations dans la scratchpad (5 resultats x 1000 tokens = 5000 tokens)
- `query_index` appelait le LLM via LlamaIndex (refine mode)

**Solutions appliquees** :
- ✅ Scout : ReAct avec max 5 iterations et fallback DirectLLM. Les sources RAG sont tronquees a 300 caracteres pour limiter la taille de la scratchpad.
- ✅ Modelisateur/Tacticien/Validateur : appels directs sans boucle ReAct (CoT dans le prompt uniquement)
- ✅ `query_index` et retrieval tronquent les sources a 300 caracteres
- ✅ `retrieve_context` top_k reduit a 3 par defaut
- ✅ Recommandation : utiliser **Groq** (reponse en 5-10 secondes)

### 3. FootyStats ne scrappe pas

**Symptome** : Le scraper ne recupere pas les donnees.

**Cause** : Cloudflare bloque le navigateur.

**Solution** : Le scraper utilise `undetected-chromedriver`. Si ca echoue :
1. Verifiez votre version Chrome (`chrome://version/`)
2. Modifiez `CHROME_VERSION` dans `scraper.py`
3. Executez le scraper manuellement et cliquez sur le Captcha si demande

### 4. Crash silencieux de Streamlit (deprecated)

**Symptome** : L'interface Streamlit se ferme immediatement sans message d'erreur lors du clic sur "Lancer l'analyse".

**Cause** : Les outils LangChain (`@tool`) crees avec `StructuredTool` ou `tool.invoke()` initialisent des callbacks et threads internes qui entrent en conflit avec l'execution multi-threadee de Streamlit et les bibliotheques `sentence-transformers`/PyTorch.

**Solution appliquee** :
- ✅ Remplacement de Streamlit par une interface **React + FastAPI** plus robuste et moderne.
- ✅ Suppression complete des outils LangChain dans le ScoutAgent. Remplacement par des appels directs aux fonctions `retrieve_context()` et `query_index()` de `src/rag/retrieval.py`.

---

## Auteurs

**Binome** : Othmane Moussawi & Ahmed Rayane Ramzi

**Module** : IA Distribuee & Systemes Multi-Agents

**Annee Universitaire** : 2025-2026

---

## Ressources

- [LangChain Documentation](https://python.langchain.com/docs/)
- [LlamaIndex Documentation](https://docs.llamaindex.ai/)
- [Ollama](https://ollama.ai/)
- [ChromaDB](https://www.trychroma.com/)
- [FootyStats](https://footystats.org/)

---

*Projet realise dans le cadre du controle continu du module IA Distribuee & Systemes Multi-Agents.*
