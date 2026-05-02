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

### Agent 1 : Scout (`src/agents/scout.py`)

**Role** : Parcourir la base vectorielle indexee via RAG pour extraire les donnees brutes.

**Prompt systeme** : Expert en scouting football, specialise dans la Botola Pro et le WAC.

**Outils** :
- `search_match_reports` : Recherche semantique dans les rapports de matchs
- `query_sport_database` : Interrogation synthetique de la base de connaissances
- `get_player_stats` : Statistiques individuelles des joueurs

**Temperature** : 0.3 (precis et factuel)

**Sortie** : Donnees brutes structurees (forme, stats, confrontations, joueurs cles)

---

### Agent 2 : Modelisateur (`src/agents/modelisateur.py`)

**Role** : Traiter les donnees du Scout pour identifier tendances et points faibles.

**Prompt systeme** : Analyste statistique football, rigoureux et objectif.

**Outils** :
- `search_match_reports` : Recherches complementaires pour valider les analyses

**Temperature** : 0.4 (equilibre entre creativite et rigueur)

**Sortie** : Analyse comparative, prediction qualitative, facteurs cles du match

---

### Agent 3 : Tacticien (`src/agents/tacticien.py`)

**Role** : Formuler des recommandations strategiques concretes pour le staff technique.

**Prompt systeme** : Strategiste football experimente, pragmatique et concret.

**Outils** :
- `search_match_reports` : Verification des donnees tactiques

**Temperature** : 0.5 (creatif mais base sur les donnees)

**Sortie** : Schema tactique, composition, consignes individuelles, plans alternatifs

---

### Agent 4 : Validateur (`src/agents/validateur.py`)

**Role** : Controle qualite et compilation du rapport final en Markdown.

**Prompt systeme** : Controleur qualite de rapports sportifs, exigeant et constructif.

**Outils** :
- `search_match_reports` : Verification des citations contre les sources

**Temperature** : 0.2 (tres strict et factuel)

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
3. **Gestion des erreurs** : Chaque etape est isolee pour eviter la cascade d'erreurs
4. **Traçabilite** : Tous les resultats intermediaires sont sauvegardes en JSON

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
│   │   ├── base.py                   # Classe de base Agent ReAct
│   │   ├── tools.py                  # Outils RAG pour les agents
│   │   ├── scout.py                  # Agent 1 : Scout
│   │   ├── modelisateur.py           # Agent 2 : Modelisateur
│   │   ├── tacticien.py              # Agent 3 : Tacticien
│   │   └── validateur.py             # Agent 4 : Validateur
│   └── orchestration/
│       ├── __init__.py
│       └── orchestrator.py           # Orchestrateur sequentiel
├── .env.example                      # Template de configuration
├── requirements.txt                  # Dependances Python
└── README.md                         # Ce fichier
```

---

## Points forts du systeme

| Critere d'evaluation | Implementation |
|---|---|---|
| **Pipeline RAG** | Ingestion multi-formats, chunking configurable, embeddings HuggingFace, ChromaDB, retrieval avec filtre de similarite |
| **Agents LangChain** | 4 agents ReAct avec roles distincts, outils specifiques, prompts optimises, temperatures adaptees |
| **Orchestration** | Flux sequentiel, etat partage entre agents, gestion d'erreurs, traçabilite complete |
| **Donnees reelles** | Integration FootyStats avec scraping automatique + bridge CSV->Texte |
| **Cas d'usage** | Assistant d'analyse sportive concret et pertinent pour le WAC |
| **Qualite du code** | Architecture modulaire, typage, docstrings, separation des responsabilites, configuration centralisee |

---

## Ameliorations possibles

- [ ] Ajouter une interface web (Streamlit/Gradio) pour la demonstration
- [ ] Implementer le parallelisme pour le Scout (recherches simultanees)
- [ ] Ajouter un agent Routeur pour classifier les types de requetes
- [ ] Automatiser le scraping via CI/CD (cron job hebdomadaire)
- [ ] Support de l'indexation incrementale (ajout de documents sans re-indexation totale)
- [ ] Ajout de tests unitaires avec pytest

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
- ✅ Agent Scout : RAG direct en code (pas de boucle ReAct)
- ✅ Agents Modelisateur/Tacticien/Validateur : appels LLM directs sans outils
- ✅ Seuils de similarite ajustes pour le modele all-MiniLM-L6-v2
- ✅ `query_index` ne fait plus appel au LLM (evite timeouts + hallucinations)
- ✅ Recommandation : utiliser **Groq** (Llama3-8b) au lieu d'Ollama

### 2. Lenteur extreme (15-20 min par analyse)

**Symptome** : L'analyse prend plus de 15 minutes ou timeout.

**Causes** :
- Ollama + Mistral = 4 appels LLM sequentiels, chacun tres lent
- `query_index` appelait le LLM via LlamaIndex (refine mode)

**Solutions appliquees** :
- ✅ Scout : un seul appel LLM apres RAG direct (au lieu de 5-10 en ReAct)
- ✅ Modelisateur/Tacticien/Validateur : appels directs sans boucle ReAct
- ✅ `query_index` retourne les sources sans synthese LLM
- ✅ Recommandation : utiliser **Groq** (reponse en 5-10 secondes)

### 3. FootyStats ne scrappe pas

**Symptome** : Le scraper ne recupere pas les donnees.

**Cause** : Cloudflare bloque le navigateur.

**Solution** : Le scraper utilise `undetected-chromedriver`. Si ca echoue :
1. Verifiez votre version Chrome (`chrome://version/`)
2. Modifiez `CHROME_VERSION` dans `scraper.py`
3. Executez le scraper manuellement et cliquez sur le Captcha si demande

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
