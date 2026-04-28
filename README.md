# 🤖 Système Multi-Agents pour Recherche Académique avec RAG

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/)
[![LangChain](https://img.shields.io/badge/LangChain-0.1%2B-green)](https://python.langchain.com/)
[![LlamaIndex](https://img.shields.io/badge/LlamaIndex-0.9%2B-orange)](https://www.llamaindex.ai/)

> **Projet de Contrôle Continu – IA Distribuée & Systèmes Multi-Agents**
> 
> Construction d'un Système Multi-Agents avec RAG & Orchestration LangChain

---

## 📋 Table des matières

- [Contexte et Objectifs](#contexte-et-objectifs)
- [Architecture du Système](#architecture-du-système)
- [Stack Technologique](#stack-technologique)
- [Installation](#installation)
- [Utilisation](#utilisation)
- [Pipeline RAG](#pipeline-rag)
- [Les Agents](#les-agents)
- [Orchestration](#orchestration)
- [Démonstration](#démonstration)
- [Structure du Projet](#structure-du-projet)
- [Auteurs](#auteurs)

---

## Contexte et Objectifs

Ce projet implémente un **Assistant de Recherche Académique** sous forme de système multi-agents intelligent. Il aide un chercheur à analyser un corpus de publications scientifiques pour produire un état de l'art structuré.

### Objectifs principaux

1. ✅ Concevoir un système multi-agents avec un cas d'usage concret (Recherche Académique)
2. ✅ Implémenter un pipeline RAG complet via **LlamaIndex** pour contextualiser les agents
3. ✅ Développer 4 agents IA avec **LangChain**, leurs outils et leur raisonnement
4. ✅ Démontrer la collaboration inter-agents via un orchestrateur intelligent

---

## Architecture du Système

```
┌─────────────────────────────────────────────────────────────────┐
│                    ORCHESTRATEUR LANGCHAIN                       │
│              (Gestion de l'état partagé & Flux)                  │
└─────────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        ▼                     ▼                     ▼
   ┌─────────┐         ┌──────────┐          ┌───────────┐
   │COLLECTEUR│         │ ANALYSTE │          │ RÉDACTEUR │
   │  (RAG)   │         │(Tendances│          │(Synthèse) │
   └────┬─────┘         └────┬─────┘          └─────┬─────┘
        │                    │                      │
        └────────────────────┼──────────────────────┘
                             ▼
                    ┌────────────────┐
                    │  VÉRIFICATEUR  │
                    │  (Contrôle QA) │
                    └────────────────┘
```

### Flux de travail

1. **Collecteur** → Parcourt le corpus indexé via RAG, extrait les passages pertinents
2. **Analyste** → Identifie les tendances, compare les approches, détecte les lacunes
3. **Rédacteur** → Génère la synthèse structurée avec citations
4. **Vérificateur** → Vérifie la cohérence, les citations et la qualité
5. **Boucle de correction** → Si le document est rejeté, retour à l'étape 3 (max 3 itérations)

---

## Stack Technologique

| Technologie | Rôle | Version |
|---|---|---|
| **LangChain** | Développement des agents, outils, orchestration et chaînes de raisonnement | >=0.1.0 |
| **LlamaIndex** | Indexation des données, pipeline RAG, query engine | >=0.9.0 |
| **ChromaDB** | Vector store pour le stockage des embeddings | >=0.4.0 |
| **Sentence-Transformers** | Modèle d'embedding (all-MiniLM-L6-v2) | >=2.2.0 |
| **Ollama** | LLM local (llama3.2) – gratuit et local | >=0.1.0 |
| **Python** | Langage de développement | >=3.9 |

### Providers LLM supportés

- **Ollama** (par défaut) : Modèles open source locaux
- **OpenAI** : GPT-3.5-turbo, GPT-4
- **Anthropic** : Claude 3
- **Groq** : Inférence rapide (Llama3, Mixtral)

---

## Installation

### Prérequis

- Python 3.9+
- [Ollama](https://ollama.ai/) installé (pour l'utilisation locale)
- Git

### Étapes

```bash
# 1. Cloner le repository
git clone <url-du-repo>
cd projet_multi_agents

# 2. Créer un environnement virtuel
python -m venv venv

# Activer l'environnement :
# Windows :
venv\Scripts\activate
# macOS/Linux :
source venv/bin/activate

# 3. Installer les dépendances
pip install -r requirements.txt

# 4. Configuration des variables d'environnement
cp .env.example .env
# Éditer .env selon votre configuration (voir section Configuration)

# 5. Télécharger le modèle Ollama (si utilisation locale)
ollama pull llama3.2

# 6. Vérifier l'installation
python -m src.main --setup
```

### Configuration (.env)

```env
# Configuration LLM
LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.2

# Alternative OpenAI
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

### 1. Démonstration complète (Workflow multi-agents)

```bash
python -m src.main --demo
```

Lance le workflow complet : Collecteur → Analyste → Rédacteur → Vérificateur.

Avec un sujet spécifique :
```bash
python -m src.main --demo --sujet "L'impact des transformers sur le NLP"
```

### 2. Tester un agent individuellement

```bash
python -m src.main --agent
```

Permet de tester chacun des 4 agents séparément pour vérifier leur fonctionnement.

### 3. Comparer avec/sans RAG

```bash
python -m src.main --compare
```

Montre la différence entre une réponse utilisant le RAG (données privées indexées) et une réponse du LLM seul.

### 4. Construire/recharger l'index

```bash
python -m src.main --setup
```

Indexe les documents présents dans `data/raw/` dans le vector store ChromaDB.

---

## Pipeline RAG

### 1. Ingestion (`src/rag/ingestion.py`)

- **Formats supportés** : PDF, TXT, CSV, JSON
- Chargement des documents avec les readers LlamaIndex
- Extraction des métadonnées (source, type)

### 2. Indexation (`src/rag/indexing.py`)

- **Chunking** : `SentenceSplitter` avec taille de chunk = 512 tokens et overlap = 50 tokens
- **Embeddings** : `sentence-transformers/all-MiniLM-L6-v2` (gratuit, performant, 384 dimensions)
- **Vector Store** : ChromaDB persistant dans `./data/processed/vector_store`
- **Index** : `VectorStoreIndex` de LlamaIndex

### 3. Retrieval (`src/rag/retrieval.py`)

- **Recherche sémantique** : `VectorIndexRetriever` avec top-k = 5 résultats
- **Post-traitement** : Filtre par similarité minimale (cutoff = 0.7)
- **Modes de réponse** : `compact` pour synthèse concise

### Données privées de test

Le dossier `data/raw/` contient 5 articles scientifiques simulés couvrant :
- L'architecture Transformer (Vaswani et al., 2017)
- BERT et le pré-entraînement bidirectionnel (Devlin et al., 2019)
- GPT-3 et le few-shot learning (Brown et al., 2020)
- Chain-of-Thought prompting (Wei et al., 2022)
- Retrieval-Augmented Generation (Lewis et al., 2020)

---

## Les Agents

### 🤖 Agent 1 : Collecteur (`src/agents/collecteur.py`)

**Rôle** : Parcourir le corpus indexé via RAG et extraire les passages pertinents.

**Prompt système** : Spécialiste de la recherche documentaire académique, méthodique et exhaustif.

**Outils** :
- `search_documents` : Recherche sémantique dans le corpus
- `query_knowledge_base` : Interrogation synthétique de la base de connaissances

**Température** : 0.3 (précis et déterministe)

---

### 🔍 Agent 2 : Analyste (`src/agents/analyste.py`)

**Rôle** : Identifier les tendances, comparer les approches, détecter les lacunes.

**Prompt système** : Analyste critique de littérature scientifique, rigoureux et objectif.

**Outils** :
- `search_documents` : Recherches complémentaires pour valider les analyses

**Température** : 0.4 (équilibré entre créativité et rigueur)

---

### 📝 Agent 3 : Rédacteur (`src/agents/redacteur.py`)

**Rôle** : Générer une synthèse structurée avec citations et références.

**Prompt système** : Rédacteur académique spécialisé en états de l'art, clair et rigoureux.

**Outils** :
- `search_documents` : Vérification des références pendant la rédaction

**Température** : 0.5 (style académique fluide)

**Format de sortie** :
- Titre, Résumé, Introduction
- Méthodologie, Synthèse thématique
- Discussion, Conclusion et perspectives
- Références bibliographiques

---

### ✅ Agent 4 : Vérificateur (`src/agents/verificateur.py`)

**Rôle** : Vérifier la cohérence, l'exactitude des citations et la qualité globale.

**Prompt système** : Contrôleur qualité de documents académiques, exigeant et constructif.

**Outils** :
- `search_documents` : Vérification des citations contre les sources originales

**Température** : 0.2 (très strict et factuel)

**Sortie** : Rapport de validation avec verdict (VALIDÉ / À REVOIR / REJETÉ), score /20 et suggestions de correction.

---

## Orchestration

### Orchestrateur (`src/orchestration/orchestrator.py`)

L'orchestrateur coordonne les agents via un **flux de travail séquentiel avec gestion d'état partagé** :

```python
class WorkflowState:
    sujet_recherche: str      # Thème de l'état de l'art
    passages_collectes: str   # Résultats du Collecteur
    analyse: str              # Résultats de l'Analyste
    document: str             # Document du Rédacteur
    rapport_verification: str # Rapport du Vérificateur
    iterations: int           # Compteur de corrections
    max_iterations: int = 3   # Limite de boucles
    status: str               # État du workflow
```

### Mécanismes d'orchestration

1. **Flux séquentiel** : Collecteur → Analyste → Rédacteur → Vérificateur
2. **État partagé** : Dictionnaire passé implicitement via les entrées textuelles entre agents
3. **Routage conditionnel** : Si le Vérificateur émet un verdict "À REVOIR", retour au Rédacteur avec le rapport de vérification
4. **Gestion des erreurs** : Chaque étape est isolée dans un try/except pour éviter la cascade d'erreurs
5. **Traçabilité** : Historique complet des exécutions dans `state.history`

### Diagramme de séquence

```
Utilisateur → Orchestrateur : sujet_recherche
Orchestrateur → Collecteur : collecter(sujet)
Collecteur → RAG : search_documents()
RAG → ChromaDB : requête vectorielle
ChromaDB → Collecteur : passages pertinents
Collecteur → Orchestrateur : passages_collectes

Orchestrateur → Analyste : analyser(passages)
Analyste → RAG : recherches complémentaires
Analyste → Orchestrateur : analyse

Orchestrateur → Rédacteur : rédiger(analyse, passages)
Rédacteur → Orchestrateur : document

Orchestrateur → Vérificateur : vérifier(document, passages)
Vérificateur → RAG : vérification citations
Vérificateur → Orchestrateur : rapport

[si "À REVOIR"] → Orchestrateur → Rédacteur : correction
[si "VALIDÉ"] → Orchestrateur → Utilisateur : document final
```

---

## Démonstration

### Scénario complet

**Entrée** : *"L'évolution des architectures Transformer et leur impact sur la recherche en NLP"*

**Étape 1 - Collecte** :
- Recherche dans le corpus des articles sur transformers, BERT, GPT-3
- Extraction de ~15 passages pertinents avec sources

**Étape 2 - Analyse** :
- Tendance 1 : Transition des RNN aux architectures attention-based
- Tendance 2 : Émergence du pré-entraînement à grande échelle
- Lacune : Manque d'études sur l'efficacité énergétique
- Comparaison : BERT (bidirectionnel) vs GPT (autorégressif)

**Étape 3 - Rédaction** :
- Génération d'un état de l'art structuré de ~2000 mots
- Citations intégrées : (Vaswani et al., 2017), (Devlin et al., 2019)...

**Étape 4 - Vérification** :
- Vérification des citations contre les sources originales
- Score de qualité : 17/20
- Verdict : VALIDÉ avec suggestions mineures

**Sortie finale** : Document académique complet et vérifiable.

---

## Structure du Projet

```
projet_multi_agents/
├── data/
│   ├── raw/                          # Documents sources (PDF, TXT, CSV, JSON)
│   │   ├── article_1_transformers.txt
│   │   ├── article_2_bert.txt
│   │   ├── article_3_gpt3.txt
│   │   ├── article_4_chain_of_thought.txt
│   │   └── article_5_rag.txt
│   └── processed/                    # Vector store ChromaDB persistant
├── src/
│   ├── __init__.py
│   ├── config.py                     # Configuration globale
│   ├── llm_utils.py                  # Factory LLM (Ollama, OpenAI, Anthropic, Groq)
│   ├── main.py                       # Point d'entrée CLI
│   ├── rag/
│   │   ├── __init__.py
│   │   ├── ingestion.py              # Chargement des documents
│   │   ├── indexing.py               # Indexation LlamaIndex + ChromaDB
│   │   └── retrieval.py              # Retrieval sémantique
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── base.py                   # Classe de base Agent ReAct
│   │   ├── tools.py                  # Outils partagés (search_documents, etc.)
│   │   ├── collecteur.py             # Agent 1 : Collecteur
│   │   ├── analyste.py               # Agent 2 : Analyste
│   │   ├── redacteur.py              # Agent 3 : Rédacteur
│   │   └── verificateur.py           # Agent 4 : Vérificateur
│   └── orchestration/
│       ├── __init__.py
│       └── orchestrator.py           # Orchestrateur avec état partagé
├── tests/                            # Tests unitaires (optionnel)
├── .env.example                      # Template de configuration
├── requirements.txt                  # Dépendances Python
└── README.md                         # Ce fichier
```

---

## Points forts du système

| Critère d'évaluation | Implémentation |
|---|---|
| **Pipeline RAG** | Ingestion multi-formats, chunking configurable, embeddings HuggingFace, ChromaDB, retrieval avec filtre de similarité |
| **Agents LangChain** | 4 agents ReAct avec rôles distincts, outils spécifiques, prompts optimisés, températures adaptées |
| **Orchestration** | Flux séquentiel, état partagé (`WorkflowState`), boucle de correction, gestion d'erreurs, traçabilité |
| **Cas d'usage** | Assistant de recherche académique concret et pertinent |
| **Qualité du code** | Architecture modulaire, typage, docstrings, séparation des responsabilités, configuration centralisée |

---

## Améliorations possibles

- [ ] Ajouter une interface web (Streamlit/Gradio) pour la démonstration
- [ ] Implémenter le parallélisme pour le Collecteur (recherches simultanées)
- [ ] Ajouter un agent Routeur pour classifier les types de requêtes
- [ ] Intégrer des métriques d'évaluation automatique (ROUGE, BLEU)
- [ ] Support de l'indexation incrémentale (ajout de documents sans ré-indexation totale)
- [ ] Ajout de tests unitaires avec pytest

---

## Auteurs

**Binôme** : [Nom étudiant 1] & [Nom étudiant 2]

**Module** : IA Distribuée & Systèmes Multi-Agents

**Année Universitaire** : 2025–2026

---

## Ressources

- [LangChain Documentation](https://python.langchain.com/docs/)
- [LlamaIndex Documentation](https://docs.llamaindex.ai/)
- [Ollama](https://ollama.ai/)
- [ChromaDB](https://www.trychroma.com/)

---

*Projet réalisé dans le cadre du contrôle continu du module IA Distribuée & Systèmes Multi-Agents.*
