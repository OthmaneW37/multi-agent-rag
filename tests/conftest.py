"""Configuration pytest pour le projet WAC Sport Analytics."""

import sys
from pathlib import Path

# Ajoute la racine du projet au PYTHONPATH pour les imports `from src...`
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
