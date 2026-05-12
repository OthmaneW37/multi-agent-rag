"""Tests unitaires de l'orchestrateur."""

from unittest.mock import MagicMock

import pytest

from src.orchestration.orchestrator import (
    WACOrchestrator,
    _verifier_qualite_scout,
)


class TestVerifierQualiteScout:
    """Tests du routage conditionnel."""

    def test_scout_ok(self):
        text = "## Section 1\n" + "Donnees\n" * 50 + "## Section 2\nPlus de donnees"
        ok, msg = _verifier_qualite_scout(text)
        assert ok is True
        assert msg == ""

    def test_scout_trop_court(self):
        ok, msg = _verifier_qualite_scout("court")
        assert ok is False
        assert "trop court" in msg

    def test_scout_aucun_resultat(self):
        ok, msg = _verifier_qualite_scout(
            "[Aucun resultat pertinent]\n" * 50
        )
        assert ok is False
        assert "aucune donnee" in msg

    def test_scout_long_non_structure_accepte(self):
        """Un texte long et substantiel est accepte meme sans ## (reformatage en amont)."""
        ok, msg = _verifier_qualite_scout("texte sans section structuree mais assez long " * 20)
        assert ok is True
        assert msg == ""

    def test_scout_court_non_structure_rejete(self):
        """Un texte court (200-400 chars) sans ## est rejete."""
        ok, msg = _verifier_qualite_scout("court texte sans rien " * 12)  # ~240 chars
        assert ok is False
        assert "pas de sections" in msg


class TestOrchestrateur:
    """Tests de l'orchestrateur avec mocks."""

    def test_pipeline_complet(self, monkeypatch):
        """L'orchestrateur doit executer les 4 etapes et produire un rapport."""
        orch = WACOrchestrator()

        # Mock des agents
        orch.scout = MagicMock()
        orch.scout.scout_match.return_value = {
            "output": "## Scout\n" + "Donnees WAC vs Raja\n" * 50
        }

        orch.modelisateur = MagicMock()
        orch.modelisateur.analyser.return_value = {
            "output": "## Modelisateur\nAnalyse\n"
        }

        orch.tacticien = MagicMock()
        orch.tacticien.formuler_strategie.return_value = {
            "output": "## Tacticien\nStrategie\n"
        }

        orch.validateur = MagicMock()
        orch.validateur.valider_et_compiler.return_value = {
            "output": "# Rapport\n## Resume\nTout va bien."
        }

        # Mock des fonctions de post-processing
        monkeypatch.setattr(
            "src.orchestration.orchestrator._extract_upcoming_match_date",
            lambda adv: "2025-09-01",
        )
        monkeypatch.setattr(
            "src.orchestration.orchestrator._extract_stadium",
            lambda adv: "Stade Mohammed V",
        )
        monkeypatch.setattr(
            "src.orchestration.orchestrator._extract_confrontations",
            lambda adv: "WAC 2-1 Raja",
        )
        monkeypatch.setattr(
            "src.orchestration.orchestrator._generate_xi_programmatic",
            lambda club: "- Gardien: Test",
        )

        result = orch.analyser_match("Raja CA", contexte="Test", verbose=False)

        assert "etapes" in result
        assert "scout" in result["etapes"]
        assert "modelisateur" in result["etapes"]
        assert "tacticien" in result["etapes"]
        assert "validateur" in result["etapes"]
        assert "rapport_final" in result
        assert "Rapport" in result["rapport_final"]
        assert "erreur" not in result

    def test_pipeline_arret_scout_insuffisant(self, monkeypatch):
        """L'orchestrateur doit s'arreter si le Scout n'a pas assez de donnees."""
        orch = WACOrchestrator()

        orch.scout = MagicMock()
        orch.scout.scout_match.return_value = {
            "output": "[Aucun resultat pertinent]\n" * 5
        }

        result = orch.analyser_match("Raja CA", verbose=False)

        assert "erreur" in result
        assert "pipeline s'est arrete" in result["rapport_final"]
        assert "modelisateur" not in result["etapes"]
