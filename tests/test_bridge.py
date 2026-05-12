"""Tests unitaires du bridge FootyStats -> Texte."""

import tempfile
from pathlib import Path

import pandas as pd
import pytest

from src.rag.footystats_bridge import (
    _slug_to_name,
    convert_team_stats,
    convert_fixtures,
    convert_players,
)


class TestSlugToName:
    """Tests du mapping slug -> nom."""

    def test_wydad(self):
        assert _slug_to_name("wydad-athletic-club-2530") == "Wydad AC"

    def test_raja(self):
        assert _slug_to_name("raja-club-athletic-de-casablanca-2536") == "Raja CA"

    def test_inconnu(self):
        assert _slug_to_name("club-inconnu-9999") == "Club Inconnu"


class TestConvertTeamStats:
    """Tests de la conversion CSV equipe -> texte."""

    def test_conversion_ok(self):
        df = pd.DataFrame({
            "Stat": ["Wins", "Losses"],
            "Overall": ["10", "2"],
            "Home": ["6", "1"],
            "Away": ["4", "1"],
        })
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "wydad-athletic-club-2530.csv"
            df.to_csv(csv_path, index=False)

            out_dir = Path(tmpdir) / "out"
            out = convert_team_stats(csv_path, out_dir)

            assert out is not None
            text = out.read_text(encoding="utf-8")
            assert "Wydad AC" in text
            assert "Wins" in text
            assert "10" in text

    def test_fichier_inexistant(self):
        out = convert_team_stats(Path("nope.csv"), Path("."))
        assert out is None


class TestConvertFixtures:
    """Tests de la conversion fixtures.csv -> texte."""

    def test_conversion_tous(self):
        df = pd.DataFrame({
            "Date": ["2025-09-01"],
            "Home": ["Wydad AC"],
            "Away": ["Raja CA"],
            "Score": ["2-1"],
            "Status": ["FT"],
        })
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "fixtures.csv"
            df.to_csv(csv_path, index=False)

            out = convert_fixtures(csv_path, Path(tmpdir))
            assert out is not None
            text = out.read_text(encoding="utf-8")
            assert "Wydad AC" in text
            assert "2-1" in text

    def test_filtre_equipe(self):
        df = pd.DataFrame({
            "Date": ["2025-09-01", "2025-09-05"],
            "Home": ["Wydad AC", "FUS Rabat"],
            "Away": ["Raja CA", "Wydad AC"],
            "Score": ["2-1", "0-0"],
            "Status": ["FT", "FT"],
        })
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "fixtures.csv"
            df.to_csv(csv_path, index=False)

            out = convert_fixtures(csv_path, Path(tmpdir), team_filter="Wydad AC")
            assert out is not None
            text = out.read_text(encoding="utf-8")
            assert text.count("Wydad AC") >= 2


class TestConvertPlayers:
    """Tests de la conversion players.csv -> texte."""

    def test_conversion_ok(self):
        df = pd.DataFrame({
            "Category": ["Top Scorers"],
            "Rank": ["1"],
            "Player": ["Joueur A"],
            "Value": ["5"],
            "Metric": ["goals"],
        })
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "players.csv"
            df.to_csv(csv_path, index=False)

            out = convert_players(csv_path, Path(tmpdir))
            assert out is not None
            text = out.read_text(encoding="utf-8")
            assert "Joueur A" in text
            assert "Top Scorers" in text
