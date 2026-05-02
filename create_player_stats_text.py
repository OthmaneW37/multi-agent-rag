import pandas as pd
from pathlib import Path

df = pd.read_csv("ScrappingDataBotola/data/player_stats.csv")

# Garder uniquement les colonnes les plus importantes pour le RAG
key_cols = [
    "player_name", "club", "position", "nationality", "age", "kit_number",
    "rating", "matches_played", "minutes", "minutes_per90",
    "goals_scored", "goals_scored_per90", "expected_goals_xg", "expected_goals_xg_per90",
    "assists", "assists_per90", "expected_assists_xa", "expected_assists_xa_per90",
    "shots_taken", "shots_taken_per90", "shot_conversion_rate",
    "key_passes", "key_passes_per90",
    "dribbles", "dribbles_per90", "dribble_success_rate",
    "yellow_cards", "red_cards", "total_cards", "total_cards_per90",
    "fouls_committed", "fouls_committed_per90",
    "tackles", "tackles_per90", "interceptions", "interceptions_per90",
    "clean_sheets", "clean_sheets_per90",
]

# Filtrer les colonnes qui existent
available_cols = [c for c in key_cols if c in df.columns]
df_key = df[available_cols].copy()

# Remplacer les valeurs vides par "N/A"
df_key = df_key.fillna("N/A")

# Créer le fichier texte
lines = [
    "STATS INDIVIDUELLES DES JOUEURS - BOTOLA PRO 2025/2026",
    "=" * 70,
    "",
    f"Source: FootyStats.org",
    f"Nombre total de joueurs: {len(df_key)}",
    f"Nombre de stats par joueur: {len(available_cols)}",
    "",
    "ATTENTION: Les stats ci-dessous proviennent de FootyStats.",
    "Certaines métriques peuvent manquer pour certains joueurs.",
    "Les stats 'per90' sont calculées par 90 minutes jouées.",
    "",
]

# Grouper par club pour faciliter la recherche RAG
clubs = df_key["club"].dropna().unique()
for club in sorted(clubs):
    if not club or club == "N/A":
        continue
    club_players = df_key[df_key["club"] == club]
    if club_players.empty:
        continue
    
    lines.append(f"\n{'='*70}")
    lines.append(f"CLUB: {club}")
    lines.append(f"{'='*70}")
    lines.append(f"Nombre de joueurs dans la base: {len(club_players)}")
    lines.append("")
    
    for _, row in club_players.iterrows():
        name = str(row.get("player_name", "")).strip()
        pos = str(row.get("position", "")).strip()
        nation = str(row.get("nationality", "")).strip()
        age = str(row.get("age", "")).strip()
        num = str(row.get("kit_number", "")).strip()
        
        lines.append(f"\n--- {name} ---")
        if pos and pos != "N/A":
            lines.append(f"Position: {pos}")
        if nation and nation != "N/A":
            lines.append(f"Nationalite: {nation}")
        if age and age != "N/A":
            lines.append(f"Age: {age}")
        if num and num != "N/A":
            lines.append(f"Numero: #{num}")
        
        # Stats générales
        mp = str(row.get("matches_played", "")).strip()
        mins = str(row.get("minutes", "")).strip()
        rating = str(row.get("rating", "")).strip()
        if mp and mp != "N/A":
            lines.append(f"Matchs joues: {mp}")
        if mins and mins != "N/A":
            lines.append(f"Minutes: {mins}")
        if rating and rating != "N/A":
            lines.append(f"Note moyenne: {rating}")
        
        # Stats attaque
        goals = str(row.get("goals_scored", "")).strip()
        xg = str(row.get("expected_goals_xg", "")).strip()
        assists = str(row.get("assists", "")).strip()
        xa = str(row.get("expected_assists_xa", "")).strip()
        if goals and goals != "N/A" and goals != "0":
            lines.append(f"Buts: {goals}")
        if xg and xg != "N/A" and xg != "0":
            lines.append(f"xG: {xg}")
        if assists and assists != "N/A" and assists != "0":
            lines.append(f"Passes decisives: {assists}")
        if xa and xa != "N/A" and xa != "0":
            lines.append(f"xA: {xa}")
        
        # Stats tirs
        shots = str(row.get("shots_taken", "")).strip()
        conv = str(row.get("shot_conversion_rate", "")).strip()
        if shots and shots != "N/A" and shots != "0":
            lines.append(f"Tirs: {shots}")
        if conv and conv != "N/A" and conv != "0.00%":
            lines.append(f"Conversion: {conv}")
        
        # Stats passes
        kp = str(row.get("key_passes", "")).strip()
        if kp and kp != "N/A" and kp != "0":
            lines.append(f"Passes cles: {kp}")
        
        # Stats dribbles
        drib = str(row.get("dribbles", "")).strip()
        drib_sr = str(row.get("dribble_success_rate", "")).strip()
        if drib and drib != "N/A" and drib != "0":
            lines.append(f"Dribbles: {drib}")
        if drib_sr and drib_sr != "N/A" and drib_sr != "0.00%":
            lines.append(f"Reussite dribbles: {drib_sr}")
        
        # Stats discipline
        yc = str(row.get("yellow_cards", "")).strip()
        rc = str(row.get("red_cards", "")).strip()
        if yc and yc != "N/A" and yc != "0":
            lines.append(f"Cartons jaunes: {yc}")
        if rc and rc != "N/A" and rc != "0":
            lines.append(f"Cartons rouges: {rc}")
        
        # Stats défense
        tackles = str(row.get("tackles", "")).strip()
        inter = str(row.get("interceptions", "")).strip()
        cs = str(row.get("clean_sheets", "")).strip()
        if tackles and tackles != "N/A" and tackles != "0":
            lines.append(f"Tacles: {tackles}")
        if inter and inter != "N/A" and inter != "0":
            lines.append(f"Interceptions: {inter}")
        if cs and cs != "N/A" and cs != "0":
            lines.append(f"Clean sheets: {cs}")

# Top buteurs global
lines.append(f"\n{'='*70}")
lines.append("TOP BUTEURS - BOTOLA PRO 2025/2026")
lines.append(f"{'='*70}")
lines.append("")

top_scorers = df_key[df_key["goals_scored"].astype(str).str.replace("N/A","0", regex=False).str.replace(",","", regex=False).astype(float, errors="ignore") > 0].copy()
if not top_scorers.empty and "goals_scored" in top_scorers.columns:
    try:
        top_scorers["goals_num"] = pd.to_numeric(top_scorers["goals_scored"], errors="coerce").fillna(0)
        top = top_scorers.nlargest(20, "goals_num")
        for _, row in top.iterrows():
            name = str(row.get("player_name", "")).strip()
            club = str(row.get("club", "")).strip()
            goals = str(row.get("goals_scored", "")).strip()
            xg = str(row.get("expected_goals_xg", "")).strip()
            mp = str(row.get("matches_played", "")).strip()
            info = f"{name} ({club}): {goals} buts"
            if xg and xg != "N/A":
                info += f" (xG: {xg})"
            if mp and mp != "N/A":
                info += f" en {mp} matchs"
            lines.append(info)
    except Exception:
        pass

# Top passeurs
lines.append(f"\n{'='*70}")
lines.append("TOP PASSEURS DECISIFS - BOTOLA PRO 2025/2026")
lines.append(f"{'='*70}")
lines.append("")

top_assisters = df_key[df_key["assists"].astype(str).str.replace("N/A","0", regex=False).str.replace(",","", regex=False).astype(float, errors="ignore") > 0].copy()
if not top_assisters.empty and "assists" in top_assisters.columns:
    try:
        top_assisters["assists_num"] = pd.to_numeric(top_assisters["assists"], errors="coerce").fillna(0)
        top = top_assisters.nlargest(15, "assists_num")
        for _, row in top.iterrows():
            name = str(row.get("player_name", "")).strip()
            club = str(row.get("club", "")).strip()
            assists = str(row.get("assists", "")).strip()
            xa = str(row.get("expected_assists_xa", "")).strip()
            info = f"{name} ({club}): {assists} passes D"
            if xa and xa != "N/A":
                info += f" (xA: {xa})"
            lines.append(info)
    except Exception:
        pass

# Write file
out_path = Path("data/raw/player_stats_botola.txt")
out_path.write_text("\n".join(lines), encoding="utf-8")
print(f"Fichier genere: {out_path}")
print(f"Joueurs: {len(df_key)}")
print(f"Clubs: {len(clubs)}")
print(f"Lignes: {len(lines)}")
