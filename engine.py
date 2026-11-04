"""Moteur d'analyse : agents déterministes, sans état, testables hors-ligne.

Agents : Données -> Force des équipes -> Modèle de score -> Marchés -> Garde-fou -> Décision.
Aucun agent n'invente de donnée : si l'information manque, le match est rejeté.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

MAX_GOALS = 10
DC_RHO = -0.05            # hypothèse fixe (correction Dixon-Coles), non ajustée sur des données
SHRINK_K = 6.0            # poids de la régression vers la moyenne de la ligue
LAMBDA_MIN, LAMBDA_MAX = 0.15, 4.5

PICKS = {
    "1X": "1X (Double Chance Domicile)",
    "X2": "X2 (Double Chance Extérieur)",
    "O15": "Plus de 1.5 Buts dans le match",
    "U35": "Moins de 3.5 Buts dans le match",
}


class Rejected(Exception):
    """Le match est écarté volontairement (donnée insuffisante, résultat douteux...)."""

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


@dataclass(frozen=True)
class SplitStats:
    played: int
    goals_for: int
    goals_against: int


@dataclass(frozen=True)
class LeagueTables:
    home: dict       # team_id -> SplitStats (matchs à domicile uniquement)
    away: dict       # team_id -> SplitStats (matchs à l'extérieur uniquement)
    positions: dict  # team_id -> rang général


# ---------------------------------------------------------------- Agent Données
def parse_standings(standings) -> LeagueTables | None:
    """Sépare proprement les tableaux TOTAL / HOME / AWAY (jamais de mélange)."""
    home, away, pos = {}, {}, {}
    for block in standings or []:
        kind = block.get("type")
        for row in block.get("table", []):
            tid = (row.get("team") or {}).get("id")
            if tid is None:
                continue
            stats = SplitStats(
                int(row.get("playedGames") or 0),
                int(row.get("goalsFor") or 0),
                int(row.get("goalsAgainst") or 0),
            )
            if kind == "HOME":
                home[tid] = stats
            elif kind == "AWAY":
                away[tid] = stats
            elif kind == "TOTAL":
                pos[tid] = int(row.get("position") or 0)
    if not home or not away:
        return None
    return LeagueTables(home, away, pos)


# ---------------------------------------------------------- Agent Force d'équipe
def expected_goals(tables: LeagueTables, home_id: int, away_id: int, min_games: int = 5):
    h = tables.home.get(home_id)
    a = tables.away.get(away_id)
    if h is None or a is None:
        raise Rejected("team_not_in_table")
    if h.played < min_games or a.played < min_games:
        raise Rejected("too_few_games")

    games_h = sum(s.played for s in tables.home.values())
    games_a = sum(s.played for s in tables.away.values())
    if games_h == 0 or games_a == 0:
        raise Rejected("empty_league_table")
    lg_home = sum(s.goals_for for s in tables.home.values()) / games_h   # buts dom./match
    lg_away = sum(s.goals_for for s in tables.away.values()) / games_a   # buts ext./match
    if lg_home <= 0 or lg_away <= 0:
        raise Rejected("degenerate_league_average")

    def shrink(ratio: float, n: int) -> float:
        w = n / (n + SHRINK_K)
        return w * ratio + (1.0 - w)

    att_h = shrink((h.goals_for / h.played) / lg_home, h.played)
    def_h = shrink((h.goals_against / h.played) / lg_away, h.played)
    att_a = shrink((a.goals_for / a.played) / lg_away, a.played)
    def_a = shrink((a.goals_against / a.played) / lg_home, a.played)

    return lg_home * att_h * def_a, lg_away * att_a * def_h


# -------------------------------------------------------------- Agent Garde-fou
def check_lambdas(lh: float, la: float) -> None:
    for v in (lh, la):
        if not math.isfinite(v) or not (LAMBDA_MIN <= v <= LAMBDA_MAX):
            raise Rejected("lambda_out_of_range")


def check_matrix(matrix) -> None:
    total = sum(sum(row) for row in matrix)
    if not math.isfinite(total) or abs(total - 1.0) > 1e-9:
        raise Rejected("matrix_not_normalised")


# ------------------------------------------------------------ Agent Modèle score
def poisson_pmf(k: int, lam: float) -> float:
    return math.exp(-lam + k * math.log(lam) - math.lgamma(k + 1))


def score_matrix(lh: float, la: float, rho: float = DC_RHO):
    m = [[poisson_pmf(i, lh) * poisson_pmf(j, la) for j in range(MAX_GOALS + 1)]
         for i in range(MAX_GOALS + 1)]
    m[0][0] *= 1 - lh * la * rho
    m[0][1] *= 1 + lh * rho
    m[1][0] *= 1 + la * rho
    m[1][1] *= 1 - rho
    total = sum(sum(r) for r in m)
    return [[x / total for x in r] for r in m]


# --------------------------------------------------------------- Agent Marchés
def market_probabilities(m) -> dict:
    p1 = px = p2 = btts = o15 = o25 = u35 = 0.0
    best, best_p = (0, 0), -1.0
    for i, row in enumerate(m):
        for j, p in enumerate(row):
            if i > j:
                p1 += p
            elif i == j:
                px += p
            else:
                p2 += p
            if i > 0 and j > 0:
                btts += p
            if i + j >= 2:
                o15 += p
            if i + j >= 3:
                o25 += p
            if i + j <= 3:
                u35 += p
            if p > best_p:
                best, best_p = (i, j), p
    return {"p1": p1, "px": px, "p2": p2, "btts": btts, "o15": o15, "o25": o25,
            "u35": u35, "best_score": best, "best_score_p": best_p}


# ------------------------------------------------------------- Agent Décision
def decide(mk: dict, min_probability: float):
    cands = {"1X": mk["p1"] + mk["px"], "X2": mk["px"] + mk["p2"],
             "O15": mk["o15"], "U35": mk["u35"]}
    key = max(cands, key=cands.get)
    if cands[key] < min_probability:
        raise Rejected("below_threshold")
    return key, cands[key]


def evaluate_pick(key: str, hg: int, ag: int) -> bool:
    return {"1X": hg >= ag, "X2": ag >= hg, "O15": hg + ag >= 2, "U35": hg + ag <= 3}[key]


# ------------------------------------------------------------------- Pipeline
def analyse_match(tables: LeagueTables, home_id: int, away_id: int,
                  min_games: int = 5, min_probability: float = 0.70,
                  priority_probability: float = 0.80) -> dict:
    lh, la = expected_goals(tables, home_id, away_id, min_games)
    check_lambdas(lh, la)
    m = score_matrix(lh, la)
    check_matrix(m)
    mk = market_probabilities(m)
    key, prob = decide(mk, min_probability)

    pct = lambda x: round(x * 100, 1)  # noqa: E731
    metrics = {
        "dom_domination": int(round(mk["p1"] * 100)),   # P(victoire domicile)
        "ext_domination": int(round(mk["p2"] * 100)),   # P(victoire extérieur)
        "draw_prob": int(round(mk["px"] * 100)),        # P(nul)
        "attack_pressure": int(round(mk["o25"] * 100)), # P(plus de 2.5 buts)
        "defense_stability": int(round(mk["u35"] * 100)),  # P(moins de 3.5 buts)
    }
    return {
        "xg_home": round(lh, 2),
        "xg_away": round(la, 2),
        "pick_key": key,
        "selected_pick": PICKS[key],
        "exact_score": f"{mk['best_score'][0]} - {mk['best_score'][1]}",
        "exact_score_prob": pct(mk["best_score_p"]),
        "btts": "Oui" if mk["btts"] >= 0.5 else "Non",
        "btts_prob": pct(mk["btts"]),
        "goals_pick": "Plus de 2.5 Buts" if mk["o25"] >= 0.58 else "Plus de 1.5 Buts",
        "over15_prob": pct(mk["o15"]),
        "over25_prob": pct(mk["o25"]),
        "confidence": pct(prob),           # = probabilité du modèle, pas une "certitude"
        "reliability_score": pct(prob),
        "is_priority": prob >= priority_probability,
        "probs": {k: round(mk[k], 4) for k in ("p1", "px", "p2", "o15", "o25", "u35", "btts")},
        "metrics": metrics,
        "demographics": metrics,
    }
