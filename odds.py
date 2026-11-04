"""Cotes gratuites (The Odds API, plan gratuit) avec garde-budget local strict.

Coût visé : 1 crédit par appel (1 marché h2h x 1 région). Avec 5 championnats et un cache de 12 h,
on reste autour de 300 crédits/mois sur 500. Sans clé ODDS_API_KEY, le module est simplement inactif.
"""
import json
import logging
import re
import threading
import time
import unicodedata
from datetime import datetime, timezone
from difflib import SequenceMatcher

import requests

log = logging.getLogger(__name__)
BASE = "https://api.the-odds-api.com/v4"
SPORT_KEYS = {  # code football-data.org -> clé The Odds API
    "PL": "soccer_epl", "PD": "soccer_spain_la_liga", "BL1": "soccer_germany_bundesliga",
    "SA": "soccer_italy_serie_a", "FL1": "soccer_france_ligue_one",
}
STOP = {"fc", "cf", "afc", "ac", "as", "sc", "ssc", "rc", "the", "club", "and", "de", "1"}


def devig(h: float, d: float, a: float):
    """Retire la marge du bookmaker (normalisation proportionnelle)."""
    inv = [1.0 / h, 1.0 / d, 1.0 / a]
    s = sum(inv)
    return [x / s for x in inv]


def consensus_probs(event: dict):
    home, away = event.get("home_team"), event.get("away_team")
    acc = []
    for b in event.get("bookmakers", []):
        for mk in b.get("markets", []):
            if mk.get("key") != "h2h":
                continue
            price = {o.get("name"): o.get("price") for o in mk.get("outcomes", [])}
            try:
                h, d, a = float(price[home]), float(price["Draw"]), float(price[away])
            except (KeyError, TypeError, ValueError):
                continue
            if min(h, d, a) <= 1.0:
                continue
            acc.append(devig(h, d, a))
    if not acc:
        return None
    n = len(acc)
    return {"p1": sum(x[0] for x in acc) / n, "px": sum(x[1] for x in acc) / n,
            "p2": sum(x[2] for x in acc) / n, "n_books": n}


def _norm(s: str) -> str:
    s = "".join(c for c in unicodedata.normalize("NFKD", s or "") if not unicodedata.combining(c)).lower()
    return " ".join(t for t in re.sub(r"[^a-z0-9 ]", " ", s).split() if t not in STOP)


def name_score(a: str, b: str) -> float:
    na, nb = _norm(a), _norm(b)
    if not na or not nb:
        return 0.0
    if na == nb:
        return 1.0
    if na in nb or nb in na:
        return 0.9
    return SequenceMatcher(None, na, nb).ratio()


def find_event(events, home: str, away: str, kickoff: datetime | None = None, min_score: float = 0.80):
    best, best_s = None, 0.0
    for ev in events or []:
        score = min(name_score(home, ev.get("home_team", "")), name_score(away, ev.get("away_team", "")))
        if score < min_score or score <= best_s:
            continue
        if kickoff is not None:
            try:
                t = datetime.fromisoformat(ev["commence_time"].replace("Z", "+00:00"))
                if abs((t - kickoff).total_seconds()) > 24 * 3600:
                    continue
            except (KeyError, ValueError, AttributeError):
                continue
        best, best_s = ev, score
    return best


def pick_vs_market(pick_key: str, probs: dict, market: dict, value_edge: float):
    """Compare le modèle à la probabilité 'consensus sans marge' (doubles chances uniquement)."""
    if pick_key == "1X":
        mine, theirs = probs["p1"] + probs["px"], market["p1"] + market["px"]
    elif pick_key == "X2":
        mine, theirs = probs["px"] + probs["p2"], market["px"] + market["p2"]
    else:
        return None
    edge = mine - theirs
    return {"model_prob": round(mine * 100, 1), "market_prob": round(theirs * 100, 1),
            "edge": round(edge * 100, 1), "value": edge >= value_edge, "books": market["n_books"]}


class OddsClient:
    def __init__(self, api_key, cache_path, state_path, ttl=12 * 3600, monthly_budget=400):
        self.api_key, self.ttl, self.budget = api_key, ttl, monthly_budget
        self.enabled = bool(api_key)
        self.cache_path, self.state_path = cache_path, state_path
        self._lock = threading.Lock()
        self._cache = self._load(cache_path)
        self._state = self._load(state_path)

    @staticmethod
    def _load(path):
        try:
            with open(path, encoding="utf-8") as f:
                return json.load(f)
        except (OSError, ValueError):
            return {}

    def _save(self):
        for path, data in ((self.cache_path, self._cache), (self.state_path, self._state)):
            try:
                with open(path, "w", encoding="utf-8") as f:
                    json.dump(data, f)
            except OSError as exc:
                log.warning("Écriture impossible (%s) : %s", path, exc)

    def _month_state(self):
        month = datetime.now(timezone.utc).strftime("%Y-%m")
        if self._state.get("month") != month:
            self._state = {"month": month, "used": 0}
        return self._state

    def events(self, sport_key):
        if not self.enabled:
            return None
        with self._lock:
            now = time.time()
            hit = self._cache.get(sport_key)
            stale = hit["events"] if hit else None
            if hit and now - hit["ts"] < self.ttl:
                return stale
            st = self._month_state()
            if st["used"] >= self.budget or st.get("remaining", 10**9) <= 20:
                log.warning("Budget de cotes épuisé : aucun appel (%s)", sport_key)
                return stale
            try:
                r = requests.get(f"{BASE}/sports/{sport_key}/odds",
                                 params={"apiKey": self.api_key, "regions": "eu",
                                         "markets": "h2h", "oddsFormat": "decimal"}, timeout=10)
                if r.status_code != 200:
                    log.warning("The Odds API : HTTP %s", r.status_code)
                    return stale
                events = r.json()
            except (requests.RequestException, ValueError) as exc:
                log.error("Échec cotes : %s", exc)
                return stale
            try:
                st["used"] += int(float(r.headers.get("x-requests-last", 1)))
                if r.headers.get("x-requests-remaining") is not None:
                    st["remaining"] = int(float(r.headers["x-requests-remaining"]))
            except ValueError:
                st["used"] += 1
            self._cache[sport_key] = {"ts": now, "events": events}
            self._save()
            return events

    def market_for(self, comp_code, home, away, kickoff):
        key = SPORT_KEYS.get(comp_code)
        if not key:
            return None
        ev = find_event(self.events(key), home, away, kickoff)
        return consensus_probs(ev) if ev else None
