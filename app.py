import json
import logging
import os
import threading
from collections import Counter
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

from flask import Flask, jsonify, render_template, request

from config import load_settings
from engine import Rejected, analyse_match, parse_standings
from football_client import FootballClient
from odds import OddsClient, pick_vs_market

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("scan")
OPEN_STATUSES = {"TIMED", "SCHEDULED"}


class PredictionLog:
    """Journal des pronostics : indispensable pour mesurer la vraie fiabilité plus tard."""

    def __init__(self, path):
        self.path, self._seen, self._lock = path, set(), threading.Lock()

    def write(self, record):
        with self._lock:
            if record["match_id"] in self._seen:
                return
            self._seen.add(record["match_id"])
            try:
                with open(self.path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
            except OSError as exc:
                log.warning("Journal illisible/non inscriptible : %s", exc)


def _parse_dt(value):
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (ValueError, AttributeError):
        return None


def group_by_country(matches):
    tree = {}
    for m in matches:
        c = tree.setdefault(m["country"], {"flag": m["flag"], "leagues": {}})
        c["leagues"].setdefault(m["league"], []).append(m)
    out = []
    for name, c in tree.items():
        leagues = []
        for lname, ms in c["leagues"].items():
            ms.sort(key=lambda x: x["confidence"], reverse=True)
            leagues.append({"league_name": lname, "max_confidence": ms[0]["confidence"], "matches": ms})
        leagues.sort(key=lambda x: x["max_confidence"], reverse=True)
        out.append({"country_name": name, "flag": c["flag"],
                    "max_confidence": leagues[0]["max_confidence"], "leagues": leagues})
    out.sort(key=lambda x: x["max_confidence"], reverse=True)
    return out


def create_app(settings=None):
    settings = settings or load_settings()
    client = FootballClient(settings)
    pred_log = PredictionLog(settings.log_path)
    odds = OddsClient(settings.odds_api_key, settings.odds_cache_path, settings.odds_state_path,
                      settings.odds_ttl, settings.odds_monthly_budget)
    tz = ZoneInfo(settings.tz)
    app = Flask(__name__)

    @app.route("/")
    def home():
        return render_template("index.html")

    @app.route("/health")
    def health():
        return jsonify(status="ok")

    @app.route("/api/scan")
    def scan():
        hours = max(1, min(request.args.get("hours", default=settings.window_hours, type=int), 72))
        now = datetime.now(timezone.utc)
        end = now + timedelta(hours=hours)

        raw = client.matches(now.date(), end.date())
        if raw is None:
            return jsonify(status="error", message="Source de données indisponible ou quota atteint.",
                           countries=[], matches=[]), 503

        tables_by_comp, skipped, flat = {}, Counter(), []
        for m in raw:
            if m.get("status") not in OPEN_STATUSES:
                continue
            dt = _parse_dt(m.get("utcDate"))
            if dt is None or not (now <= dt <= end):
                continue
            home, away = m.get("homeTeam") or {}, m.get("awayTeam") or {}
            comp = m.get("competition") or {}
            if not (home.get("id") and away.get("id") and comp.get("id")):
                skipped["incomplete_match_data"] += 1
                continue

            if comp["id"] not in tables_by_comp:
                tables_by_comp[comp["id"]] = parse_standings(client.standings(comp["id"]))
            tables = tables_by_comp[comp["id"]]
            if tables is None:
                skipped["no_home_away_table"] += 1
                continue

            try:
                a = analyse_match(tables, home["id"], away["id"], settings.min_games,
                                  settings.min_probability, settings.priority_probability)
            except Rejected as r:
                skipped[r.reason] += 1
                continue

            market = odds.market_for(comp.get("code"), home.get("name", ""), away.get("name", ""), dt) \
                if odds.enabled else None
            if market:
                a["market_1x2"] = {"home": round(market["p1"] * 100, 1), "draw": round(market["px"] * 100, 1),
                                   "away": round(market["p2"] * 100, 1), "books": market["n_books"]}
                a["market_pick"] = pick_vs_market(a["pick_key"], a["probs"], market, settings.value_edge)

            area = m.get("area") or {}
            item = {
                "id": m.get("id"), "home": home.get("name", "Domicile"), "away": away.get("name", "Extérieur"),
                "league": comp.get("name", "Championnat"), "country": area.get("name", "International"),
                "flag": area.get("flag", ""), "time": dt.astimezone(tz).strftime("%H:%M"),
                "analysis": a, "prediction": a["selected_pick"], "exact_score": a["exact_score"],
                "btts": a["btts"], "goals_pick": a["goals_pick"], "confidence": a["confidence"],
                "is_priority": a["is_priority"], "metrics": a["metrics"],
            }
            item["home_team"], item["away_team"] = item["home"], item["away"]
            flat.append(item)
            pred_log.write({"match_id": item["id"], "utc_date": m.get("utcDate"), "home": item["home"],
                            "away": item["away"], "pick_key": a["pick_key"], "prob": a["confidence"],
                            "xg_home": a["xg_home"], "xg_away": a["xg_away"],
                            "logged_at": now.isoformat()})

        flat.sort(key=lambda x: x["confidence"], reverse=True)
        return jsonify(status="success", time_window=f"Prochaines {hours}h ({settings.tz})",
                       count=len(flat), countries=group_by_country(flat), matches=flat,
                       skipped=dict(skipped))

    return app


app = create_app() if os.environ.get("FOOTBALL_DATA_KEY") else None

if __name__ == "__main__":
    (app or create_app()).run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
