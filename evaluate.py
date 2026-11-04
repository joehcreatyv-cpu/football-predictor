"""Compare les pronostics journalisés aux scores réels : taux de réussite + calibration.
Usage : FOOTBALL_DATA_KEY=... python evaluate.py   (NON testé contre l'API réelle)"""
import json
from collections import defaultdict
from datetime import datetime, timedelta, timezone

from config import load_settings
from engine import evaluate_pick
from football_client import FootballClient


def main():
    s = load_settings()
    client = FootballClient(s)
    preds = {}
    with open(s.log_path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            preds.setdefault(r["match_id"], r)          # on garde le 1er pronostic
    if not preds:
        return print("Aucun pronostic journalisé.")

    days = sorted(datetime.fromisoformat(p["utc_date"].replace("Z", "+00:00")).date() for p in preds.values())
    results, d = {}, days[0]
    today = datetime.now(timezone.utc).date()
    while d <= min(days[-1], today):
        end = min(d + timedelta(days=8), today)
        for m in client.matches(d, end) or []:
            ft = (m.get("score") or {}).get("fullTime") or {}
            if m.get("status") == "FINISHED" and ft.get("home") is not None:
                results[m["id"]] = (ft["home"], ft["away"])
        d = end + timedelta(days=1)

    buckets, hits, n, brier = defaultdict(lambda: [0, 0]), 0, 0, 0.0
    for mid, p in preds.items():
        if mid not in results:
            continue
        won = evaluate_pick(p["pick_key"], *results[mid])
        n += 1; hits += won; brier += (p["prob"] / 100 - won) ** 2
        b = buckets[int(p["prob"] // 5) * 5]
        b[0] += 1; b[1] += won
    if not n:
        return print("Aucun match terminé à évaluer pour l'instant.")
    print(f"{n} pronostics évalués | réussite {hits / n:.1%} | Brier {brier / n:.3f}")
    for lo in sorted(buckets):
        t, w = buckets[lo]
        print(f"  prob annoncée {lo}-{lo + 5}% : {w}/{t} = {w / t:.0%}")


if __name__ == "__main__":
    main()
