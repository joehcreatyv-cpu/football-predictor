"""Backtest 'walk-forward' sur les CSV gratuits de football-data.co.uk.

Pour chaque match, le modèle n'utilise QUE les matchs déjà joués avant lui dans la même saison.
Il est comparé au marché (cotes sans marge) : c'est le seul test qui compte vraiment.

Usage : python backtest.py data/E0_2425.csv data/D1_2425.csv ...   (un fichier = une ligue, une saison)
"""
import csv
import glob
import math
import sys
from collections import defaultdict
from datetime import datetime

from engine import LeagueTables, Rejected, SplitStats, check_lambdas, evaluate_pick, expected_goals, \
    market_probabilities, score_matrix
from odds import devig

# Cotes de clôture d'abord (les plus fiables), puis cotes d'ouverture, puis moyenne du marché.
ODDS_SETS = [("PSCH", "PSCD", "PSCA"), ("B365CH", "B365CD", "B365CA"), ("PSH", "PSD", "PSA"),
             ("B365H", "B365D", "B365A"), ("AvgH", "AvgD", "AvgA")]
THRESHOLDS = (0.65, 0.70, 0.75, 0.80)


def _date(s):
    for fmt in ("%d/%m/%Y", "%d/%m/%y"):
        try:
            return datetime.strptime(s.strip(), fmt)
        except ValueError:
            pass
    return None


def _odds(row):
    for cols in ODDS_SETS:
        try:
            o = [float(row[c]) for c in cols]
        except (KeyError, ValueError, TypeError):
            continue
        if min(o) > 1.0:
            return o, cols[0]
    return None, None


def load_rows(path):
    rows = []
    with open(path, encoding="utf-8-sig", errors="replace", newline="") as f:
        for r in csv.DictReader(f):
            d = _date(r.get("Date", ""))
            try:
                hg, ag = int(r["FTHG"]), int(r["FTAG"])
            except (KeyError, ValueError, TypeError):
                continue
            if d is None or not r.get("HomeTeam") or not r.get("AwayTeam"):
                continue
            odds, src = _odds(r)
            rows.append({"date": d, "time": r.get("Time", "") or "", "home": r["HomeTeam"], "away": r["AwayTeam"],
                         "hg": hg, "ag": ag, "odds": odds, "src": src})
    rows.sort(key=lambda x: (x["date"], x["time"]))
    return rows


def predict_season(rows, min_games=5):
    home_acc, away_acc, out = defaultdict(lambda: [0, 0, 0]), defaultdict(lambda: [0, 0, 0]), []
    for r in rows:
        tables = LeagueTables({t: SplitStats(*v) for t, v in home_acc.items()},
                              {t: SplitStats(*v) for t, v in away_acc.items()}, {})
        try:
            lh, la = expected_goals(tables, r["home"], r["away"], min_games)
            check_lambdas(lh, la)
            mk = market_probabilities(score_matrix(lh, la))
            out.append({**r, "mk": mk})
        except Rejected:
            pass
        home_acc[r["home"]][0] += 1; home_acc[r["home"]][1] += r["hg"]; home_acc[r["home"]][2] += r["ag"]
        away_acc[r["away"]][0] += 1; away_acc[r["away"]][1] += r["ag"]; away_acc[r["away"]][2] += r["hg"]
    return out


def _idx(r):
    return 0 if r["hg"] > r["ag"] else (1 if r["hg"] == r["ag"] else 2)


def _brier(p, i):
    return sum((p[k] - (1.0 if k == i else 0.0)) ** 2 for k in range(3))


def _logloss(p, i):
    return -math.log(max(p[i], 1e-12))


def summarise(preds, edges=(0.03, 0.05, 0.10)):
    s = {"n": len(preds)}
    mb = [(_brier([r["mk"][k] for k in ("p1", "px", "p2")], _idx(r)),
           _logloss([r["mk"][k] for k in ("p1", "px", "p2")], _idx(r))) for r in preds]
    s["model_brier"] = sum(x[0] for x in mb) / len(mb) if mb else float("nan")
    s["model_logloss"] = sum(x[1] for x in mb) / len(mb) if mb else float("nan")

    with_odds = [r for r in preds if r["odds"]]
    s["n_odds"] = len(with_odds)
    if with_odds:
        mp = [devig(*r["odds"]) for r in with_odds]
        s["market_brier"] = sum(_brier(p, _idx(r)) for p, r in zip(mp, with_odds)) / len(with_odds)
        s["market_logloss"] = sum(_logloss(p, _idx(r)) for p, r in zip(mp, with_odds)) / len(with_odds)
        s["model_brier_same"] = sum(_brier([r["mk"][k] for k in ("p1", "px", "p2")], _idx(r))
                                    for r in with_odds) / len(with_odds)
        s["model_logloss_same"] = sum(_logloss([r["mk"][k] for k in ("p1", "px", "p2")], _idx(r))
                                      for r in with_odds) / len(with_odds)
        s["sources"] = sorted({r["src"] for r in with_odds})
        s["value"] = {}
        for e in edges:
            profits = []
            for r in with_odds:
                p = [r["mk"][k] for k in ("p1", "px", "p2")]
                ev = [p[k] * r["odds"][k] - 1.0 for k in range(3)]
                k = max(range(3), key=ev.__getitem__)
                if ev[k] >= e:
                    profits.append(r["odds"][k] - 1.0 if _idx(r) == k else -1.0)
            n = len(profits)
            if n:
                mean = sum(profits) / n
                sd = math.sqrt(sum((x - mean) ** 2 for x in profits) / max(1, n - 1))
                s["value"][e] = (n, mean, 1.96 * sd / math.sqrt(n))
            else:
                s["value"][e] = (0, 0.0, 0.0)

    s["picks"] = {}
    for thr in THRESHOLDS:
        hits = n = 0
        psum = 0.0
        for r in preds:
            mk = r["mk"]
            c = {"1X": mk["p1"] + mk["px"], "X2": mk["px"] + mk["p2"], "O15": mk["o15"], "U35": mk["u35"]}
            key = max(c, key=c.get)
            if c[key] >= thr:
                n += 1; psum += c[key]; hits += evaluate_pick(key, r["hg"], r["ag"])
        s["picks"][thr] = (n, psum / n if n else 0.0, hits / n if n else 0.0)
    return s


def report(s):
    print(f"\nMatchs évalués : {s['n']} (dont {s['n_odds']} avec cotes)")
    print(f"Modèle  : Brier {s['model_brier']:.4f} | log-loss {s['model_logloss']:.4f}")
    if s["n_odds"]:
        print(f"Sur les mêmes matchs, cotes utilisées : {', '.join(s['sources'])}")
        print(f"  Modèle : Brier {s['model_brier_same']:.4f} | log-loss {s['model_logloss_same']:.4f}")
        print(f"  Marché : Brier {s['market_brier']:.4f} | log-loss {s['market_logloss']:.4f}  (plus bas = meilleur)")
        better = s["model_logloss_same"] < s["market_logloss"]
        print("  -> Le modèle bat le marché." if better else "  -> Le modèle NE bat PAS le marché (cas le plus courant).")
        print("Paris 'valeur' 1X2 (mise fixe 1, ROI ± intervalle à 95 %) :")
        for e, (n, roi, ci) in s["value"].items():
            print(f"  écart >= {e:.0%} : {n} paris | ROI {roi:+.1%} ± {ci:.1%}")
    print("Pronostics (annoncé vs réel) :")
    for thr, (n, pm, hr) in s["picks"].items():
        print(f"  seuil {thr:.0%} : {n} pronostics | annoncé {pm:.1%} | réel {hr:.1%} | écart {hr - pm:+.1%}")
    print("\nAvertissement : un ROI dont l'intervalle contient 0 n'est PAS une preuve de rentabilité.")


def main(args):
    files = [f for a in args for f in (glob.glob(a) or [a])]
    if not files:
        return print(__doc__)
    preds = []
    for f in files:
        preds += predict_season(load_rows(f))
    if not preds:
        return print("Aucun match exploitable (fichier vide, colonnes absentes ou saison trop courte).")
    report(summarise(preds))


if __name__ == "__main__":
    main(sys.argv[1:])
