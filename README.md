# Football Predictor (v2)

Démarrage :
1. `pip install -r requirements.txt`
2. Copiez votre dossier `templates/` (avec `index.html`) à côté de `app.py`.
3. `export FOOTBALL_DATA_KEY=<votre NOUVELLE clé>` puis `python app.py`
4. Production : `gunicorn -w 1 --threads 4 app:app` (1 seul worker : cache et limiteur partagés).
5. Tests : `python -m unittest discover -s tests -v`
6. Mesure réelle, après quelques jours de scans : `python evaluate.py`

Compatibilité : même JSON que l'ancienne version. Les clés `metrics` gardent leurs noms
mais contiennent maintenant de vraies probabilités (voir engine.py). `confidence` = probabilité du modèle.
Les erreurs de source renvoient HTTP 503 (avant : 200 avec un message).

## Budget 0 $ : ce qui est gratuit
- **Résultats/classements en direct** : football-data.org (plan gratuit, 10 req/min, limiteur intégré).
- **Backtest** : fichiers CSV gratuits de football-data.co.uk (résultats + cotes historiques, un fichier = une ligue et une saison).
  `python backtest.py data/E0.csv data/D1.csv ...` compare le modèle au marché et affiche calibration + ROI avec marge d'erreur.
- **Cotes en direct (optionnel)** : The Odds API, plan gratuit (500 crédits/mois). Le module `odds.py` limite à 1 appel/championnat/12 h
  et s'arrête de lui-même à 400 crédits. Championnats couverts : PL, PD, BL1, SA, FL1. Sans `ODDS_API_KEY`, tout fonctionne sauf `market_1x2`/`market_pick`.
- **Hébergement** : lancer en local suffit. Les offres d'hébergement gratuit changent souvent : vérifiez leurs limites actuelles avant de choisir.

## Lire les nouveaux champs
`market_1x2` = probabilités du marché sans marge (consensus des bookmakers). `market_pick.edge` = écart modèle - marché sur la double chance.
Ce n'est PAS une cote jouable : les vraies cotes de double chance contiennent la marge du bookmaker.
