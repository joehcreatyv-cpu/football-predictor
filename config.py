"""Configuration : tout vient de l'environnement, aucune clé dans le code."""
import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    api_key: str
    base_url: str = "https://api.football-data.org/v4"
    min_games: int = 5              # matchs min. dans le tableau domicile/extérieur
    min_probability: float = 0.70   # en dessous : aucun pronostic affiché
    priority_probability: float = 0.80
    standings_ttl: int = 6 * 3600
    matches_ttl: int = 300
    max_calls_per_minute: int = 9   # quota gratuit football-data : 10/min
    window_hours: int = 48
    tz: str = "UTC"
    log_path: str = "predictions.jsonl"
    odds_api_key: str = ""              # optionnel : sans clé, pas de cotes (le reste marche)
    odds_monthly_budget: int = 400      # plan gratuit = 500 crédits/mois, marge de sécurité
    odds_ttl: int = 12 * 3600           # 1 appel par championnat et par 12 h au maximum
    odds_cache_path: str = "odds_cache.json"
    odds_state_path: str = "odds_state.json"
    value_edge: float = 0.04            # écart modèle - marché pour signaler une "valeur"


def load_settings() -> Settings:
    key = os.environ.get("FOOTBALL_DATA_KEY", "").strip()
    if not key:
        raise RuntimeError("Variable d'environnement FOOTBALL_DATA_KEY manquante.")
    return Settings(
        api_key=key,
        min_probability=float(os.environ.get("MIN_PROBABILITY", "0.70")),
        window_hours=int(os.environ.get("WINDOW_HOURS", "48")),
        tz=os.environ.get("APP_TZ", "UTC"),
        log_path=os.environ.get("PREDICTION_LOG", "predictions.jsonl"),
        odds_api_key=os.environ.get("ODDS_API_KEY", "").strip(),
    )
