"""Client API : cache, limiteur de débit local, erreurs explicites (None = échec)."""
import logging
import threading
import time
from collections import deque

import requests

log = logging.getLogger(__name__)


class RateLimiter:
    def __init__(self, max_calls: int, period: float = 60.0):
        self.max_calls, self.period = max_calls, period
        self._calls = deque()
        self._lock = threading.Lock()

    def try_acquire(self) -> bool:
        now = time.monotonic()
        with self._lock:
            while self._calls and now - self._calls[0] > self.period:
                self._calls.popleft()
            if len(self._calls) >= self.max_calls:
                return False
            self._calls.append(now)
            return True


class FootballClient:
    def __init__(self, settings):
        self.s = settings
        self.session = requests.Session()
        self.session.headers["X-Auth-Token"] = settings.api_key
        self.limiter = RateLimiter(settings.max_calls_per_minute)
        self._cache = {}
        self._lock = threading.Lock()

    def _get(self, path, params=None, ttl=0):
        key = (path, tuple(sorted((params or {}).items())))
        now = time.time()
        with self._lock:
            hit = self._cache.get(key)
        if hit and now - hit[1] < ttl:
            return hit[0]
        stale = hit[0] if hit else None
        if not self.limiter.try_acquire():
            log.warning("Limite locale de requêtes atteinte (%s)", path)
            return stale
        try:
            r = self.session.get(self.s.base_url + path, params=params, timeout=10)
            if r.status_code == 429:
                log.warning("429 reçu de l'API (%s)", path)
                return stale
            r.raise_for_status()
            data = r.json()
        except (requests.RequestException, ValueError) as exc:
            log.error("Échec API %s : %s", path, exc)
            return stale
        with self._lock:
            self._cache[key] = (data, now)
        return data

    def matches(self, d_from, d_to):
        data = self._get(
            "/matches",
            {"dateFrom": d_from.strftime("%Y-%m-%d"), "dateTo": d_to.strftime("%Y-%m-%d")},
            ttl=self.s.matches_ttl,
        )
        return None if data is None else data.get("matches", [])

    def standings(self, competition_id):
        data = self._get(f"/competitions/{competition_id}/standings", ttl=self.s.standings_ttl)
        return None if data is None else data.get("standings", [])
