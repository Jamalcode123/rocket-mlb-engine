"""
ROCKET MLB ENGINE — DATA LAYER v1
===================================
Automated data ingestion, normalization, role tagging, and daily pipeline
for the Rocket MLB betting system.

Data Sources (all free, no auth required unless noted):
  - MLB Stats API (statsapi.mlb.com) — schedule, rosters, lineups, stats, boxscores
  - Pybaseball / Baseball Savant / Statcast — pitch-level data, advanced metrics
  - FanGraphs (via pybaseball) — aggregated season stats, park factors
  - Open-Meteo — hourly weather forecasts
  - UmpScorecards — umpire zone profiles (scraped/cached)
  - Odds API — lines/props (free tier or paid)

Requirements:
  pip install requests pandas pybaseball numpy

Usage:
  from rocket_data_layer import RocketDataLayer
  rdl = RocketDataLayer()
  slate = rdl.run_daily_pipeline("2026-06-15")
"""

import os
import json
import time
import logging
import hashlib
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from typing import Optional
from enum import Enum

import requests
import pandas as pd
import numpy as np

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

MLB_API_BASE = "https://statsapi.mlb.com/api/v1"
OPEN_METEO_BASE = "https://api.open-meteo.com/v1/forecast"
CACHE_DIR = os.path.join(os.path.dirname(__file__), ".rocket_cache")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("RocketDataLayer")


# ══════════════════════════════════════════════════════════════════════════════
# ENUMS — ROLE CLASSIFICATIONS
# ══════════════════════════════════════════════════════════════════════════════

class SPRole(Enum):
    WORKHORSE_ACE = "Workhorse Ace"
    K_SPECIALIST = "K Specialist"
    GROUNDBALL_SUPPRESSOR = "Ground-Ball Suppressor"
    CONTACT_MANAGER = "Contact Manager"
    FLYBALL_FRAGILE = "Fly-Ball Fragile"
    SHORT_LEASH_VOLATILE = "Short-Leash Volatile"
    WALK_RISK_ARM = "Walk-Risk Arm"
    REVERSE_SPLIT_ARM = "Reverse-Split Arm"


class BullpenRole(Enum):
    LOCKDOWN = "Lockdown Pen"
    HIGH_K_BRIDGE = "High-K Bridge Pen"
    VOLATILE_MIDDLE = "Volatile Middle Relief"
    OVERWORKED = "Overworked Pen"
    LOW_LEV_BLEED = "Low-Leverage Bleed Pen"
    COMMITTEE_CHAOS = "Committee Chaos Pen"
    CLOSER_HEAVY = "Closer-Heavy Top-End Pen"


class HitterRole(Enum):
    TABLE_SETTER = "Table Setter"
    CONTACT_BAT = "Contact Bat"
    PLATOON_MASHER = "Platoon Masher"
    LIFT_PULL_POWER = "Lift-and-Pull Power Bat"
    K_PRONE_SLUGGER = "K-Prone Slugger"
    GAP_XBH = "Gap / XBH Bat"
    RUN_PRODUCER = "Run Producer"
    SB_PRESSURE = "SB Pressure Bat"
    PASSIVE_WALKER = "Passive Walker"
    BOOM_BUST_BARREL = "Boom/Bust Barrel Bat"


class TeamOffenseArchetype(Enum):
    HIGH_K = "High-K Offense"
    PATIENT_WALK = "Patient Walk Team"
    POWER_CLUSTER = "Power Cluster Lineup"
    CONTACT_CHAIN = "Contact Chain Offense"
    PLATOON_HEAVY = "Platoon-Heavy Offense"
    AGGRESSIVE_BASEPATH = "Aggressive Basepath Team"
    TOP_HEAVY = "Top-Heavy Offense"
    BOTTOM_ORDER_DEAD = "Bottom-Order Dead Zone"


class Confidence(Enum):
    GREEN = "GREEN"
    YELLOW = "YELLOW"
    RED = "RED"


class ScriptTag(Enum):
    STABLE = "Stable"
    FRAGILE = "Fragile"
    CHAOS = "Chaos"
    PITCHER_DOMINANT = "Pitcher-Dominant"
    BULLPEN_FRAGILE = "Bullpen-Fragile"
    RUN_SUPPRESSING = "Run-Suppressing"
    HR_INFLATED = "HR-Inflated"
    SB_FRIENDLY = "Stolen-Base Friendly"


class VolatilityProfile(Enum):
    FLOOR = "FLOOR"
    HYBRID = "HYBRID"
    CEILING = "CEILING"


# ══════════════════════════════════════════════════════════════════════════════
# DATA CLASSES — STRUCTURED INPUTS
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class PitcherProfile:
    player_id: int
    name: str
    team: str
    throws: str  # "L" or "R"
    role: Optional[SPRole] = None
    # Recent stats (L5/L10/season)
    k_pct: float = 0.0
    bb_pct: float = 0.0
    csw_pct: float = 0.0
    swstr_pct: float = 0.0
    gb_pct: float = 0.0
    fb_pct: float = 0.0
    hr_per_fb: float = 0.0
    era: float = 0.0
    fip: float = 0.0
    # Pitch count / leash
    avg_pitch_count_l5: float = 0.0
    avg_batters_faced_l5: float = 0.0
    avg_ip_l5: float = 0.0
    pitch_count_trend: list = field(default_factory=list)  # last 5 pitch counts
    # TTO efficiency (wOBA by pass)
    tto_1st: float = 0.0
    tto_2nd: float = 0.0
    tto_3rd: float = 0.0
    # Pitch mix percentages
    pitch_mix: dict = field(default_factory=dict)  # {"FF": 0.45, "SL": 0.30, ...}
    # Handedness splits
    vs_lhb_k_pct: float = 0.0
    vs_rhb_k_pct: float = 0.0
    vs_lhb_woba: float = 0.0
    vs_rhb_woba: float = 0.0
    # Leash stability score
    lss: int = 0
    lss_confidence: Confidence = Confidence.YELLOW
    # Volatility
    volatility_profile: VolatilityProfile = VolatilityProfile.HYBRID
    # Injury flags
    injury_flag: bool = False
    first_start_back: bool = False


@dataclass
class HitterProfile:
    player_id: int
    name: str
    team: str
    bats: str  # "L", "R", "S"
    lineup_slot: int = 0
    role: Optional[HitterRole] = None
    # Recent stats
    k_pct: float = 0.0
    bb_pct: float = 0.0
    iso: float = 0.0
    babip: float = 0.0
    woba: float = 0.0
    wrc_plus: float = 0.0
    # Statcast
    avg_exit_velo: float = 0.0
    barrel_pct: float = 0.0
    hard_hit_pct: float = 0.0
    sweet_spot_pct: float = 0.0
    chase_pct: float = 0.0
    whiff_pct: float = 0.0
    # Speed / SB
    sprint_speed: float = 0.0
    sb_attempts: int = 0
    sb_success_pct: float = 0.0
    # Handedness splits
    vs_lhp_woba: float = 0.0
    vs_rhp_woba: float = 0.0
    vs_lhp_k_pct: float = 0.0
    vs_rhp_k_pct: float = 0.0
    # PA stability
    expected_pa: float = 0.0
    pss: int = 0
    pss_confidence: Confidence = Confidence.YELLOW
    # Volatility
    volatility_profile: VolatilityProfile = VolatilityProfile.HYBRID
    # Risk flags
    platoon_risk: bool = False
    pinch_hit_risk: bool = False
    confirmed_starter: bool = False


@dataclass
class BullpenProfile:
    team: str
    role: Optional[BullpenRole] = None
    aggregate_k_pct: float = 0.0
    aggregate_bb_pct: float = 0.0
    aggregate_era: float = 0.0
    # Fatigue tracking
    ip_last_3_days: float = 0.0
    high_leverage_available: bool = True
    closer_available: bool = True
    arms_used_yesterday: int = 0
    # Derived
    fatigue_score: float = 0.0  # 0-10, higher = more tired
    strength_score: float = 0.0  # 0-10, higher = better


@dataclass
class TeamOffenseProfile:
    team: str
    team_id: int
    archetype: Optional[TeamOffenseArchetype] = None
    team_k_pct: float = 0.0
    team_bb_pct: float = 0.0
    team_iso: float = 0.0
    team_woba: float = 0.0
    team_sb_attempts: int = 0
    runs_per_game: float = 0.0
    # vs handedness
    vs_lhp_woba: float = 0.0
    vs_rhp_woba: float = 0.0


@dataclass
class GameEnvironment:
    game_pk: int
    date: str
    venue_name: str
    venue_id: int
    home_team: str
    away_team: str
    home_team_id: int
    away_team_id: int
    # Pitchers
    home_sp: Optional[PitcherProfile] = None
    away_sp: Optional[PitcherProfile] = None
    # Bullpens
    home_bullpen: Optional[BullpenProfile] = None
    away_bullpen: Optional[BullpenProfile] = None
    # Lineups
    home_lineup: list = field(default_factory=list)  # List[HitterProfile]
    away_lineup: list = field(default_factory=list)
    lineup_confirmed: bool = False
    # Team offense
    home_offense: Optional[TeamOffenseProfile] = None
    away_offense: Optional[TeamOffenseProfile] = None
    # Park
    park_factor_runs: float = 1.0
    park_factor_hr: float = 1.0
    park_factor_h: float = 1.0
    # Weather
    temperature: float = 72.0
    wind_speed: float = 5.0
    wind_direction: str = "Out to CF"
    precipitation_pct: float = 0.0
    roof_closed: bool = False
    # Umpire
    umpire_name: str = ""
    umpire_k_impact: float = 0.0  # + = more Ks than expected
    umpire_bb_impact: float = 0.0
    umpire_zone_type: str = "neutral"  # "wide", "tight", "neutral"
    # Script
    script_tag: ScriptTag = ScriptTag.STABLE
    f5_projected_runs_home: float = 0.0
    f5_projected_runs_away: float = 0.0
    fg_projected_runs_home: float = 0.0
    fg_projected_runs_away: float = 0.0
    # Travel / rest
    home_rest_days: int = 0
    away_rest_days: int = 0
    is_getaway: bool = False
    is_day_after_night: bool = False
    # Confidence gates
    lineup_certainty: Confidence = Confidence.YELLOW
    weather_risk: Confidence = Confidence.GREEN
    volatility_score: float = 0.0


# ══════════════════════════════════════════════════════════════════════════════
# VENUE COORDINATES — For weather lookups
# ══════════════════════════════════════════════════════════════════════════════

VENUE_COORDS = {
    # venue_id: (lat, lon, has_retractable_roof)
    1: (33.4453, -112.0667, True),    # Chase Field (ARI)
    4: (33.8903, -84.4681, False),    # Truist Park (ATL)
    2: (39.2838, -76.6216, False),    # Camden Yards (BAL)
    3: (42.3467, -71.0972, False),    # Fenway Park (BOS)
    17: (41.9484, -87.6553, False),   # Wrigley Field (CHC)
    4705: (41.8300, -87.6339, False), # Guaranteed Rate (CWS)
    2602: (39.0974, -84.5082, False), # Great American (CIN)
    5: (41.4962, -81.6852, False),    # Progressive Field (CLE)
    19: (39.7561, -104.9942, False),  # Coors Field (COL)
    2394: (42.3390, -83.0485, False), # Comerica Park (DET)
    2392: (29.7573, -95.3555, True),  # Minute Maid (HOU)
    7: (39.0517, -94.4803, False),    # Kauffman Stadium (KC)
    1727: (34.0739, -118.2400, False),# Dodger Stadium (LAD)
    1: (33.8003, -117.8827, False),   # Angel Stadium (LAA)
    32: (25.7781, -80.2197, True),    # loanDepot Park (MIA)
    3312: (43.0280, -87.9712, True),  # American Family (MIL)
    3313: (44.9817, -93.2776, False), # Target Field (MIN)
    3289: (40.7571, -73.8458, False), # Citi Field (NYM)
    3313: (40.8296, -73.9262, False), # Yankee Stadium (NYY)
    10: (37.7516, -122.2005, False),  # Oakland Coliseum (OAK)
    2681: (39.9061, -75.1665, False), # Citizens Bank (PHI)
    31: (40.4468, -79.9583, False),   # PNC Park (PIT)
    15: (32.7076, -117.1570, False),  # Petco Park (SD)
    680: (37.7786, -122.3893, False), # Oracle Park (SF)
    680: (47.5914, -122.3325, True),  # T-Mobile Park (SEA)
    2889: (38.6226, -90.1928, False), # Busch Stadium (STL)
    12: (27.7682, -82.6534, True),    # Tropicana Field (TB)
    13: (32.7512, -97.0832, True),    # Globe Life Field (TEX)
    14: (43.6414, -79.3894, True),    # Rogers Centre (TOR)
    3309: (38.8730, -77.0074, False), # Nationals Park (WSH)
}


# ══════════════════════════════════════════════════════════════════════════════
# PARK FACTORS — Static reference (update monthly during season)
# Indexed by venue_id: {runs, hr, h, 2b, 3b, bb, k, lhb_hr, rhb_hr}
# ══════════════════════════════════════════════════════════════════════════════

PARK_FACTORS = {
    # Coors Field — extreme hitter's park
    19: {"runs": 1.25, "hr": 1.15, "h": 1.15, "2b": 1.30, "3b": 1.50, "bb": 1.0, "k": 0.92},
    # Default — neutral
    "default": {"runs": 1.0, "hr": 1.0, "h": 1.0, "2b": 1.0, "3b": 1.0, "bb": 1.0, "k": 1.0},
}


# ══════════════════════════════════════════════════════════════════════════════
# API FETCHER — Cached HTTP client for MLB Stats API
# ══════════════════════════════════════════════════════════════════════════════

class APIFetcher:
    """Handles all HTTP requests with caching and rate limiting."""

    def __init__(self, cache_dir: str = CACHE_DIR, cache_ttl_hours: int = 4):
        self.cache_dir = cache_dir
        self.cache_ttl = timedelta(hours=cache_ttl_hours)
        os.makedirs(cache_dir, exist_ok=True)
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "RocketMLB/1.0"})

    def _cache_key(self, url: str, params: dict) -> str:
        raw = url + json.dumps(params, sort_keys=True)
        return hashlib.md5(raw.encode()).hexdigest()

    def _cache_path(self, key: str) -> str:
        return os.path.join(self.cache_dir, f"{key}.json")

    def get(self, url: str, params: dict = None, cache_ttl: timedelta = None) -> dict:
        """GET with file-based caching."""
        params = params or {}
        ttl = cache_ttl or self.cache_ttl
        key = self._cache_key(url, params)
        path = self._cache_path(key)

        # Check cache
        if os.path.exists(path):
            age = datetime.now() - datetime.fromtimestamp(os.path.getmtime(path))
            if age < ttl:
                with open(path, "r") as f:
                    return json.load(f)

        # Fetch
        try:
            resp = self.session.get(url, params=params, timeout=15)
            resp.raise_for_status()
            data = resp.json()
            # Write cache
            with open(path, "w") as f:
                json.dump(data, f)
            time.sleep(0.25)  # Rate limiting courtesy
            return data
        except requests.RequestException as e:
            logger.error(f"API request failed: {url} — {e}")
            # Return cached data even if stale
            if os.path.exists(path):
                with open(path, "r") as f:
                    return json.load(f)
            return {}

    def clear_cache(self):
        """Wipe all cached files."""
        for f in os.listdir(self.cache_dir):
            os.remove(os.path.join(self.cache_dir, f))


# ══════════════════════════════════════════════════════════════════════════════
# MLB STATS API CLIENT
# ══════════════════════════════════════════════════════════════════════════════

class MLBStatsClient:
    """Wraps the free MLB Stats API (statsapi.mlb.com/api/v1)."""

    def __init__(self, fetcher: APIFetcher):
        self.fetcher = fetcher
        self.base = MLB_API_BASE

    # ── Schedule ──────────────────────────────────────────────────────────

    def get_schedule(self, date: str, sport_id: int = 1) -> list:
        """
        Fetch today's games with probable pitchers.
        Returns list of game dicts with gamePk, teams, venue, probables.
        """
        data = self.fetcher.get(
            f"{self.base}/schedule",
            params={
                "sportId": sport_id,
                "date": date,
                "hydrate": "probablePitcher,venue,team,linescore,weather",
            },
        )
        games = []
        for date_entry in data.get("dates", []):
            for game in date_entry.get("games", []):
                games.append(game)
        return games

    # ── Live Feed (lineups, play-by-play, weather) ────────────────────────

    def get_live_feed(self, game_pk: int) -> dict:
        """Full live game feed — lineups, plays, weather, umpires."""
        return self.fetcher.get(f"{self.base}/game/{game_pk}/feed/live")

    # ── Boxscore ──────────────────────────────────────────────────────────

    def get_boxscore(self, game_pk: int) -> dict:
        """Full boxscore — batting and pitching lines."""
        return self.fetcher.get(f"{self.base}/game/{game_pk}/boxscore")

    # ── Player Stats ──────────────────────────────────────────────────────

    def get_player_stats(
        self, player_id: int, stats: str = "season", group: str = "pitching",
        season: int = None
    ) -> dict:
        """
        Fetch player stats.
        stats: "season", "career", "gameLog", "byDateRange", "vsTeam"
        group: "pitching", "hitting", "fielding"
        """
        season = season or datetime.now().year
        return self.fetcher.get(
            f"{self.base}/people/{player_id}/stats",
            params={"stats": stats, "group": group, "season": season},
        )

    def get_player_game_log(self, player_id: int, group: str = "pitching",
                            season: int = None) -> list:
        """Get game-by-game log for a player this season."""
        season = season or datetime.now().year
        data = self.fetcher.get(
            f"{self.base}/people/{player_id}/stats",
            params={"stats": "gameLog", "group": group, "season": season},
        )
        splits = []
        for stat_block in data.get("stats", []):
            splits.extend(stat_block.get("splits", []))
        return splits

    # ── Player Info ───────────────────────────────────────────────────────

    def get_player_info(self, player_id: int) -> dict:
        """Basic player info — name, position, throws, bats."""
        data = self.fetcher.get(f"{self.base}/people/{player_id}")
        people = data.get("people", [])
        return people[0] if people else {}

    # ── Roster ────────────────────────────────────────────────────────────

    def get_roster(self, team_id: int, roster_type: str = "active") -> list:
        """Get team roster. Types: active, fullSeason, 40Man, depthChart."""
        data = self.fetcher.get(
            f"{self.base}/teams/{team_id}/roster",
            params={"rosterType": roster_type},
        )
        return data.get("roster", [])

    # ── Team Stats ────────────────────────────────────────────────────────

    def get_team_stats(self, team_id: int, group: str = "hitting",
                       season: int = None) -> dict:
        """Aggregated team stats for the season."""
        season = season or datetime.now().year
        return self.fetcher.get(
            f"{self.base}/teams/{team_id}/stats",
            params={"stats": "season", "group": group, "season": season},
        )


# ══════════════════════════════════════════════════════════════════════════════
# WEATHER CLIENT
# ══════════════════════════════════════════════════════════════════════════════

class WeatherClient:
    """Fetch hourly weather from Open-Meteo (free, no auth)."""

    def __init__(self, fetcher: APIFetcher):
        self.fetcher = fetcher

    def get_forecast(self, lat: float, lon: float, game_hour: int = 19) -> dict:
        """
        Get weather at game time.
        game_hour: local hour of first pitch (default 7pm).
        Returns: {temperature_f, wind_speed_mph, wind_direction_deg, precip_pct}
        """
        data = self.fetcher.get(
            OPEN_METEO_BASE,
            params={
                "latitude": lat,
                "longitude": lon,
                "hourly": "temperature_2m,windspeed_10m,winddirection_10m,precipitation_probability",
                "temperature_unit": "fahrenheit",
                "windspeed_unit": "mph",
                "forecast_days": 1,
            },
            cache_ttl=timedelta(hours=2),
        )
        hourly = data.get("hourly", {})
        temps = hourly.get("temperature_2m", [])
        winds = hourly.get("windspeed_10m", [])
        wind_dirs = hourly.get("winddirection_10m", [])
        precips = hourly.get("precipitation_probability", [])

        idx = min(game_hour, len(temps) - 1) if temps else 0
        return {
            "temperature_f": temps[idx] if temps else 72.0,
            "wind_speed_mph": winds[idx] if winds else 5.0,
            "wind_direction_deg": wind_dirs[idx] if wind_dirs else 0,
            "precipitation_pct": precips[idx] if precips else 0,
        }


# ══════════════════════════════════════════════════════════════════════════════
# ROLE TAGGING ENGINE
# ══════════════════════════════════════════════════════════════════════════════

class RoleEngine:
    """
    Tags players and teams with their structural roles.
    This is YOUR proprietary logic — the rules below are starting points.
    Tune thresholds based on your own analysis.
    """

    # ── SP Role Classification ────────────────────────────────────────────

    @staticmethod
    def classify_sp(p: PitcherProfile) -> SPRole:
        """Assign primary role to a starting pitcher."""

        # Walk-Risk Arm: BB% >= 10%
        if p.bb_pct >= 0.10:
            return SPRole.WALK_RISK_ARM

        # K Specialist: K% >= 28% and reasonable walk rate
        if p.k_pct >= 0.28 and p.bb_pct < 0.08:
            return SPRole.K_SPECIALIST

        # Workhorse Ace: high IP, stable pitch counts, low FIP
        if p.avg_ip_l5 >= 6.0 and p.fip < 3.50 and min(p.pitch_count_trend or [0]) >= 85:
            return SPRole.WORKHORSE_ACE

        # Ground-Ball Suppressor: GB% >= 50%
        if p.gb_pct >= 0.50:
            return SPRole.GROUNDBALL_SUPPRESSOR

        # Fly-Ball Fragile: FB% >= 40% and HR/FB >= 12%
        if p.fb_pct >= 0.40 and p.hr_per_fb >= 0.12:
            return SPRole.FLYBALL_FRAGILE

        # Short-Leash Volatile: low avg IP, inconsistent pitch counts
        if p.avg_ip_l5 < 5.0:
            return SPRole.SHORT_LEASH_VOLATILE

        # Contact Manager: low K, low BB, decent results
        if p.k_pct < 0.20 and p.bb_pct < 0.07:
            return SPRole.CONTACT_MANAGER

        # Default fallback
        return SPRole.CONTACT_MANAGER

    # ── Hitter Role Classification ────────────────────────────────────────

    @staticmethod
    def classify_hitter(h: HitterProfile) -> HitterRole:
        """Assign primary role to a hitter."""

        # SB Pressure Bat: high speed + active stealer
        if h.sprint_speed >= 28.0 and h.sb_attempts >= 15:
            return HitterRole.SB_PRESSURE

        # K-Prone Slugger: high K% + high ISO
        if h.k_pct >= 0.28 and h.iso >= 0.200:
            return HitterRole.K_PRONE_SLUGGER

        # Boom/Bust Barrel Bat: very high barrel%, high K%
        if h.barrel_pct >= 12.0 and h.k_pct >= 0.25:
            return HitterRole.BOOM_BUST_BARREL

        # Lift-and-Pull Power: high ISO, moderate K
        if h.iso >= 0.220 and h.k_pct < 0.28:
            return HitterRole.LIFT_PULL_POWER

        # Table Setter: high OBP proxy (low K, decent BB, speed)
        if h.bb_pct >= 0.10 and h.sprint_speed >= 27.0:
            return HitterRole.TABLE_SETTER

        # Passive Walker: very high BB%, low aggression
        if h.bb_pct >= 0.12:
            return HitterRole.PASSIVE_WALKER

        # Gap / XBH Bat: moderate power, moderate K
        if h.iso >= 0.150 and h.k_pct < 0.22:
            return HitterRole.GAP_XBH

        # Run Producer: middle-order bat with RBI opportunity
        if h.lineup_slot in [3, 4, 5] and h.iso >= 0.170:
            return HitterRole.RUN_PRODUCER

        # Platoon Masher: big split differential
        if h.bats in ("L", "R"):
            weak_side = h.vs_lhp_woba if h.bats == "L" else h.vs_rhp_woba
            strong_side = h.vs_rhp_woba if h.bats == "L" else h.vs_lhp_woba
            if strong_side > 0 and weak_side > 0 and (strong_side - weak_side) >= 0.050:
                return HitterRole.PLATOON_MASHER

        # Contact Bat: low K, low power
        if h.k_pct < 0.18:
            return HitterRole.CONTACT_BAT

        return HitterRole.CONTACT_BAT

    # ── Bullpen Role Classification ───────────────────────────────────────

    @staticmethod
    def classify_bullpen(bp: BullpenProfile) -> BullpenRole:
        """Assign role to a team's bullpen unit."""

        if bp.fatigue_score >= 7.0:
            return BullpenRole.OVERWORKED

        if bp.aggregate_k_pct >= 0.28 and bp.aggregate_era < 3.50:
            if bp.closer_available and bp.high_leverage_available:
                return BullpenRole.LOCKDOWN
            return BullpenRole.HIGH_K_BRIDGE

        if bp.closer_available and not bp.high_leverage_available:
            return BullpenRole.CLOSER_HEAVY

        if bp.aggregate_era >= 4.50:
            return BullpenRole.LOW_LEV_BLEED

        if not bp.closer_available:
            return BullpenRole.COMMITTEE_CHAOS

        return BullpenRole.VOLATILE_MIDDLE

    # ── Team Offense Archetype ────────────────────────────────────────────

    @staticmethod
    def classify_team_offense(t: TeamOffenseProfile) -> TeamOffenseArchetype:
        """Assign archetype to a team's offense."""

        if t.team_k_pct >= 0.24:
            return TeamOffenseArchetype.HIGH_K

        if t.team_bb_pct >= 0.10:
            return TeamOffenseArchetype.PATIENT_WALK

        if t.team_iso >= 0.180:
            return TeamOffenseArchetype.POWER_CLUSTER

        if t.team_k_pct < 0.19 and t.team_bb_pct < 0.08:
            return TeamOffenseArchetype.CONTACT_CHAIN

        if t.team_sb_attempts >= 100:  # season pace
            return TeamOffenseArchetype.AGGRESSIVE_BASEPATH

        return TeamOffenseArchetype.CONTACT_CHAIN

    # ── Leash Stability Gate (LSG) ────────────────────────────────────────

    @staticmethod
    def compute_lss(p: PitcherProfile, opponent: TeamOffenseProfile = None,
                    weather_stress: bool = False) -> int:
        """
        Leash Stability Score per MODULE 2 of your engine doc.
        Returns integer score. >= 2 GREEN, 0-1 YELLOW, <= -1 RED.
        """
        lss = 0

        # Role / Usage
        if p.role == SPRole.WORKHORSE_ACE:
            lss += 2
        elif p.role in (SPRole.K_SPECIALIST, SPRole.GROUNDBALL_SUPPRESSOR, SPRole.CONTACT_MANAGER):
            lss += 1
        elif p.role == SPRole.SHORT_LEASH_VOLATILE:
            lss -= 2
        # Opener/bulk ambiguity would be -3 (detected elsewhere)

        # Pitch Count Trend
        pc = p.pitch_count_trend
        if pc:
            avg_pc = sum(pc) / len(pc)
            pc_std = (sum((x - avg_pc) ** 2 for x in pc) / len(pc)) ** 0.5
            if avg_pc >= 95 and pc_std < 10:
                lss += 2
            elif avg_pc >= 85 and pc_std < 12:
                lss += 1
            elif avg_pc < 80:
                lss -= 2
            elif pc_std > 15:
                lss -= 1

        # Injury / Return
        if p.first_start_back:
            lss -= 3
        elif p.injury_flag:
            lss -= 1

        # Matchup Stress
        if opponent:
            if opponent.team_bb_pct >= 0.10:
                lss -= 1
            if opponent.archetype == TeamOffenseArchetype.PATIENT_WALK:
                lss -= 1
        if weather_stress:
            lss -= 1

        return lss

    # ── PA Stability Gate (PSG) ───────────────────────────────────────────

    @staticmethod
    def compute_pss(h: HitterProfile, game_env_high_total: bool = False) -> int:
        """
        PA Stability Score per MODULE 3 of your engine doc.
        Returns integer score. >= 2 GREEN, 0-1 YELLOW, <= -1 RED.
        """
        pss = 0

        # Lineup Slot
        if h.lineup_slot <= 5:
            pss += 2
        elif h.lineup_slot <= 7:
            pss += 1
        else:
            pss -= 1

        # Start Certainty
        if h.confirmed_starter:
            pss += 2
        elif h.platoon_risk or h.pinch_hit_risk:
            pss -= 2

        # Game Environment
        if game_env_high_total:
            pss += 1
        # Low-run fragile would be -1

        # Role
        if h.role in (HitterRole.TABLE_SETTER, HitterRole.CONTACT_BAT,
                       HitterRole.LIFT_PULL_POWER, HitterRole.RUN_PRODUCER):
            pss += 1
        elif h.role == HitterRole.PLATOON_MASHER:
            pss -= 2

        return pss

    # ── Player Volatility Profile (PVP) ──────────────────────────────────

    @staticmethod
    def classify_volatility_sp(p: PitcherProfile) -> VolatilityProfile:
        if p.role == SPRole.WORKHORSE_ACE:
            return VolatilityProfile.FLOOR
        elif p.role in (SPRole.K_SPECIALIST, SPRole.GROUNDBALL_SUPPRESSOR):
            return VolatilityProfile.HYBRID
        elif p.role in (SPRole.FLYBALL_FRAGILE, SPRole.SHORT_LEASH_VOLATILE):
            return VolatilityProfile.CEILING
        return VolatilityProfile.HYBRID

    @staticmethod
    def classify_volatility_hitter(h: HitterProfile) -> VolatilityProfile:
        if h.role in (HitterRole.TABLE_SETTER, HitterRole.CONTACT_BAT):
            return VolatilityProfile.HYBRID
        elif h.role in (HitterRole.K_PRONE_SLUGGER, HitterRole.BOOM_BUST_BARREL):
            return VolatilityProfile.CEILING
        elif h.role in (HitterRole.LIFT_PULL_POWER, HitterRole.GAP_XBH):
            return VolatilityProfile.HYBRID
        return VolatilityProfile.HYBRID


# ══════════════════════════════════════════════════════════════════════════════
# STATCAST CLIENT (via pybaseball)
# ══════════════════════════════════════════════════════════════════════════════

class StatcastClient:
    """
    Wraps pybaseball for Statcast / FanGraphs data.
    Requires: pip install pybaseball
    Note: pybaseball scrapes Baseball Savant — be respectful with volume.
    """

    @staticmethod
    def get_pitcher_statcast(player_id: int, days_back: int = 30) -> pd.DataFrame:
        """Fetch pitch-level Statcast data for a pitcher."""
        try:
            from pybaseball import statcast_pitcher
            end = datetime.now().strftime("%Y-%m-%d")
            start = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")
            df = statcast_pitcher(start, end, player_id)
            return df
        except ImportError:
            logger.warning("pybaseball not installed — Statcast data unavailable")
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"Statcast pitcher fetch failed for {player_id}: {e}")
            return pd.DataFrame()

    @staticmethod
    def get_batter_statcast(player_id: int, days_back: int = 30) -> pd.DataFrame:
        """Fetch pitch-level Statcast data for a batter."""
        try:
            from pybaseball import statcast_batter
            end = datetime.now().strftime("%Y-%m-%d")
            start = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")
            df = statcast_batter(start, end, player_id)
            return df
        except ImportError:
            logger.warning("pybaseball not installed — Statcast data unavailable")
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"Statcast batter fetch failed for {player_id}: {e}")
            return pd.DataFrame()

    @staticmethod
    def compute_pitcher_metrics(df: pd.DataFrame) -> dict:
        """
        Derive Rocket-relevant metrics from raw Statcast pitch data.
        Returns dict of computed values.
        """
        if df.empty:
            return {}

        total_pitches = len(df)
        swinging_strikes = len(df[df["description"].isin([
            "swinging_strike", "swinging_strike_blocked", "foul_tip"
        ])])
        called_strikes = len(df[df["description"] == "called_strike"])
        all_strikes = len(df[df["type"] == "S"])

        # CSW% = (Called Strikes + Swinging Strikes) / Total Pitches
        csw_pct = (called_strikes + swinging_strikes) / total_pitches if total_pitches else 0

        # SwStr% = Swinging Strikes / Total Pitches
        swstr_pct = swinging_strikes / total_pitches if total_pitches else 0

        # Pitch mix
        pitch_mix = {}
        if total_pitches:
            for pt, count in df["pitch_type"].value_counts().items():
                if pd.notna(pt):
                    pitch_mix[pt] = round(count / total_pitches, 3)

        # Ground ball / fly ball (from batted balls)
        batted = df[df["bb_type"].notna()]
        gb_count = len(batted[batted["bb_type"] == "ground_ball"])
        fb_count = len(batted[batted["bb_type"] == "fly_ball"])
        total_bb = len(batted)
        gb_pct = gb_count / total_bb if total_bb else 0
        fb_pct = fb_count / total_bb if total_bb else 0

        # TTO efficiency — approximate by inning grouping
        # 1st-3rd inning ≈ 1st TTO, 4th-6th ≈ 2nd, 7th+ ≈ 3rd
        tto_data = {}
        for label, innings in [("1st", [1, 2, 3]), ("2nd", [4, 5, 6]), ("3rd", [7, 8, 9])]:
            subset = df[df["inning"].isin(innings)]
            events = subset[subset["events"].notna()]
            if len(events) > 0:
                # Use estimated wOBA as proxy
                woba_col = "estimated_woba_using_speedangle"
                if woba_col in events.columns:
                    tto_data[label] = events[woba_col].mean()

        return {
            "csw_pct": round(csw_pct, 4),
            "swstr_pct": round(swstr_pct, 4),
            "pitch_mix": pitch_mix,
            "gb_pct": round(gb_pct, 4),
            "fb_pct": round(fb_pct, 4),
            "tto_1st_woba": round(tto_data.get("1st", 0), 4),
            "tto_2nd_woba": round(tto_data.get("2nd", 0), 4),
            "tto_3rd_woba": round(tto_data.get("3rd", 0), 4),
            "total_pitches_sample": total_pitches,
        }

    @staticmethod
    def compute_batter_metrics(df: pd.DataFrame) -> dict:
        """Derive Rocket-relevant metrics from raw Statcast batter data."""
        if df.empty:
            return {}

        total_pitches = len(df)
        swinging_strikes = len(df[df["description"].isin([
            "swinging_strike", "swinging_strike_blocked"
        ])])
        whiff_swings = len(df[df["description"].isin([
            "swinging_strike", "swinging_strike_blocked", "foul", "foul_tip",
            "hit_into_play", "hit_into_play_no_out", "hit_into_play_score"
        ])])
        whiff_pct = swinging_strikes / whiff_swings if whiff_swings else 0

        # Chase rate (pitches outside zone that batter swung at)
        outside_zone = df[df["zone"].isin([11, 12, 13, 14])] if "zone" in df.columns else pd.DataFrame()
        if len(outside_zone) > 0:
            chased = outside_zone[outside_zone["description"].str.contains("swing|foul|hit_into", na=False)]
            chase_pct = len(chased) / len(outside_zone)
        else:
            chase_pct = 0

        # Exit velo / barrel
        batted = df[df["launch_speed"].notna()]
        avg_ev = batted["launch_speed"].mean() if len(batted) else 0
        barrel_count = len(batted[
            (batted["launch_speed"] >= 98) & (batted["launch_angle"].between(26, 30))
        ]) if len(batted) else 0
        barrel_pct = barrel_count / len(batted) * 100 if len(batted) else 0
        hard_hit = len(batted[batted["launch_speed"] >= 95]) / len(batted) * 100 if len(batted) else 0

        return {
            "whiff_pct": round(whiff_pct, 4),
            "chase_pct": round(chase_pct, 4),
            "avg_exit_velo": round(avg_ev, 1),
            "barrel_pct": round(barrel_pct, 1),
            "hard_hit_pct": round(hard_hit, 1),
            "total_pitches_sample": total_pitches,
        }


# ══════════════════════════════════════════════════════════════════════════════
# LINE ERROR DETECTOR (LED) — Module 1
# ══════════════════════════════════════════════════════════════════════════════

class LineErrorDetector:
    """
    Detect sportsbook mispricing before simulation.
    Compares recency-weighted hit rates to implied probability from odds.
    """

    @staticmethod
    def american_to_implied(odds: int) -> float:
        """Convert American odds to implied probability."""
        if odds < 0:
            return abs(odds) / (abs(odds) + 100)
        else:
            return 100 / (odds + 100)

    @staticmethod
    def compute_edge(p_true: float, odds: int) -> dict:
        """
        Compute edge % and LED flag.
        Returns: {p_implied, p_true, edge_pct, led_flag}
        """
        p_implied = LineErrorDetector.american_to_implied(odds)
        edge_pct = (p_true - p_implied) * 100

        if edge_pct >= 8.0:
            flag = Confidence.GREEN
        elif edge_pct >= 4.0:
            flag = Confidence.YELLOW
        else:
            flag = Confidence.RED

        return {
            "p_implied": round(p_implied, 4),
            "p_true": round(p_true, 4),
            "edge_pct": round(edge_pct, 2),
            "led_flag": flag,
        }

    @staticmethod
    def recency_weighted_rate(l5_rate: float, l10_rate: float,
                               broader_rate: float) -> float:
        """
        Weighted hit rate per your doc:
        L5 = 50%, L10 = 35%, broader = 15%
        """
        return (l5_rate * 0.50) + (l10_rate * 0.35) + (broader_rate * 0.15)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN ORCHESTRATOR — DAILY PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

class RocketDataLayer:
    """
    Main orchestrator for the Rocket MLB daily pipeline.

    Usage:
        rdl = RocketDataLayer()
        slate = rdl.run_daily_pipeline("2026-06-15")
        # slate is a list of GameEnvironment objects, fully populated
    """

    def __init__(self):
        self.fetcher = APIFetcher()
        self.mlb = MLBStatsClient(self.fetcher)
        self.weather = WeatherClient(self.fetcher)
        self.statcast = StatcastClient()
        self.roles = RoleEngine()
        self.led = LineErrorDetector()

    def run_daily_pipeline(self, date: str) -> list:
        """
        Execute the full daily pipeline for a given date.
        Returns list of GameEnvironment objects.
        """
        logger.info(f"═══ ROCKET MLB PIPELINE — {date} ═══")

        # ── PHASE 1: Schedule Fetch ───────────────────────────────────────
        logger.info("PHASE 1: Fetching schedule...")
        games_raw = self.mlb.get_schedule(date)
        logger.info(f"  Found {len(games_raw)} games")

        environments = []

        for game_raw in games_raw:
            try:
                env = self._build_game_environment(game_raw, date)
                environments.append(env)
            except Exception as e:
                game_pk = game_raw.get("gamePk", "?")
                logger.error(f"  Failed to build environment for gamePk {game_pk}: {e}")

        # ── PHASE 2: Role Tagging ─────────────────────────────────────────
        logger.info("PHASE 2: Role tagging...")
        for env in environments:
            self._tag_roles(env)

        # ── PHASE 3: Stability Gates ──────────────────────────────────────
        logger.info("PHASE 3: Running stability gates...")
        for env in environments:
            self._run_stability_gates(env)

        # ── PHASE 4: Script Tagging ───────────────────────────────────────
        logger.info("PHASE 4: Tagging game scripts...")
        for env in environments:
            self._tag_script(env)

        logger.info(f"═══ PIPELINE COMPLETE — {len(environments)} games processed ═══")
        return environments

    def _build_game_environment(self, game_raw: dict, date: str) -> GameEnvironment:
        """Build a GameEnvironment from raw schedule data."""
        game_pk = game_raw["gamePk"]
        venue = game_raw.get("venue", {})
        home = game_raw.get("teams", {}).get("home", {})
        away = game_raw.get("teams", {}).get("away", {})

        env = GameEnvironment(
            game_pk=game_pk,
            date=date,
            venue_name=venue.get("name", ""),
            venue_id=venue.get("id", 0),
            home_team=home.get("team", {}).get("abbreviation", ""),
            away_team=away.get("team", {}).get("abbreviation", ""),
            home_team_id=home.get("team", {}).get("id", 0),
            away_team_id=away.get("team", {}).get("id", 0),
        )

        # Probable pitchers
        home_prob = home.get("probablePitcher", {})
        away_prob = away.get("probablePitcher", {})
        if home_prob.get("id"):
            env.home_sp = self._build_pitcher_profile(home_prob["id"], env.home_team)
        if away_prob.get("id"):
            env.away_sp = self._build_pitcher_profile(away_prob["id"], env.away_team)

        # Park factors
        pf = PARK_FACTORS.get(env.venue_id, PARK_FACTORS["default"])
        env.park_factor_runs = pf["runs"]
        env.park_factor_hr = pf["hr"]
        env.park_factor_h = pf["h"]

        # Weather
        coords = VENUE_COORDS.get(env.venue_id)
        if coords:
            lat, lon, has_roof = coords[0], coords[1], coords[2]
            wx = self.weather.get_forecast(lat, lon)
            env.temperature = wx["temperature_f"]
            env.wind_speed = wx["wind_speed_mph"]
            env.precipitation_pct = wx["precipitation_pct"]
            if has_roof and (wx["precipitation_pct"] > 50 or wx["temperature_f"] < 50):
                env.roof_closed = True

        # Weather risk flag
        if env.precipitation_pct > 40:
            env.weather_risk = Confidence.YELLOW
        if env.precipitation_pct > 70:
            env.weather_risk = Confidence.RED

        return env

    def _build_pitcher_profile(self, player_id: int, team: str) -> PitcherProfile:
        """Fetch and build a PitcherProfile from API data."""
        info = self.mlb.get_player_info(player_id)
        profile = PitcherProfile(
            player_id=player_id,
            name=info.get("fullName", f"ID:{player_id}"),
            team=team,
            throws=info.get("pitchHand", {}).get("code", "R"),
        )

        # Season stats
        season_data = self.mlb.get_player_stats(player_id, "season", "pitching")
        for stat_block in season_data.get("stats", []):
            for split in stat_block.get("splits", []):
                s = split.get("stat", {})
                ip = float(s.get("inningsPitched", "0").replace(".", ""))
                # Convert innings pitched string (e.g., "120.1" → actual)
                ip_str = s.get("inningsPitched", "0")
                if "." in ip_str:
                    parts = ip_str.split(".")
                    ip = int(parts[0]) + int(parts[1]) / 3
                else:
                    ip = float(ip_str)

                bf = int(s.get("battersFaced", 0)) or 1
                profile.k_pct = int(s.get("strikeOuts", 0)) / bf
                profile.bb_pct = int(s.get("baseOnBalls", 0)) / bf
                profile.era = float(s.get("era", "0"))
                # GB/FB from detailed stats if available
                profile.gb_pct = float(s.get("groundOutsToAirouts", "1.0")) / 2  # rough proxy
                break

        # Game log (last 5 starts for pitch count trend)
        game_log = self.mlb.get_player_game_log(player_id, "pitching")
        recent = game_log[:5]  # Most recent first
        pitch_counts = []
        ips = []
        bfs = []
        for g in recent:
            s = g.get("stat", {})
            pc = int(s.get("numberOfPitches", 0))
            if pc > 0:
                pitch_counts.append(pc)
            ip_str = s.get("inningsPitched", "0")
            if "." in ip_str:
                parts = ip_str.split(".")
                ip_val = int(parts[0]) + int(parts[1]) / 3
            else:
                ip_val = float(ip_str)
            ips.append(ip_val)
            bfs.append(int(s.get("battersFaced", 0)))

        profile.pitch_count_trend = pitch_counts
        profile.avg_pitch_count_l5 = sum(pitch_counts) / len(pitch_counts) if pitch_counts else 0
        profile.avg_ip_l5 = sum(ips) / len(ips) if ips else 0
        profile.avg_batters_faced_l5 = sum(bfs) / len(bfs) if bfs else 0

        return profile

    def _tag_roles(self, env: GameEnvironment):
        """Apply role classifications to all entities in a game."""
        if env.home_sp:
            env.home_sp.role = self.roles.classify_sp(env.home_sp)
            env.home_sp.volatility_profile = self.roles.classify_volatility_sp(env.home_sp)

        if env.away_sp:
            env.away_sp.role = self.roles.classify_sp(env.away_sp)
            env.away_sp.volatility_profile = self.roles.classify_volatility_sp(env.away_sp)

        for h in env.home_lineup + env.away_lineup:
            h.role = self.roles.classify_hitter(h)
            h.volatility_profile = self.roles.classify_volatility_hitter(h)

        if env.home_bullpen:
            env.home_bullpen.role = self.roles.classify_bullpen(env.home_bullpen)
        if env.away_bullpen:
            env.away_bullpen.role = self.roles.classify_bullpen(env.away_bullpen)

    def _run_stability_gates(self, env: GameEnvironment):
        """Run LSG and PSG for all players."""
        # LSG for starting pitchers
        if env.home_sp:
            lss = self.roles.compute_lss(env.home_sp, env.away_offense)
            env.home_sp.lss = lss
            if lss >= 2:
                env.home_sp.lss_confidence = Confidence.GREEN
            elif lss >= 0:
                env.home_sp.lss_confidence = Confidence.YELLOW
            else:
                env.home_sp.lss_confidence = Confidence.RED

        if env.away_sp:
            lss = self.roles.compute_lss(env.away_sp, env.home_offense)
            env.away_sp.lss = lss
            if lss >= 2:
                env.away_sp.lss_confidence = Confidence.GREEN
            elif lss >= 0:
                env.away_sp.lss_confidence = Confidence.YELLOW
            else:
                env.away_sp.lss_confidence = Confidence.RED

        # PSG for hitters
        for h in env.home_lineup + env.away_lineup:
            pss = self.roles.compute_pss(h)
            h.pss = pss
            if pss >= 2:
                h.pss_confidence = Confidence.GREEN
            elif pss >= 0:
                h.pss_confidence = Confidence.YELLOW
            else:
                h.pss_confidence = Confidence.RED

    def _tag_script(self, env: GameEnvironment):
        """Assign game script tag based on environment analysis."""
        # Pitcher-dominant: both SPs are strong
        both_strong = (
            env.home_sp and env.away_sp
            and env.home_sp.role in (SPRole.WORKHORSE_ACE, SPRole.K_SPECIALIST)
            and env.away_sp.role in (SPRole.WORKHORSE_ACE, SPRole.K_SPECIALIST)
        )
        if both_strong:
            env.script_tag = ScriptTag.PITCHER_DOMINANT
            return

        # Bullpen-fragile: one or both pens are stressed
        home_bp_weak = env.home_bullpen and env.home_bullpen.role in (
            BullpenRole.OVERWORKED, BullpenRole.LOW_LEV_BLEED, BullpenRole.COMMITTEE_CHAOS
        )
        away_bp_weak = env.away_bullpen and env.away_bullpen.role in (
            BullpenRole.OVERWORKED, BullpenRole.LOW_LEV_BLEED, BullpenRole.COMMITTEE_CHAOS
        )
        if home_bp_weak or away_bp_weak:
            env.script_tag = ScriptTag.BULLPEN_FRAGILE
            return

        # HR-inflated: park + weather + fly-ball arms
        if env.park_factor_hr >= 1.10 and env.temperature >= 80 and env.wind_speed >= 10:
            env.script_tag = ScriptTag.HR_INFLATED
            return

        # Fragile: one or both SPs are volatile
        sp_fragile = (
            (env.home_sp and env.home_sp.role in (SPRole.FLYBALL_FRAGILE, SPRole.SHORT_LEASH_VOLATILE, SPRole.WALK_RISK_ARM))
            or (env.away_sp and env.away_sp.role in (SPRole.FLYBALL_FRAGILE, SPRole.SHORT_LEASH_VOLATILE, SPRole.WALK_RISK_ARM))
        )
        if sp_fragile:
            env.script_tag = ScriptTag.FRAGILE
            return

        # Run-suppressing: cold, wind blowing in, pitcher's park
        if env.temperature < 55 and env.park_factor_runs < 0.95:
            env.script_tag = ScriptTag.RUN_SUPPRESSING
            return

        env.script_tag = ScriptTag.STABLE

    # ── Utility: Export slate as JSON ─────────────────────────────────────

    def export_slate(self, environments: list, filepath: str = "rocket_slate.json"):
        """Export the full slate to JSON for downstream consumption."""
        output = []
        for env in environments:
            d = {
                "game_pk": env.game_pk,
                "date": env.date,
                "matchup": f"{env.away_team} @ {env.home_team}",
                "venue": env.venue_name,
                "script_tag": env.script_tag.value,
                "weather": {
                    "temp": env.temperature,
                    "wind": env.wind_speed,
                    "precip_pct": env.precipitation_pct,
                    "roof_closed": env.roof_closed,
                },
                "park_factor_runs": env.park_factor_runs,
                "home_sp": {
                    "name": env.home_sp.name if env.home_sp else None,
                    "role": env.home_sp.role.value if env.home_sp and env.home_sp.role else None,
                    "lss": env.home_sp.lss if env.home_sp else None,
                    "lss_conf": env.home_sp.lss_confidence.value if env.home_sp else None,
                    "k_pct": env.home_sp.k_pct if env.home_sp else None,
                    "bb_pct": env.home_sp.bb_pct if env.home_sp else None,
                    "avg_ip_l5": env.home_sp.avg_ip_l5 if env.home_sp else None,
                    "volatility": env.home_sp.volatility_profile.value if env.home_sp else None,
                },
                "away_sp": {
                    "name": env.away_sp.name if env.away_sp else None,
                    "role": env.away_sp.role.value if env.away_sp and env.away_sp.role else None,
                    "lss": env.away_sp.lss if env.away_sp else None,
                    "lss_conf": env.away_sp.lss_confidence.value if env.away_sp else None,
                    "k_pct": env.away_sp.k_pct if env.away_sp else None,
                    "bb_pct": env.away_sp.bb_pct if env.away_sp else None,
                    "avg_ip_l5": env.away_sp.avg_ip_l5 if env.away_sp else None,
                    "volatility": env.away_sp.volatility_profile.value if env.away_sp else None,
                },
                "weather_risk": env.weather_risk.value,
                "lineup_confirmed": env.lineup_confirmed,
            }
            output.append(d)

        with open(filepath, "w") as f:
            json.dump(output, f, indent=2)
        logger.info(f"Slate exported to {filepath}")
        return output


# ══════════════════════════════════════════════════════════════════════════════
# CLI ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    date = sys.argv[1] if len(sys.argv) > 1 else datetime.now().strftime("%Y-%m-%d")
    rdl = RocketDataLayer()
    slate = rdl.run_daily_pipeline(date)

    print(f"\n{'='*60}")
    print(f"  ROCKET MLB SLATE — {date}")
    print(f"  {len(slate)} games processed")
    print(f"{'='*60}\n")

    for env in slate:
        home_sp = env.home_sp.name if env.home_sp else "TBD"
        away_sp = env.away_sp.name if env.away_sp else "TBD"
        home_role = env.home_sp.role.value if env.home_sp and env.home_sp.role else "?"
        away_role = env.away_sp.role.value if env.away_sp and env.away_sp.role else "?"
        home_lss = env.home_sp.lss_confidence.value if env.home_sp else "?"
        away_lss = env.away_sp.lss_confidence.value if env.away_sp else "?"

        print(f"  {env.away_team} @ {env.home_team}  |  {env.venue_name}")
        print(f"    Script: {env.script_tag.value}  |  Temp: {env.temperature}°F  |  Wind: {env.wind_speed}mph")
        print(f"    Home SP: {home_sp} ({home_role}) LSS:{home_lss}")
        print(f"    Away SP: {away_sp} ({away_role}) LSS:{away_lss}")
        print(f"    Park HR Factor: {env.park_factor_hr}")
        print()

    # Export
    rdl.export_slate(slate)
