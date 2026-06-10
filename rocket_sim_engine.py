"""
ROCKET MLB ENGINE — SIMULATION ENGINE v1
==========================================
150,000-iteration Monte Carlo simulation framework for:
  - SP Outs / Innings Pitched
  - SP Strikeouts
  - Hitter Total Bases / Hits / R+RBI
  - Team Totals (F5 and Full Game)
  - Stolen Bases

Consumes GameEnvironment objects from rocket_data_layer.py.

Requirements:
  pip install numpy pandas

Usage:
  from rocket_data_layer import RocketDataLayer, GameEnvironment
  from rocket_sim_engine import RocketSimEngine

  rdl = RocketDataLayer()
  slate = rdl.run_daily_pipeline("2026-06-15")
  sim = RocketSimEngine()
  results = sim.run_full_slate(slate)
"""

import logging
import json
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum

import numpy as np
import pandas as pd

from rocket_data_layer import (
    GameEnvironment, PitcherProfile, HitterProfile, BullpenProfile,
    TeamOffenseProfile, SPRole, BullpenRole, HitterRole,
    TeamOffenseArchetype, Confidence, ScriptTag, VolatilityProfile,
    LineErrorDetector, RoleEngine,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("RocketSimEngine")

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

SIM_ITERATIONS = 150_000  # Hard gate — minimum per your doc
RNG = np.random.default_rng(seed=42)


# ══════════════════════════════════════════════════════════════════════════════
# SIMULATION OUTPUT STRUCTURES
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class SimDistribution:
    """Statistical summary of a simulation run."""
    median: float = 0.0
    p60: float = 0.0       # 60th percentile
    p75: float = 0.0       # 75th percentile
    p25: float = 0.0       # 25th percentile
    mean: float = 0.0
    std: float = 0.0
    min_val: float = 0.0
    max_val: float = 0.0
    # Tail hit rates (for ladder analysis)
    tail_rates: dict = field(default_factory=dict)  # {threshold: hit_rate}
    # Failure modes
    failure_notes: list = field(default_factory=list)
    raw_sims: Optional[np.ndarray] = None

    @classmethod
    def from_array(cls, arr: np.ndarray, thresholds: list = None):
        """Build distribution summary from raw simulation array."""
        d = cls(
            median=float(np.median(arr)),
            p60=float(np.percentile(arr, 60)),
            p75=float(np.percentile(arr, 75)),
            p25=float(np.percentile(arr, 25)),
            mean=float(np.mean(arr)),
            std=float(np.std(arr)),
            min_val=float(np.min(arr)),
            max_val=float(np.max(arr)),
        )
        if thresholds:
            for t in thresholds:
                d.tail_rates[t] = float(np.mean(arr >= t))
        return d


@dataclass
class PropProjection:
    """Full projection for a single prop market."""
    player_name: str
    player_id: int
    market: str            # "SP Outs", "SP Ks", "Hitter TB", etc.
    line: float            # The book's line (e.g., 16.5 outs)
    odds: int              # American odds (e.g., -115)
    projection: SimDistribution = field(default_factory=SimDistribution)
    # Edge analysis
    p_true_over: float = 0.0
    p_true_under: float = 0.0
    p_implied: float = 0.0
    edge_pct: float = 0.0
    led_flag: Confidence = Confidence.RED
    # Stability gates
    stability_confidence: Confidence = Confidence.YELLOW
    # Tier
    tier: str = "PASS"     # "S++", "S", "A", "PASS"
    volatility: VolatilityProfile = VolatilityProfile.HYBRID
    # Ladder eligibility
    ladder_score: int = 0  # 0-100
    ladder_eligible: bool = False
    # Reason
    reason: str = ""


@dataclass
class GameSimResult:
    """Full simulation output for a single game."""
    game_pk: int
    matchup: str
    script_tag: str
    # F5 projections
    f5_home_runs: SimDistribution = field(default_factory=SimDistribution)
    f5_away_runs: SimDistribution = field(default_factory=SimDistribution)
    f5_total: SimDistribution = field(default_factory=SimDistribution)
    # Full game projections
    fg_home_runs: SimDistribution = field(default_factory=SimDistribution)
    fg_away_runs: SimDistribution = field(default_factory=SimDistribution)
    fg_total: SimDistribution = field(default_factory=SimDistribution)
    # Player props
    prop_projections: list = field(default_factory=list)  # List[PropProjection]
    # Allowed bet types
    allowed_f5: bool = True
    allowed_full_game: bool = True
    allowed_sp_outs: bool = True
    allowed_sp_ks: bool = True
    allowed_team_total: bool = True
    allowed_hitter_props: bool = True
    # Confidence
    overall_confidence: Confidence = Confidence.YELLOW


# ══════════════════════════════════════════════════════════════════════════════
# SP OUTS / INNINGS PITCHED SIMULATOR
# ══════════════════════════════════════════════════════════════════════════════

class SPOutsSimulator:
    """
    Monte Carlo simulation for Starting Pitcher outs recorded.
    Injects variance for: pitch count, walks, TTO decay, hit clustering,
    manager pull probability.
    """

    def simulate(self, sp: PitcherProfile, opponent: TeamOffenseProfile = None,
                 env: GameEnvironment = None, n: int = SIM_ITERATIONS) -> np.ndarray:
        """
        Simulate SP outs recorded across n iterations.
        Returns array of shape (n,) with outs recorded per iteration.
        """
        # ── Base parameters ───────────────────────────────────────────────
        avg_ip = sp.avg_ip_l5 if sp.avg_ip_l5 > 0 else 5.5
        ip_std = max(0.5, np.std(sp.pitch_count_trend) / 15 if sp.pitch_count_trend else 0.8)

        avg_pc = sp.avg_pitch_count_l5 if sp.avg_pitch_count_l5 > 0 else 90
        pc_std = np.std(sp.pitch_count_trend) if sp.pitch_count_trend else 8

        bb_rate = sp.bb_pct if sp.bb_pct > 0 else 0.08

        # ── Opponent adjustments ──────────────────────────────────────────
        opp_patience_mod = 1.0
        if opponent:
            if opponent.archetype == TeamOffenseArchetype.PATIENT_WALK:
                opp_patience_mod = 1.12  # 12% more pitches per inning
            elif opponent.archetype == TeamOffenseArchetype.HIGH_K:
                opp_patience_mod = 0.92  # 8% fewer pitches per inning
            # Walk-heavy opponents inflate pitch count
            if opponent.team_bb_pct >= 0.10:
                bb_rate = min(bb_rate * 1.15, 0.18)

        # ── Umpire adjustment ─────────────────────────────────────────────
        ump_mod = 0.0
        if env:
            if env.umpire_zone_type == "wide":
                bb_rate *= 0.90
                ump_mod = 0.3  # slight outs boost
            elif env.umpire_zone_type == "tight":
                bb_rate *= 1.10
                ump_mod = -0.3

        # ── Simulate ──────────────────────────────────────────────────────
        outs = np.zeros(n)

        for i in range(n):
            # Pitch count ceiling for this sim
            pc_ceiling = max(50, RNG.normal(avg_pc, pc_std))

            # Pitches per inning (base ~16, adjusted by opponent)
            pitches_per_inning = max(12, RNG.normal(16, 2.5) * opp_patience_mod)

            # TTO decay — efficiency drops as pitcher goes deeper
            # 1st TTO (innings 1-3): normal
            # 2nd TTO (innings 4-6): slight decay
            # 3rd TTO (7+): significant decay
            tto_decay = [1.0, 1.0, 1.0, 1.05, 1.08, 1.12, 1.20, 1.25, 1.30]

            total_pitches = 0
            total_outs = 0
            inning = 0
            pulled = False

            while not pulled and inning < 9:
                inning += 1
                decay = tto_decay[min(inning - 1, len(tto_decay) - 1)]
                inn_pitches = pitches_per_inning * decay

                # Walk probability per batter this inning
                batters_this_inning = RNG.poisson(4.2)
                walks = RNG.binomial(batters_this_inning, bb_rate)

                # Each walk adds ~5 extra pitches
                inn_pitches += walks * 5

                # Hit clustering — BABIP variance can spike an inning
                if RNG.random() < 0.12:  # ~12% chance of a hit cluster
                    inn_pitches += RNG.integers(8, 20)

                total_pitches += inn_pitches

                # Manager pull check
                pull_prob = 0.0
                if total_pitches >= pc_ceiling * 0.85:
                    pull_prob = 0.15
                if total_pitches >= pc_ceiling * 0.95:
                    pull_prob = 0.50
                if total_pitches >= pc_ceiling:
                    pull_prob = 0.85
                if total_pitches >= pc_ceiling * 1.10:
                    pull_prob = 0.98

                # Short-leash managers pull earlier
                if sp.role == SPRole.SHORT_LEASH_VOLATILE:
                    pull_prob = min(1.0, pull_prob * 1.5)

                if RNG.random() < pull_prob:
                    # Partial inning — get 0-2 outs this inning
                    partial_outs = RNG.integers(0, 3)
                    total_outs += partial_outs
                    pulled = True
                else:
                    total_outs += 3

            outs[i] = total_outs + ump_mod

        return np.clip(outs, 0, 27)


# ══════════════════════════════════════════════════════════════════════════════
# SP STRIKEOUTS SIMULATOR
# ══════════════════════════════════════════════════════════════════════════════

class SPKsSimulator:
    """
    Monte Carlo simulation for Starting Pitcher strikeouts.
    Tied to the outs sim — can't get Ks if you get pulled early.
    """

    def simulate(self, sp: PitcherProfile, outs_sims: np.ndarray,
                 opponent: TeamOffenseProfile = None,
                 env: GameEnvironment = None, n: int = SIM_ITERATIONS) -> np.ndarray:
        """
        Simulate SP strikeouts across n iterations.
        outs_sims: array from SPOutsSimulator (ties K ceiling to leash).
        """
        k_rate = sp.k_pct if sp.k_pct > 0 else 0.22

        # ── Opponent K adjustment ─────────────────────────────────────────
        if opponent:
            if opponent.archetype == TeamOffenseArchetype.HIGH_K:
                k_rate = min(k_rate * 1.15, 0.40)
            elif opponent.archetype == TeamOffenseArchetype.CONTACT_CHAIN:
                k_rate *= 0.85
            elif opponent.archetype == TeamOffenseArchetype.PATIENT_WALK:
                k_rate *= 0.95  # Patience = fewer chases = fewer Ks

        # ── Umpire adjustment ─────────────────────────────────────────────
        if env:
            if env.umpire_zone_type == "wide":
                k_rate *= 1.05
            elif env.umpire_zone_type == "tight":
                k_rate *= 0.95

        # ── Simulate ──────────────────────────────────────────────────────
        ks = np.zeros(n)

        for i in range(n):
            # Batters faced is derived from outs recorded
            # ~1.35 batters per out on average (walks, hits, errors add batters)
            outs_this_game = outs_sims[i]
            batters_faced = int(outs_this_game * RNG.normal(1.35, 0.08))
            batters_faced = max(int(outs_this_game), batters_faced)

            # K per batter — binomial draw
            game_k_rate = max(0, RNG.normal(k_rate, 0.03))  # slight per-game variance
            strikeouts = RNG.binomial(batters_faced, min(game_k_rate, 0.45))
            ks[i] = strikeouts

        return ks


# ══════════════════════════════════════════════════════════════════════════════
# HITTER TOTAL BASES SIMULATOR
# ══════════════════════════════════════════════════════════════════════════════

class HitterTBSimulator:
    """
    Monte Carlo simulation for Hitter Total Bases.
    Accounts for: PA count, K rate, contact quality, park/weather, pitcher matchup.
    """

    # PA expectations by lineup slot
    PA_EXPECTATIONS = {
        1: 4.5, 2: 4.3, 3: 4.2, 4: 4.1, 5: 4.0,
        6: 3.8, 7: 3.6, 8: 3.4, 9: 3.2
    }

    def simulate(self, hitter: HitterProfile, opp_sp: PitcherProfile = None,
                 env: GameEnvironment = None, n: int = SIM_ITERATIONS) -> np.ndarray:
        """Simulate hitter total bases across n iterations."""

        # ── PA count ──────────────────────────────────────────────────────
        base_pa = self.PA_EXPECTATIONS.get(hitter.lineup_slot, 3.8)
        if hitter.expected_pa > 0:
            base_pa = hitter.expected_pa

        # ── K probability per PA ──────────────────────────────────────────
        k_rate = hitter.k_pct if hitter.k_pct > 0 else 0.22
        if opp_sp:
            # Pitcher K% modifies hitter K rate
            sp_k_mod = opp_sp.k_pct / 0.22  # normalize against league avg
            k_rate = k_rate * (0.5 + 0.5 * sp_k_mod)  # blend
            k_rate = min(k_rate, 0.45)

        # ── BB probability per PA ─────────────────────────────────────────
        bb_rate = hitter.bb_pct if hitter.bb_pct > 0 else 0.08

        # ── Contact outcome probabilities (conditional on contact) ────────
        # Base rates — tune these based on hitter profile
        iso = hitter.iso if hitter.iso > 0 else 0.150
        babip = hitter.babip if hitter.babip > 0 else 0.300

        # HR probability per AB (given contact)
        hr_rate = iso * 0.35  # rough proxy: ~35% of ISO comes from HR
        # Park/weather modifier
        if env:
            hr_rate *= env.park_factor_hr
            if env.temperature >= 80:
                hr_rate *= 1.05
            if env.wind_speed >= 10 and not env.roof_closed:
                hr_rate *= 1.08  # Wind out assumption — refine with direction

        # XBH rates
        double_rate = iso * 0.30
        triple_rate = 0.005 if hitter.sprint_speed >= 28 else 0.002
        single_rate = babip - hr_rate - double_rate - triple_rate
        single_rate = max(single_rate, 0.10)

        # Normalize
        contact_rate = 1.0 - k_rate - bb_rate
        contact_rate = max(contact_rate, 0.20)

        # ── Simulate ──────────────────────────────────────────────────────
        tb = np.zeros(n)

        for i in range(n):
            # PA count for this game
            pa = max(1, int(RNG.normal(base_pa, 0.5)))

            total_bases = 0
            for _ in range(pa):
                roll = RNG.random()

                if roll < k_rate:
                    # Strikeout — 0 TB
                    continue
                elif roll < k_rate + bb_rate:
                    # Walk — 0 TB (but on base)
                    continue
                else:
                    # Contact made — what type of hit?
                    hit_roll = RNG.random()
                    hit_total = single_rate + double_rate + triple_rate + hr_rate
                    # Normalize probabilities
                    p_single = single_rate / hit_total
                    p_double = double_rate / hit_total
                    p_triple = triple_rate / hit_total
                    # p_hr = remainder

                    if hit_roll < p_single:
                        total_bases += 1
                    elif hit_roll < p_single + p_double:
                        total_bases += 2
                    elif hit_roll < p_single + p_double + p_triple:
                        total_bases += 3
                    else:
                        total_bases += 4  # HR

            tb[i] = total_bases

        return tb


# ══════════════════════════════════════════════════════════════════════════════
# HITTER HITS / RUNS+RBI / SB SIMULATORS
# ══════════════════════════════════════════════════════════════════════════════

class HitterHitsSimulator:
    """Simulate total hits (any type) for a hitter."""

    PA_EXPECTATIONS = HitterTBSimulator.PA_EXPECTATIONS

    def simulate(self, hitter: HitterProfile, opp_sp: PitcherProfile = None,
                 env: GameEnvironment = None, n: int = SIM_ITERATIONS) -> np.ndarray:
        base_pa = self.PA_EXPECTATIONS.get(hitter.lineup_slot, 3.8)
        k_rate = hitter.k_pct if hitter.k_pct > 0 else 0.22
        bb_rate = hitter.bb_pct if hitter.bb_pct > 0 else 0.08
        babip = hitter.babip if hitter.babip > 0 else 0.300

        contact_rate = max(0.20, 1.0 - k_rate - bb_rate)
        hit_rate_per_pa = contact_rate * babip

        hits = np.zeros(n)
        for i in range(n):
            pa = max(1, int(RNG.normal(base_pa, 0.5)))
            hits[i] = RNG.binomial(pa, min(hit_rate_per_pa, 0.45))
        return hits


class HitterRunsRBISimulator:
    """
    Simulate Runs + RBIs for a hitter.
    This is heavily sequencing-dependent — we model it as a function of
    TB + lineup context + team run environment.
    """

    def simulate(self, hitter: HitterProfile, tb_sims: np.ndarray,
                 team_runs_sims: np.ndarray = None,
                 n: int = SIM_ITERATIONS) -> np.ndarray:
        """
        tb_sims: from HitterTBSimulator
        team_runs_sims: team total runs (for context on RBI opportunities)
        """
        r_rbi = np.zeros(n)

        # Base R+RBI correlates with TB but adds sequencing noise
        for i in range(n):
            tb = tb_sims[i]

            # Runs: ~0.3-0.5 runs per TB for middle-order hitters
            run_rate = 0.35
            if hitter.lineup_slot <= 2:
                run_rate = 0.45  # Top of order scores more
            elif hitter.lineup_slot >= 7:
                run_rate = 0.25  # Bottom scores less

            runs = RNG.poisson(max(0.1, tb * run_rate))

            # RBIs: correlated with TB + team context
            rbi_rate = 0.30
            if hitter.lineup_slot in [3, 4, 5]:
                rbi_rate = 0.50  # Middle-order drives in more
            elif hitter.lineup_slot <= 2:
                rbi_rate = 0.20

            # If team scores a lot, more RBI opportunities
            if team_runs_sims is not None:
                team_runs = team_runs_sims[i]
                if team_runs >= 6:
                    rbi_rate *= 1.20
                elif team_runs <= 2:
                    rbi_rate *= 0.70

            rbis = RNG.poisson(max(0.1, tb * rbi_rate))

            r_rbi[i] = runs + rbis

        return r_rbi


class SBSimulator:
    """Simulate stolen bases for a hitter."""

    def simulate(self, hitter: HitterProfile, opp_catcher_cs_pct: float = 0.27,
                 opp_sp_time_to_plate: float = 1.35,
                 n: int = SIM_ITERATIONS) -> np.ndarray:
        """
        opp_catcher_cs_pct: opponent catcher's caught stealing rate
        opp_sp_time_to_plate: pitcher's time to plate (sec)
        """
        sbs = np.zeros(n)

        # Base SB probability per game
        if hitter.role != HitterRole.SB_PRESSURE:
            # Non-SB specialists: very low SB chance
            base_sb_rate = 0.05
        else:
            base_sb_rate = 0.35  # Active stealers

        # Adjust for battery weakness
        if opp_catcher_cs_pct < 0.22:
            base_sb_rate *= 1.25  # Weak arm
        elif opp_catcher_cs_pct > 0.32:
            base_sb_rate *= 0.70  # Strong arm

        if opp_sp_time_to_plate > 1.40:
            base_sb_rate *= 1.15  # Slow to plate

        # Adjust for speed
        if hitter.sprint_speed >= 29.5:
            base_sb_rate *= 1.20
        elif hitter.sprint_speed < 27:
            base_sb_rate *= 0.50

        for i in range(n):
            # First: does hitter reach base? (simplified)
            on_base_prob = 0.33  # league avg OBP
            if hitter.bb_pct > 0:
                on_base_prob = min(0.45, hitter.bb_pct + 0.250)  # rough OBP proxy

            reached = RNG.binomial(4, on_base_prob)  # ~4 PA
            if reached > 0:
                # For each time on base, chance of SB attempt
                attempts = RNG.binomial(reached, base_sb_rate)
                success_rate = hitter.sb_success_pct if hitter.sb_success_pct > 0 else 0.75
                sbs[i] = RNG.binomial(max(0, attempts), success_rate)

        return sbs


# ══════════════════════════════════════════════════════════════════════════════
# TEAM RUNS SIMULATOR (F5 + Full Game)
# ══════════════════════════════════════════════════════════════════════════════

class TeamRunsSimulator:
    """
    Monte Carlo simulation for team run scoring.
    Models F5 (first 5 innings) and full game separately.
    Uses modified Poisson with matchup-adjusted lambda.
    """

    def simulate_f5(self, batting_team: TeamOffenseProfile,
                    opp_sp: PitcherProfile, env: GameEnvironment = None,
                    n: int = SIM_ITERATIONS) -> np.ndarray:
        """Simulate runs scored in first 5 innings."""

        # Base run rate per inning (league avg ~0.5 runs/inning)
        base_lambda = 0.50

        # Team offense quality
        if batting_team:
            base_lambda *= (batting_team.runs_per_game / 4.5) if batting_team.runs_per_game > 0 else 1.0

        # Opponent SP quality adjustment
        if opp_sp:
            if opp_sp.role == SPRole.WORKHORSE_ACE:
                base_lambda *= 0.70
            elif opp_sp.role == SPRole.K_SPECIALIST:
                base_lambda *= 0.75
            elif opp_sp.role == SPRole.GROUNDBALL_SUPPRESSOR:
                base_lambda *= 0.82
            elif opp_sp.role == SPRole.WALK_RISK_ARM:
                base_lambda *= 1.15
            elif opp_sp.role == SPRole.FLYBALL_FRAGILE:
                base_lambda *= 1.10
            elif opp_sp.role == SPRole.SHORT_LEASH_VOLATILE:
                base_lambda *= 1.08

            # Handedness matchup
            if batting_team:
                if opp_sp.throws == "L" and batting_team.vs_lhp_woba > 0:
                    matchup_mod = batting_team.vs_lhp_woba / 0.320  # vs league avg
                    base_lambda *= (0.5 + 0.5 * matchup_mod)
                elif opp_sp.throws == "R" and batting_team.vs_rhp_woba > 0:
                    matchup_mod = batting_team.vs_rhp_woba / 0.320
                    base_lambda *= (0.5 + 0.5 * matchup_mod)

        # Park factor
        if env:
            base_lambda *= env.park_factor_runs

        # F5 = 5 innings of SP
        f5_lambda = base_lambda * 5

        # Simulate
        runs = RNG.poisson(max(0.1, f5_lambda), size=n).astype(float)
        return runs

    def simulate_full_game(self, batting_team: TeamOffenseProfile,
                           opp_sp: PitcherProfile, opp_bullpen: BullpenProfile,
                           f5_sims: np.ndarray, env: GameEnvironment = None,
                           n: int = SIM_ITERATIONS) -> np.ndarray:
        """
        Simulate full game runs.
        = F5 runs + bullpen-phase runs (innings 6-9).
        """
        # Bullpen-phase lambda per inning
        bp_lambda = 0.50  # base

        if batting_team:
            bp_lambda *= (batting_team.runs_per_game / 4.5) if batting_team.runs_per_game > 0 else 1.0

        # Bullpen quality adjustment
        if opp_bullpen:
            if opp_bullpen.role == BullpenRole.LOCKDOWN:
                bp_lambda *= 0.65
            elif opp_bullpen.role == BullpenRole.HIGH_K_BRIDGE:
                bp_lambda *= 0.75
            elif opp_bullpen.role == BullpenRole.OVERWORKED:
                bp_lambda *= 1.25
            elif opp_bullpen.role == BullpenRole.LOW_LEV_BLEED:
                bp_lambda *= 1.30
            elif opp_bullpen.role == BullpenRole.COMMITTEE_CHAOS:
                bp_lambda *= 1.20

        if env:
            bp_lambda *= env.park_factor_runs

        # ~4 bullpen innings on average (depends on SP depth)
        bp_innings = np.clip(RNG.normal(4.0, 0.8, size=n), 2, 5)
        bp_runs = np.array([
            RNG.poisson(max(0.1, bp_lambda * innings))
            for innings in bp_innings
        ])

        full_game = f5_sims + bp_runs
        return full_game


# ══════════════════════════════════════════════════════════════════════════════
# PRE-LIST SCANNER (RPS) — Module 5
# ══════════════════════════════════════════════════════════════════════════════

class PreListScanner:
    """
    Reduce the slate to ~10-18 candidates worth full simulation.
    Does NOT generate picks — creates the Pre-List.
    """

    @dataclass
    class Candidate:
        player_name: str
        player_id: int
        market: str
        reason: str
        priority: int = 0  # higher = stronger candidate

    def scan(self, environments: list) -> list:
        """
        Scan all games and return Pre-List candidates.
        """
        candidates = []

        for env in environments:
            # ── SP Outs candidates ────────────────────────────────────────
            for sp, opp_team, label in [
                (env.home_sp, env.away_team, "home"),
                (env.away_sp, env.home_team, "away"),
            ]:
                if not sp:
                    continue

                # Must have stable leash
                if sp.lss_confidence == Confidence.RED:
                    continue

                if sp.role in (SPRole.WORKHORSE_ACE, SPRole.K_SPECIALIST,
                               SPRole.GROUNDBALL_SUPPRESSOR, SPRole.CONTACT_MANAGER):
                    priority = 3 if sp.role == SPRole.WORKHORSE_ACE else 2
                    if sp.lss_confidence == Confidence.GREEN:
                        priority += 1
                    candidates.append(self.Candidate(
                        player_name=sp.name,
                        player_id=sp.player_id,
                        market="SP Outs",
                        reason=f"{sp.role.value} | LSS:{sp.lss} | avgIP:{sp.avg_ip_l5:.1f} | vs {opp_team}",
                        priority=priority,
                    ))

            # ── SP Ks candidates ──────────────────────────────────────────
            for sp, opp_offense, opp_team in [
                (env.home_sp, env.away_offense, env.away_team),
                (env.away_sp, env.home_offense, env.home_team),
            ]:
                if not sp:
                    continue
                if sp.k_pct < 0.22:
                    continue
                if sp.lss_confidence == Confidence.RED:
                    continue

                priority = 2
                if sp.role == SPRole.K_SPECIALIST:
                    priority = 4
                if opp_offense and opp_offense.archetype == TeamOffenseArchetype.HIGH_K:
                    priority += 1

                candidates.append(self.Candidate(
                    player_name=sp.name,
                    player_id=sp.player_id,
                    market="SP Ks",
                    reason=f"K%:{sp.k_pct:.1%} | {sp.role.value} | vs {opp_team}",
                    priority=priority,
                ))

            # ── F5 candidates ─────────────────────────────────────────────
            if env.script_tag in (ScriptTag.PITCHER_DOMINANT, ScriptTag.STABLE):
                candidates.append(self.Candidate(
                    player_name=f"{env.away_team}@{env.home_team}",
                    player_id=env.game_pk,
                    market="F5",
                    reason=f"Script:{env.script_tag.value} | Both SPs tagged",
                    priority=2,
                ))

            # ── Team Total candidates ─────────────────────────────────────
            if env.script_tag in (ScriptTag.FRAGILE, ScriptTag.BULLPEN_FRAGILE, ScriptTag.HR_INFLATED):
                candidates.append(self.Candidate(
                    player_name=f"{env.away_team}@{env.home_team}",
                    player_id=env.game_pk,
                    market="Team Total",
                    reason=f"Script:{env.script_tag.value}",
                    priority=2,
                ))

            # ── Hitter props ──────────────────────────────────────────────
            for lineup, opp_sp_profile in [
                (env.home_lineup, env.away_sp),
                (env.away_lineup, env.home_sp),
            ]:
                for h in lineup:
                    if h.pss_confidence == Confidence.RED:
                        continue
                    if h.lineup_slot <= 5 and h.confirmed_starter:
                        candidates.append(self.Candidate(
                            player_name=h.name,
                            player_id=h.player_id,
                            market="Hitter TB",
                            reason=f"Slot:{h.lineup_slot} | {h.role.value if h.role else '?'} | PSS:{h.pss}",
                            priority=1 if h.pss_confidence == Confidence.GREEN else 0,
                        ))

        # Sort by priority descending, cap at ~18
        candidates.sort(key=lambda c: c.priority, reverse=True)
        return candidates[:18]


# ══════════════════════════════════════════════════════════════════════════════
# PROP QUALIFICATION GATE
# ══════════════════════════════════════════════════════════════════════════════

class PropQualificationGate:
    """
    No prop is official unless all three layers are satisfied:
    1. Role Edge — does the player's role create a real path?
    2. Matchup Edge — does the opposing role/environment support that path?
    3. Environment Edge — does the game state allow it in neutral scripts?
    Must pass at least 2 of 3.
    """

    @staticmethod
    def qualify(prop: PropProjection, env: GameEnvironment,
                sp: PitcherProfile = None, hitter: HitterProfile = None) -> bool:
        gates_passed = 0

        # 1. Role Edge
        if sp:
            if prop.market == "SP Outs" and sp.role in (
                SPRole.WORKHORSE_ACE, SPRole.K_SPECIALIST,
                SPRole.GROUNDBALL_SUPPRESSOR, SPRole.CONTACT_MANAGER
            ):
                gates_passed += 1
            elif prop.market == "SP Ks" and sp.k_pct >= 0.22:
                gates_passed += 1

        if hitter:
            if prop.market == "Hitter TB" and hitter.role in (
                HitterRole.LIFT_PULL_POWER, HitterRole.GAP_XBH,
                HitterRole.RUN_PRODUCER, HitterRole.BOOM_BUST_BARREL
            ):
                gates_passed += 1
            elif prop.market == "Hitter Hits" and hitter.role in (
                HitterRole.TABLE_SETTER, HitterRole.CONTACT_BAT
            ):
                gates_passed += 1

        # 2. Matchup Edge
        if prop.edge_pct >= 4.0:
            gates_passed += 1

        # 3. Environment Edge
        if env.script_tag in (ScriptTag.STABLE, ScriptTag.PITCHER_DOMINANT):
            gates_passed += 1
        elif env.script_tag == ScriptTag.FRAGILE and prop.market in ("SP Ks", "Team Total"):
            gates_passed += 1

        return gates_passed >= 2


# ══════════════════════════════════════════════════════════════════════════════
# CONFIDENCE & TIER ASSIGNMENT
# ══════════════════════════════════════════════════════════════════════════════

class TierAssigner:
    """
    Assign tier based on edge + stability.
    S++ = high edge + high stability + neutral script survivability
    S   = high edge + medium stability
    A   = playable but exposure-capped
    PASS = fragile or market-efficient
    """

    @staticmethod
    def assign(prop: PropProjection) -> str:
        edge = prop.edge_pct
        stability = prop.stability_confidence

        if edge >= 8.0 and stability == Confidence.GREEN:
            return "S++"
        elif edge >= 6.0 and stability in (Confidence.GREEN, Confidence.YELLOW):
            return "S"
        elif edge >= 4.0 and stability != Confidence.RED:
            return "A"
        else:
            return "PASS"

    @staticmethod
    def compute_ladder_score(prop: PropProjection, sp: PitcherProfile = None,
                             hitter: HitterProfile = None) -> int:
        """
        Ladder Readiness Score (0-100).
        75+ = Ladder Eligible, 85+ = Aggressive Ladder, <70 = Base only.
        """
        score = 50  # base

        # Edge contribution
        if prop.edge_pct >= 10:
            score += 20
        elif prop.edge_pct >= 7:
            score += 12
        elif prop.edge_pct >= 4:
            score += 5

        # Stability contribution
        if prop.stability_confidence == Confidence.GREEN:
            score += 15
        elif prop.stability_confidence == Confidence.YELLOW:
            score += 5
        else:
            score -= 15

        # Volatility penalty
        if prop.volatility == VolatilityProfile.CEILING:
            score -= 15
        elif prop.volatility == VolatilityProfile.FLOOR:
            score += 10

        # SP-specific
        if sp:
            if sp.role == SPRole.WORKHORSE_ACE:
                score += 10
            if sp.lss >= 3:
                score += 5

        # Hitter-specific
        if hitter:
            if hitter.lineup_slot <= 3:
                score += 5
            if hitter.pss >= 3:
                score += 5

        return max(0, min(100, score))


# ══════════════════════════════════════════════════════════════════════════════
# MAIN SIM ENGINE ORCHESTRATOR
# ══════════════════════════════════════════════════════════════════════════════

class RocketSimEngine:
    """
    Main simulation engine. Consumes GameEnvironment objects from the data layer
    and produces full simulation outputs with prop projections.

    Usage:
        sim = RocketSimEngine(iterations=150000)
        results = sim.run_full_slate(environments)
    """

    def __init__(self, iterations: int = SIM_ITERATIONS):
        self.n = iterations
        self.sp_outs_sim = SPOutsSimulator()
        self.sp_ks_sim = SPKsSimulator()
        self.hitter_tb_sim = HitterTBSimulator()
        self.hitter_hits_sim = HitterHitsSimulator()
        self.hitter_rrbi_sim = HitterRunsRBISimulator()
        self.sb_sim = SBSimulator()
        self.team_runs_sim = TeamRunsSimulator()
        self.scanner = PreListScanner()
        self.qual_gate = PropQualificationGate()
        self.tier_assigner = TierAssigner()
        self.led = LineErrorDetector()

    def run_full_slate(self, environments: list) -> list:
        """
        Run full simulation suite for all games in the slate.
        Returns list of GameSimResult objects.
        """
        logger.info(f"═══ ROCKET SIM ENGINE — {len(environments)} games — {self.n:,} iterations ═══")

        # ── Pre-List Scan ─────────────────────────────────────────────────
        logger.info("Running Pre-List Scanner...")
        pre_list = self.scanner.scan(environments)
        logger.info(f"  Pre-List: {len(pre_list)} candidates")

        # ── Run sims per game ─────────────────────────────────────────────
        results = []
        for env in environments:
            result = self._simulate_game(env, pre_list)
            results.append(result)

        logger.info(f"═══ SIM ENGINE COMPLETE — {len(results)} games simulated ═══")
        return results

    def _simulate_game(self, env: GameEnvironment, pre_list: list) -> GameSimResult:
        """Full simulation for a single game."""
        logger.info(f"  Simulating: {env.away_team} @ {env.home_team} (gamePk:{env.game_pk})")

        result = GameSimResult(
            game_pk=env.game_pk,
            matchup=f"{env.away_team} @ {env.home_team}",
            script_tag=env.script_tag.value,
        )

        # ── Team Runs (F5 + Full Game) ────────────────────────────────────
        # Home team batting (vs away SP + away bullpen)
        f5_home = self.team_runs_sim.simulate_f5(
            env.home_offense, env.away_sp, env, self.n
        )
        fg_home = self.team_runs_sim.simulate_full_game(
            env.home_offense, env.away_sp, env.away_bullpen, f5_home, env, self.n
        )

        # Away team batting (vs home SP + home bullpen)
        f5_away = self.team_runs_sim.simulate_f5(
            env.away_offense, env.home_sp, env, self.n
        )
        fg_away = self.team_runs_sim.simulate_full_game(
            env.away_offense, env.home_sp, env.home_bullpen, f5_away, env, self.n
        )

        result.f5_home_runs = SimDistribution.from_array(f5_home, thresholds=[1, 2, 3, 4, 5])
        result.f5_away_runs = SimDistribution.from_array(f5_away, thresholds=[1, 2, 3, 4, 5])
        result.f5_total = SimDistribution.from_array(f5_home + f5_away, thresholds=[3, 4, 5, 6, 7, 8])

        result.fg_home_runs = SimDistribution.from_array(fg_home, thresholds=[2, 3, 4, 5, 6, 7])
        result.fg_away_runs = SimDistribution.from_array(fg_away, thresholds=[2, 3, 4, 5, 6, 7])
        result.fg_total = SimDistribution.from_array(fg_home + fg_away, thresholds=[5, 6, 7, 8, 9, 10, 11])

        # ── SP Props ──────────────────────────────────────────────────────
        for sp, opp_offense, label in [
            (env.home_sp, env.away_offense, "home"),
            (env.away_sp, env.home_offense, "away"),
        ]:
            if not sp:
                continue

            # SP Outs
            outs_raw = self.sp_outs_sim.simulate(sp, opp_offense, env, self.n)
            outs_dist = SimDistribution.from_array(
                outs_raw, thresholds=[12, 15, 16, 17, 18, 19, 20, 21]
            )

            # SP Ks (tied to outs sim)
            ks_raw = self.sp_ks_sim.simulate(sp, outs_raw, opp_offense, env, self.n)
            ks_dist = SimDistribution.from_array(
                ks_raw, thresholds=[3, 4, 5, 6, 7, 8, 9, 10]
            )

            # Create prop projections (without market lines — those come from odds feed)
            outs_prop = PropProjection(
                player_name=sp.name,
                player_id=sp.player_id,
                market="SP Outs",
                line=0,  # Filled when odds are available
                odds=0,
                projection=outs_dist,
                stability_confidence=sp.lss_confidence,
                volatility=sp.volatility_profile,
                reason=f"{sp.role.value} | LSS:{sp.lss} | median:{outs_dist.median:.1f} outs",
            )

            ks_prop = PropProjection(
                player_name=sp.name,
                player_id=sp.player_id,
                market="SP Ks",
                line=0,
                odds=0,
                projection=ks_dist,
                stability_confidence=sp.lss_confidence,
                volatility=sp.volatility_profile,
                reason=f"K%:{sp.k_pct:.1%} | median:{ks_dist.median:.1f} Ks",
            )

            result.prop_projections.extend([outs_prop, ks_prop])

        # ── Hitter Props ──────────────────────────────────────────────────
        for lineup, opp_sp in [
            (env.home_lineup, env.away_sp),
            (env.away_lineup, env.home_sp),
        ]:
            team_runs = fg_home if lineup == env.home_lineup else fg_away

            for h in lineup:
                if h.pss_confidence == Confidence.RED:
                    continue
                if h.lineup_slot > 6:
                    continue  # Focus on core lineup

                # TB sim
                tb_raw = self.hitter_tb_sim.simulate(h, opp_sp, env, self.n)
                tb_dist = SimDistribution.from_array(
                    tb_raw, thresholds=[1, 2, 3, 4, 5]
                )

                tb_prop = PropProjection(
                    player_name=h.name,
                    player_id=h.player_id,
                    market="Hitter TB",
                    line=0,
                    odds=0,
                    projection=tb_dist,
                    stability_confidence=h.pss_confidence,
                    volatility=h.volatility_profile,
                    reason=f"Slot:{h.lineup_slot} | {h.role.value if h.role else '?'} | median:{tb_dist.median:.1f} TB",
                )
                result.prop_projections.append(tb_prop)

                # Hits sim
                hits_raw = self.hitter_hits_sim.simulate(h, opp_sp, env, self.n)
                hits_dist = SimDistribution.from_array(
                    hits_raw, thresholds=[1, 2, 3]
                )

                hits_prop = PropProjection(
                    player_name=h.name,
                    player_id=h.player_id,
                    market="Hitter Hits",
                    line=0,
                    odds=0,
                    projection=hits_dist,
                    stability_confidence=h.pss_confidence,
                    volatility=h.volatility_profile,
                    reason=f"Slot:{h.lineup_slot} | median:{hits_dist.median:.1f} hits",
                )
                result.prop_projections.append(hits_prop)

                # R+RBI sim
                rrbi_raw = self.hitter_rrbi_sim.simulate(h, tb_raw, team_runs, self.n)
                rrbi_dist = SimDistribution.from_array(
                    rrbi_raw, thresholds=[1, 2, 3, 4]
                )

                rrbi_prop = PropProjection(
                    player_name=h.name,
                    player_id=h.player_id,
                    market="Hitter R+RBI",
                    line=0,
                    odds=0,
                    projection=rrbi_dist,
                    stability_confidence=h.pss_confidence,
                    volatility=h.volatility_profile,
                    reason=f"Slot:{h.lineup_slot} | median:{rrbi_dist.median:.1f} R+RBI",
                )
                result.prop_projections.append(rrbi_prop)

                # SB sim (only for SB-eligible)
                if h.role == HitterRole.SB_PRESSURE or h.sprint_speed >= 28:
                    sb_raw = self.sb_sim.simulate(h, n=self.n)
                    sb_dist = SimDistribution.from_array(sb_raw, thresholds=[1, 2])

                    sb_prop = PropProjection(
                        player_name=h.name,
                        player_id=h.player_id,
                        market="SB",
                        line=0,
                        odds=0,
                        projection=sb_dist,
                        stability_confidence=h.pss_confidence,
                        volatility=VolatilityProfile.CEILING,
                        reason=f"Speed:{h.sprint_speed} | SB%:{h.sb_success_pct:.0%}",
                    )
                    result.prop_projections.append(sb_prop)

        # ── Determine allowed bet types ───────────────────────────────────
        result.allowed_f5 = env.script_tag != ScriptTag.CHAOS
        result.allowed_full_game = env.script_tag != ScriptTag.CHAOS
        if env.home_sp and env.home_sp.lss_confidence == Confidence.RED:
            result.allowed_sp_outs = False
        if env.away_sp and env.away_sp.lss_confidence == Confidence.RED:
            result.allowed_sp_outs = False
        if not env.lineup_confirmed:
            result.allowed_hitter_props = False

        return result

    def apply_odds(self, results: list, odds_data: dict):
        """
        Apply odds/lines to prop projections and compute edges.
        odds_data format: {
            "player_id|market": {"line": 5.5, "over_odds": -115, "under_odds": -105},
            ...
        }
        """
        for result in results:
            for prop in result.prop_projections:
                key = f"{prop.player_id}|{prop.market}"
                if key in odds_data:
                    odds_info = odds_data[key]
                    prop.line = odds_info["line"]
                    prop.odds = odds_info.get("over_odds", -110)

                    # Compute true over probability from sim
                    if prop.projection.raw_sims is not None:
                        prop.p_true_over = float(np.mean(prop.projection.raw_sims >= prop.line))
                    else:
                        # Use tail rates as proxy
                        prop.p_true_over = prop.projection.tail_rates.get(prop.line, 0.5)

                    prop.p_true_under = 1.0 - prop.p_true_over

                    # LED edge calc
                    edge_info = self.led.compute_edge(prop.p_true_over, prop.odds)
                    prop.p_implied = edge_info["p_implied"]
                    prop.edge_pct = edge_info["edge_pct"]
                    prop.led_flag = edge_info["led_flag"]

                    # Tier assignment
                    prop.tier = self.tier_assigner.assign(prop)

                    # Ladder score
                    prop.ladder_score = self.tier_assigner.compute_ladder_score(prop)
                    prop.ladder_eligible = prop.ladder_score >= 75

    def generate_output(self, results: list) -> list:
        """
        Generate the minimum viable daily output per Section XIX of your doc.
        Returns list of dicts ready for display or JSON export.
        """
        output = []
        for r in results:
            game_output = {
                "game_pk": r.game_pk,
                "matchup": r.matchup,
                "environment_tag": r.script_tag,
                "allowed_bet_types": {
                    "F5": r.allowed_f5,
                    "full_game": r.allowed_full_game,
                    "SP_outs": r.allowed_sp_outs,
                    "SP_Ks": r.allowed_sp_ks,
                    "team_total": r.allowed_team_total,
                    "hitter_props": r.allowed_hitter_props,
                },
                "projections": {
                    "f5_total": {
                        "median": r.f5_total.median,
                        "p60": r.f5_total.p60,
                        "p75": r.f5_total.p75,
                    },
                    "fg_total": {
                        "median": r.fg_total.median,
                        "p60": r.fg_total.p60,
                        "p75": r.fg_total.p75,
                    },
                },
                "top_props": [],
            }

            # Filter to qualified props with positive edge
            qualified = [
                p for p in r.prop_projections
                if p.tier in ("S++", "S", "A") and p.edge_pct >= 4.0
            ]
            qualified.sort(key=lambda p: p.edge_pct, reverse=True)

            for p in qualified[:8]:  # Top 8 per game
                game_output["top_props"].append({
                    "player": p.player_name,
                    "market": p.market,
                    "line": p.line,
                    "projection_median": p.projection.median,
                    "projection_p60": p.projection.p60,
                    "edge_pct": p.edge_pct,
                    "tier": p.tier,
                    "confidence": p.stability_confidence.value,
                    "volatility": p.volatility.value,
                    "ladder_eligible": p.ladder_eligible,
                    "ladder_score": p.ladder_score,
                    "reason": p.reason,
                })

            output.append(game_output)

        return output

    def export_output(self, output: list, filepath: str = "rocket_sim_output.json"):
        """Export simulation output to JSON."""
        with open(filepath, "w") as f:
            json.dump(output, f, indent=2, default=str)
        logger.info(f"Sim output exported to {filepath}")


# ══════════════════════════════════════════════════════════════════════════════
# CLI ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    from datetime import datetime

    # Example: python rocket_sim_engine.py 2026-06-15
    date = sys.argv[1] if len(sys.argv) > 1 else datetime.now().strftime("%Y-%m-%d")

    # Import data layer
    from rocket_data_layer import RocketDataLayer

    # Run data pipeline
    rdl = RocketDataLayer()
    slate = rdl.run_daily_pipeline(date)

    # Run sim engine
    sim = RocketSimEngine(iterations=150_000)
    results = sim.run_full_slate(slate)

    # Generate output
    output = sim.generate_output(results)

    # Print summary
    print(f"\n{'='*70}")
    print(f"  🚀 ROCKET MLB SIM OUTPUT — {date}")
    print(f"  {len(output)} games | {sim.n:,} iterations per market")
    print(f"{'='*70}\n")

    for game in output:
        print(f"  {game['matchup']}")
        print(f"    Script: {game['environment_tag']}")
        print(f"    F5 Total — Median: {game['projections']['f5_total']['median']:.1f}")
        print(f"    FG Total — Median: {game['projections']['fg_total']['median']:.1f}")

        allowed = game["allowed_bet_types"]
        allowed_str = " | ".join(k for k, v in allowed.items() if v)
        print(f"    Allowed: {allowed_str}")

        if game["top_props"]:
            print(f"    ── Top Props ──")
            for p in game["top_props"]:
                print(f"      [{p['tier']}] {p['player']} {p['market']} "
                      f"| Median:{p['projection_median']:.1f} "
                      f"| Edge:{p['edge_pct']:.1f}% "
                      f"| {p['confidence']} "
                      f"| {'🪜' if p['ladder_eligible'] else ''}")
                print(f"        → {p['reason']}")
        else:
            print(f"    No qualified props (apply odds first)")
        print()

    # Export
    sim.export_output(output)
    print(f"  Output saved to rocket_sim_output.json")
