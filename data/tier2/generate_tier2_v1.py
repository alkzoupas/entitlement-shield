#!/usr/bin/env python3
"""Tier-2 synthetic world generator.

Deterministic generator for the 16 accepted scenario families in
`families.md`. Reads nothing else in the repository. Writes only under
data/tier2/.

Run: .venv/bin/python data/tier2/generate_tier2.py
"""
from __future__ import annotations

import bisect
import hashlib
import json
import random
from datetime import datetime, timedelta, timezone
from pathlib import Path

import polars as pl

# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------

SEED = 20260902
WORLD_START = datetime(2024, 7, 1, tzinfo=timezone.utc)
WORLD_END = datetime(2026, 6, 30, tzinfo=timezone.utc)
OUT_DIR = Path(__file__).parent

rng = random.Random(SEED)


def h(*parts: str) -> str:
    """Stable content hash (never uses builtin hash())."""
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:10]


def clip(dt: datetime, lo: datetime, hi: datetime) -> datetime:
    if dt < lo:
        return lo
    if dt > hi:
        return hi
    return dt


# --------------------------------------------------------------------------
# Hour-of-day weight profiles (index 0..23). Non-uniform: sleep hours low.
# --------------------------------------------------------------------------

def hour_profile(peaks, base=0.05, peak_w=1.0):
    w = [base] * 24
    for a, b in peaks:
        for hr in range(a, b + 1):
            w[hr % 24] = peak_w
    return w


GENERAL_AWAKE = hour_profile([(7, 23)], base=0.03, peak_w=1.0)
HOUSEHOLD_EVENING = hour_profile([(17, 22)], base=0.08, peak_w=1.6)
MORNING_SHIFT = hour_profile([(6, 10)], base=0.03, peak_w=1.8)
AFTERNOON_SHIFT = hour_profile([(12, 16)], base=0.03, peak_w=1.8)
NIGHT_SHIFT = hour_profile([(21, 23), (0, 1)], base=0.03, peak_w=1.8)
KIOSK_HOURS = hour_profile([(9, 17)], base=0.02, peak_w=1.0)
BUSINESS_HOURS = hour_profile([(9, 17)], base=0.02, peak_w=1.0)
HOTDESK_HOURS = hour_profile([(8, 18)], base=0.05, peak_w=1.0)
BURST_TIGHT = hour_profile([(10, 22)], base=0.1, peak_w=1.0)

WEEKDAYS = [0, 1, 2, 3, 4]
WEEKEND = [5, 6]

# weekday_boost: probability of *accepting* a candidate day of that weekday
# (used for rejection sampling so e.g. weekends are relatively favoured).
WEEKEND_BOOST = [0.55, 0.55, 0.55, 0.55, 0.6, 1.0, 1.0]
WEEKDAY_ONLY_BOOST = [1.0, 1.0, 1.0, 1.0, 1.0, 0.02, 0.02]


def pick_datetime(r: random.Random, start: datetime, end: datetime, hour_weights,
                   weekdays=None, weekday_boost=None) -> datetime:
    if end < start:
        end = start
    span_days = (end.date() - start.date()).days
    day = start.date()
    for _ in range(60):
        offset = r.randint(0, span_days) if span_days > 0 else 0
        cand = start.date() + timedelta(days=offset)
        wd = cand.weekday()
        if weekdays is not None and wd not in weekdays:
            continue
        if weekday_boost is not None and r.random() > weekday_boost[wd]:
            continue
        day = cand
        break
    else:
        day = start.date()
    hour = r.choices(range(24), weights=hour_weights, k=1)[0]
    minute = r.randint(0, 59)
    second = r.randint(0, 59)
    dt = datetime(day.year, day.month, day.day, hour, minute, second, tzinfo=timezone.utc)
    return clip(dt, start, end)


# --------------------------------------------------------------------------
# World accumulators
# --------------------------------------------------------------------------

class World:
    def __init__(self):
        self.acct_n = 0
        self.evt_n = 0
        self.obs_n = 0
        self.cluster_n = 0
        self.cycle_count = 0

        self.accounts = []          # dicts
        self.events = []            # dicts
        self.linkage = []           # dicts
        self.ground_truth = []      # dicts
        self.membership = []        # dicts

    # ---- id factories ----
    def next_acct_id(self):
        self.acct_n += 1
        return f"acct-{self.acct_n:06d}"

    def next_evt_id(self):
        self.evt_n += 1
        return f"evt-{self.evt_n:08d}"

    def next_obs_id(self):
        self.obs_n += 1
        return f"obs-{self.obs_n:07d}"

    def next_cluster_id(self):
        self.cluster_n += 1
        return f"cluster-{self.cluster_n:04d}"

    # ---- row builders ----
    def new_cluster(self, family: str, label: str, notes: str) -> str:
        cid = self.next_cluster_id()
        self.ground_truth.append(dict(cluster_id=cid, scenario_family=family,
                                       label=label, notes=notes))
        return cid

    def add_membership(self, account_id: str, cluster_id: str):
        self.membership.append(dict(account_id=account_id, cluster_id=cluster_id))

    def new_account(self, created_at: datetime, domain: str, verification: str) -> str:
        aid = self.next_acct_id()
        self.accounts.append(dict(account_id=aid, created_at=created_at,
                                   email_domain_type=domain,
                                   verification_level=verification))
        self.add_event(aid, created_at, "signup", 0, "low")
        return aid

    def add_event(self, account_id: str, ts: datetime, event_type: str,
                  credits_amount: int = 0, tier: str = "low"):
        ts = clip(ts, WORLD_START, WORLD_END)
        eid = self.next_evt_id()
        self.events.append(dict(event_id=eid, account_id=account_id, timestamp=ts,
                                 event_type=event_type, credits_amount=credits_amount,
                                 endpoint_cost_tier=tier))
        return eid

    def add_linkage(self, account_id: str, ts: datetime, signal_type: str,
                     signal_value: str, reliability: str):
        ts = clip(ts, WORLD_START, WORLD_END)
        oid = self.next_obs_id()
        self.linkage.append(dict(observation_id=oid, account_id=account_id, timestamp=ts,
                                  signal_type=signal_type, signal_value=signal_value,
                                  reliability=reliability))
        return oid


W = World()


# --------------------------------------------------------------------------
# Generic behaviors
# --------------------------------------------------------------------------

def maybe_add_verification(r: random.Random, account_id: str, created_at: datetime,
                            verification_level: str, fail_prob: float = 0.05):
    if verification_level == "none":
        if r.random() < fail_prob:
            W.add_event(account_id, created_at + timedelta(minutes=r.randint(1, 120)),
                        "verification_failed", 0, "low")
        return
    if r.random() < fail_prob:
        W.add_event(account_id, created_at + timedelta(minutes=r.randint(1, 60)),
                    "verification_failed", 0, "low")
    W.add_event(account_id, created_at + timedelta(minutes=r.randint(2, 240)),
                "verification", 0, "low")


def add_logins(r: random.Random, account_id: str, start: datetime, end: datetime,
               hour_weights, weekdays=None, weekday_boost=None, n_range=(3, 12)):
    if end <= start:
        return
    n = r.randint(*n_range)
    for _ in range(n):
        t = pick_datetime(r, start, end, hour_weights, weekdays, weekday_boost)
        W.add_event(account_id, t, "login", 0, "low")


def maybe_add_appeal(r: random.Random, account_id: str, start: datetime, end: datetime,
                      hour_weights, prob=0.01):
    if end <= start or r.random() >= prob:
        return
    t = pick_datetime(r, start, end, hour_weights)
    W.add_event(account_id, t, "appeal", 0, "low")


def run_credit_cycles(r: random.Random, account_id: str, created_at: datetime,
                       activity_end: datetime, hour_weights, weekdays=None,
                       weekday_boost=None, consume_events_range=(2, 6),
                       consume_amt_range=(1, 5), max_fraction=1.0, tier="high",
                       claim_delay_seconds=(3600, 5 * 24 * 3600), claim_prob=1.0,
                       cycle_days=30):
    """Standard ongoing usage: at each 30-day cycle, maybe claim 25 credits
    and consume some fraction of them, never exceeding the granted amount."""
    activity_end = min(activity_end, WORLD_END)
    cycle_start = created_at
    while cycle_start <= activity_end:
        cycle_end = min(cycle_start + timedelta(days=cycle_days), activity_end)
        if r.random() < claim_prob:
            delay = timedelta(seconds=r.randint(*claim_delay_seconds))
            claim_time = clip(cycle_start + delay, cycle_start, cycle_end)
            W.add_event(account_id, claim_time, "claim_credits", 25, "low")
            W.cycle_count += 1
            budget = max(0, min(25, int(round(25 * max_fraction))))
            remaining = budget
            n_events = r.randint(*consume_events_range)
            for _ in range(n_events):
                if remaining <= 0:
                    break
                amt = min(r.randint(*consume_amt_range), remaining)
                if amt <= 0:
                    continue
                t = pick_datetime(r, claim_time, cycle_end, hour_weights, weekdays, weekday_boost)
                W.add_event(account_id, t, "consume_credits", amt, tier)
                remaining -= amt
        cycle_start = cycle_start + timedelta(days=cycle_days)


# --------------------------------------------------------------------------
# Family 1: Household Shared Device (legitimate)
# --------------------------------------------------------------------------

def fam_household_shared_device(n_clusters=40):
    fam = "Household Shared Device"
    for i in range(n_clusters):
        cid = W.new_cluster(fam, "legitimate",
                             "Household shares one tablet; each member's own cycle cadence.")
        device = f"device-{h(cid, 'dev')}"
        network = f"net-{h(cid, 'net')}"
        payment = f"pay-{h(cid, 'pay')}"
        size = rng.randint(3, 5)
        base = pick_datetime(rng, WORLD_START, WORLD_END - timedelta(days=400), GENERAL_AWAKE)
        members = []
        for m in range(size):
            created = base if m == 0 else clip(base + timedelta(days=rng.randint(5, 220)),
                                                 WORLD_START, WORLD_END)
            domain = rng.choices(["consumer", "disposable"], weights=[0.96, 0.04])[0]
            verification = rng.choices(["email", "phone", "none"], weights=[0.5, 0.3, 0.2])[0]
            acct = W.new_account(created, domain, verification)
            W.add_membership(acct, cid)
            W.add_linkage(acct, created, "stable_device", device, "high")
            W.add_linkage(acct, created, "network_group", network, "high")
            if m < 2 or rng.random() < 0.4:
                W.add_linkage(acct, created, "payment_token", payment, "medium")
            maybe_add_verification(rng, acct, created, verification)
            activity_end = clip(created + timedelta(days=rng.randint(300, 700)), created, WORLD_END)
            run_credit_cycles(rng, acct, created, activity_end, HOUSEHOLD_EVENING,
                               weekday_boost=WEEKEND_BOOST, consume_events_range=(2, 6),
                               consume_amt_range=(1, 4), max_fraction=rng.uniform(0.5, 1.0))
            add_logins(rng, acct, created, activity_end, HOUSEHOLD_EVENING,
                       weekday_boost=WEEKEND_BOOST, n_range=(5, 18))
            maybe_add_appeal(rng, acct, created, activity_end, HOUSEHOLD_EVENING)
            members.append((acct, created, activity_end))
        # occasional "parent hands tablet to kid" close-in-time sessions
        if len(members) >= 2:
            for _ in range(rng.randint(1, 3)):
                a, ca, ea = rng.choice(members)
                b, cb, eb = rng.choice(members)
                if a == b:
                    continue
                lo = max(ca, cb)
                hi = min(ea, eb)
                if hi <= lo:
                    continue
                t = pick_datetime(rng, lo, hi, HOUSEHOLD_EVENING, weekday_boost=WEEKEND_BOOST)
                W.add_event(a, t, "login", 0, "low")
                W.add_event(b, t + timedelta(minutes=rng.randint(1, 8)), "login", 0, "low")


# --------------------------------------------------------------------------
# Family 2: Sequential Household Users (legitimate)
# --------------------------------------------------------------------------

def fam_sequential_household(n_clusters=30):
    fam = "Sequential Household Users"
    profiles = [MORNING_SHIFT, AFTERNOON_SHIFT, NIGHT_SHIFT]
    for i in range(n_clusters):
        cid = W.new_cluster(fam, "legitimate",
                             "Household device tiled across incompatible shift schedules.")
        device = f"device-{h(cid, 'dev')}"
        network = f"net-{h(cid, 'net')}"
        size = rng.randint(3, 4)
        base = pick_datetime(rng, WORLD_START, WORLD_END - timedelta(days=400), GENERAL_AWAKE)
        for m in range(size):
            created = base if m == 0 else clip(base + timedelta(days=rng.randint(2, 60)),
                                                 WORLD_START, WORLD_END)
            domain = "consumer"
            verification = rng.choices(["email", "phone", "none"], weights=[0.5, 0.3, 0.2])[0]
            acct = W.new_account(created, domain, verification)
            W.add_membership(acct, cid)
            W.add_linkage(acct, created, "stable_device", device, "high")
            W.add_linkage(acct, created, "network_group", network, "high")
            W.add_linkage(acct, created, "behavioral_template",
                          f"behav-{h(cid, str(m))}", "medium")
            maybe_add_verification(rng, acct, created, verification)
            profile = profiles[m % 3]
            tier_bias = "high" if m % 2 == 0 else "low"
            activity_end = clip(created + timedelta(days=rng.randint(350, 700)), created, WORLD_END)
            run_credit_cycles(rng, acct, created, activity_end, profile,
                               consume_events_range=(2, 7), consume_amt_range=(1, 4),
                               max_fraction=rng.uniform(0.4, 0.9), tier=tier_bias)
            add_logins(rng, acct, created, activity_end, profile, n_range=(8, 25))


# --------------------------------------------------------------------------
# Family 3: Shared Public Kiosk / Library Terminal (legitimate)
# --------------------------------------------------------------------------

def fam_kiosk(n_clusters=15):
    fam = "Shared Public Kiosk / Library Terminal"
    for i in range(n_clusters):
        cid = W.new_cluster(fam, "legitimate",
                             "Public terminal; many unrelated thin, one-off users.")
        device = f"device-{h(cid, 'dev')}"
        network = f"net-{h(cid, 'net')}"
        size = rng.randint(8, 20)
        for m in range(size):
            created = pick_datetime(rng, WORLD_START, WORLD_END - timedelta(days=5), KIOSK_HOURS,
                                     weekdays=WEEKDAYS, weekday_boost=WEEKDAY_ONLY_BOOST)
            domain = rng.choices(["consumer", "disposable"], weights=[0.9, 0.1])[0]
            verification = rng.choices(["none", "email"], weights=[0.75, 0.25])[0]
            acct = W.new_account(created, domain, verification)
            W.add_membership(acct, cid)
            W.add_linkage(acct, created, "stable_device", device, "high")
            W.add_linkage(acct, created, "network_group", network, "high")
            if rng.random() < 0.03:
                W.add_event(acct, created + timedelta(minutes=rng.randint(1, 30)),
                            "verification_failed", 0, "low")
            activity_end = clip(created + timedelta(days=rng.randint(1, 20)), created, WORLD_END)
            run_credit_cycles(rng, acct, created, activity_end, KIOSK_HOURS,
                               weekdays=WEEKDAYS, weekday_boost=WEEKDAY_ONLY_BOOST,
                               consume_events_range=(1, 3), consume_amt_range=(1, 3),
                               max_fraction=rng.uniform(0.08, 0.3), claim_prob=0.8)
            add_logins(rng, acct, created, activity_end, KIOSK_HOURS, weekdays=WEEKDAYS,
                       weekday_boost=WEEKDAY_ONLY_BOOST, n_range=(1, 4))


# --------------------------------------------------------------------------
# Family 4: Small Business Turnover (legitimate)
# --------------------------------------------------------------------------

def fam_small_business(n_clusters=25):
    fam = "Small Business Turnover"
    for i in range(n_clusters):
        cid = W.new_cluster(fam, "legitimate",
                             "Staff hired over time on office network/card; departures go quiet.")
        network = f"net-{h(cid, 'net')}"
        payment = f"pay-{h(cid, 'pay')}"
        size = rng.randint(4, 8)
        year_start = pick_datetime(rng, WORLD_START, WORLD_END - timedelta(days=200), BUSINESS_HOURS,
                                    weekdays=WEEKDAYS, weekday_boost=WEEKDAY_ONLY_BOOST)
        hire_dates = sorted(
            clip(year_start + timedelta(days=rng.randint(0, 500)), WORLD_START, WORLD_END)
            for _ in range(size)
        )
        for m, created in enumerate(hire_dates):
            domain = rng.choices(["enterprise", "consumer"], weights=[0.6, 0.4])[0]
            verification = rng.choices(["email", "phone", "enhanced"], weights=[0.4, 0.4, 0.2])[0]
            acct = W.new_account(created, domain, verification)
            W.add_membership(acct, cid)
            W.add_linkage(acct, created, "network_group", network, "high")
            if rng.random() < 0.75:
                W.add_linkage(acct, created, "payment_token", payment, "high")
            maybe_add_verification(rng, acct, created, verification)
            departs = rng.random() < 0.4
            if departs:
                activity_end = clip(created + timedelta(days=rng.randint(60, 500)), created, WORLD_END)
            else:
                activity_end = WORLD_END
            run_credit_cycles(rng, acct, created, activity_end, BUSINESS_HOURS,
                               weekdays=WEEKDAYS, weekday_boost=WEEKDAY_ONLY_BOOST,
                               consume_events_range=(2, 6), consume_amt_range=(1, 4),
                               max_fraction=rng.uniform(0.3, 0.85))
            add_logins(rng, acct, created, activity_end, BUSINESS_HOURS, weekdays=WEEKDAYS,
                       weekday_boost=WEEKDAY_ONLY_BOOST, n_range=(10, 40))


# --------------------------------------------------------------------------
# Family 5: Account Recovery Migration (legitimate)
# --------------------------------------------------------------------------

def fam_recovery_migration(n_clusters=35):
    fam = "Account Recovery Migration"
    for i in range(n_clusters):
        cid = W.new_cluster(fam, "legitimate",
                             "Lost-access replacement; single non-recurring handoff.")
        share_device = rng.random() < 0.4
        device = f"device-{h(cid, 'dev')}" if share_device else None
        recovery = f"rec-{h(cid, 'rec')}"
        payment = f"pay-{h(cid, 'pay')}"
        link_kind = rng.choice(["recovery", "payment", "both"])

        old_created = pick_datetime(rng, WORLD_START, WORLD_END - timedelta(days=120), GENERAL_AWAKE)
        old_domain = rng.choices(["consumer", "enterprise"], weights=[0.85, 0.15])[0]
        old_ver = rng.choices(["email", "phone", "none"], weights=[0.5, 0.3, 0.2])[0]
        old_acct = W.new_account(old_created, old_domain, old_ver)
        W.add_membership(old_acct, cid)
        if share_device:
            W.add_linkage(old_acct, old_created, "stable_device", device, "high")
        if link_kind in ("recovery", "both"):
            W.add_linkage(old_acct, old_created, "recovery_contact", recovery, "high")
        if link_kind in ("payment", "both"):
            W.add_linkage(old_acct, old_created, "payment_token", payment, "medium")
        maybe_add_verification(rng, old_acct, old_created, old_ver)
        # old account: normal use, then exhausts credits and goes abruptly quiet
        pre_abandon = clip(old_created + timedelta(days=rng.randint(20, 200)), old_created, WORLD_END)
        run_credit_cycles(rng, old_acct, old_created, pre_abandon, GENERAL_AWAKE,
                           consume_events_range=(2, 6), consume_amt_range=(2, 6),
                           max_fraction=rng.uniform(0.85, 1.0), claim_prob=1.0)
        add_logins(rng, old_acct, old_created, pre_abandon, GENERAL_AWAKE, n_range=(4, 15))

        gap = timedelta(days=rng.randint(0, 21), hours=rng.randint(0, 23))
        new_created = clip(pre_abandon + gap, WORLD_START, WORLD_END)
        if new_created <= pre_abandon:
            new_created = clip(pre_abandon + timedelta(hours=1), WORLD_START, WORLD_END)
        new_domain = rng.choices(["consumer", "enterprise"], weights=[0.85, 0.15])[0]
        new_ver = rng.choices(["email", "phone"], weights=[0.6, 0.4])[0]
        new_acct = W.new_account(new_created, new_domain, new_ver)
        W.add_membership(new_acct, cid)
        if share_device:
            W.add_linkage(new_acct, new_created, "stable_device", device, "high")
        if link_kind in ("recovery", "both"):
            W.add_linkage(new_acct, new_created, "recovery_contact", recovery, "high")
        if link_kind in ("payment", "both"):
            W.add_linkage(new_acct, new_created, "payment_token", payment, "medium")
        maybe_add_verification(rng, new_acct, new_created, new_ver)
        new_activity_end = WORLD_END if rng.random() < 0.6 else clip(
            new_created + timedelta(days=rng.randint(100, 500)), new_created, WORLD_END)
        run_credit_cycles(rng, new_acct, new_created, new_activity_end, GENERAL_AWAKE,
                           consume_events_range=(2, 6), consume_amt_range=(1, 5),
                           max_fraction=rng.uniform(0.4, 0.9))
        add_logins(rng, new_acct, new_created, new_activity_end, GENERAL_AWAKE, n_range=(6, 20))


# --------------------------------------------------------------------------
# Family 6: Roommate Turnover (ambiguous)
# --------------------------------------------------------------------------

def fam_roommate_turnover(n_clusters=25):
    fam = "Roommate Turnover"
    for i in range(n_clusters):
        cid = W.new_cluster(fam, "ambiguous",
                             "Shared apartment; roommate roster recurs over years.")
        device = f"device-{h(cid, 'dev')}"
        network = f"net-{h(cid, 'net')}"
        n_roommates = rng.randint(3, 6)
        cluster_start = pick_datetime(rng, WORLD_START, WORLD_END - timedelta(days=500), GENERAL_AWAKE)
        cursor = cluster_start
        for m in range(n_roommates):
            created = clip(cursor, WORLD_START, WORLD_END)
            domain = rng.choices(["consumer", "education"], weights=[0.8, 0.2])[0]
            verification = rng.choices(["email", "phone", "none"], weights=[0.5, 0.3, 0.2])[0]
            acct = W.new_account(created, domain, verification)
            W.add_membership(acct, cid)
            W.add_linkage(acct, created, "stable_device", device, "high")
            W.add_linkage(acct, created, "network_group", network, "high")
            W.add_linkage(acct, created, "payment_token", f"pay-{h(cid, str(m))}", "medium")
            maybe_add_verification(rng, acct, created, verification)
            tenure = timedelta(days=rng.randint(90, 260))
            activity_end = clip(created + tenure, created, WORLD_END)
            run_credit_cycles(rng, acct, created, activity_end, HOUSEHOLD_EVENING,
                               weekday_boost=WEEKEND_BOOST, consume_events_range=(2, 6),
                               consume_amt_range=(1, 4), max_fraction=rng.uniform(0.4, 0.9))
            add_logins(rng, acct, created, activity_end, HOUSEHOLD_EVENING,
                       weekday_boost=WEEKEND_BOOST, n_range=(5, 18))
            # next roommate overlaps a bit before this one fully leaves
            overlap = timedelta(days=rng.randint(-40, 20))
            cursor = created + tenure - overlap
            if cursor <= created:
                cursor = created + timedelta(days=30)


# --------------------------------------------------------------------------
# Family 7: Classroom / Course Cohort (legitimate)
# --------------------------------------------------------------------------

def fam_classroom(n_clusters=10):
    fam = "Classroom / Course Cohort"
    for i in range(n_clusters):
        cid = W.new_cluster(fam, "legitimate",
                             "Institutional cohort; synchronized class windows, term boundary drop-off.")
        network = f"net-{h(cid, 'net')}"
        size = rng.randint(15, 30)
        term_start = pick_datetime(rng, WORLD_START, WORLD_END - timedelta(days=110), BUSINESS_HOURS,
                                    weekdays=WEEKDAYS, weekday_boost=WEEKDAY_ONLY_BOOST)
        term_len = timedelta(days=rng.randint(70, 100))
        term_end = clip(term_start + term_len, term_start, WORLD_END)
        class_weekdays = sorted(rng.sample(WEEKDAYS, k=2))
        window_start_hour = rng.choice([9, 10, 13, 14, 16])
        class_hours = hour_profile([(window_start_hour, window_start_hour + 1)],
                                    base=0.01, peak_w=1.0)
        signup_window_end = clip(term_start + timedelta(days=7), term_start, WORLD_END)
        for m in range(size):
            created = pick_datetime(rng, term_start - timedelta(days=3), signup_window_end,
                                     BUSINESS_HOURS, weekdays=WEEKDAYS,
                                     weekday_boost=WEEKDAY_ONLY_BOOST)
            domain = "education"
            verification = rng.choices(["email", "phone"], weights=[0.7, 0.3])[0]
            acct = W.new_account(created, domain, verification)
            W.add_membership(acct, cid)
            W.add_linkage(acct, created, "network_group", network, "high")
            maybe_add_verification(rng, acct, created, verification)
            active_start = max(created, term_start)
            run_credit_cycles(rng, acct, active_start, term_end, class_hours,
                               weekdays=class_weekdays, consume_events_range=(2, 5),
                               consume_amt_range=(1, 3), max_fraction=rng.uniform(0.3, 0.75))
            add_logins(rng, acct, active_start, term_end, class_hours,
                       weekdays=class_weekdays, n_range=(8, 20))
            # a trickle of post-term stragglers, then nothing (cohort-wide drop-off)


# --------------------------------------------------------------------------
# Family 8: Trial Credit Farm (abusive)
# --------------------------------------------------------------------------

def fam_trial_farm(n_clusters=10):
    fam = "Trial Credit Farm"
    for i in range(n_clusters):
        cid = W.new_cluster(fam, "abusive",
                             "Scripted burst signups, immediate full claim, burn-and-abandon.")
        device = f"device-{h(cid, 'dev')}"
        network = f"net-{h(cid, 'net')}"
        behav = f"behav-{h(cid, 'behav')}"
        total = rng.randint(5, 15)
        n_bursts = rng.randint(2, 4)
        remaining = total
        for b in range(n_bursts):
            burst_size = max(1, remaining // (n_bursts - b))
            remaining -= burst_size
            burst_start = pick_datetime(rng, WORLD_START, WORLD_END - timedelta(days=3), BURST_TIGHT)
            for k in range(burst_size):
                created = burst_start + timedelta(seconds=rng.randint(0, 600))
                created = clip(created, WORLD_START, WORLD_END)
                domain = "disposable"
                verification = "none"
                acct = W.new_account(created, domain, verification)
                W.add_membership(acct, cid)
                if rng.random() < 0.8:
                    W.add_linkage(acct, created, "stable_device", device, "high")
                if rng.random() < 0.8:
                    W.add_linkage(acct, created, "network_group", network, "high")
                if rng.random() < 0.7:
                    W.add_linkage(acct, created, "behavioral_template", behav, "high")
                W.add_event(acct, created + timedelta(seconds=rng.randint(1, 30)),
                            "verification_failed", 0, "low")
                claim_t = clip(created + timedelta(seconds=rng.randint(2, 60)), created, WORLD_END)
                W.add_event(acct, claim_t, "claim_credits", 25, "low")
                W.cycle_count += 1
                remaining_credits = 25
                burn_end = clip(claim_t + timedelta(hours=rng.randint(1, 36)), claim_t, WORLD_END)
                n_events = rng.randint(4, 9)
                for _ in range(n_events):
                    if remaining_credits <= 0:
                        break
                    amt = min(rng.randint(3, 8), remaining_credits)
                    t = pick_datetime(rng, claim_t, burn_end, BURST_TIGHT)
                    W.add_event(acct, t, "consume_credits", amt, "high")
                    remaining_credits -= amt
                # abandoned: no further activity


# --------------------------------------------------------------------------
# Family 9: Serial Reroll on Expiry (abusive)
# --------------------------------------------------------------------------

def fam_serial_reroll(n_clusters=12):
    fam = "Serial Reroll on Expiry"
    for i in range(n_clusters):
        cid = W.new_cluster(fam, "abusive",
                             "Cycle-aligned exhaustion followed by fresh-account reroll, recurring.")
        device = f"device-{h(cid, 'dev')}"
        link_kind = rng.choice(["payment", "recovery", "both"])
        payment = f"pay-{h(cid, 'pay')}"
        recovery = f"rec-{h(cid, 'rec')}"
        chain_len = rng.randint(3, 6)
        created = pick_datetime(rng, WORLD_START, WORLD_END - timedelta(days=200), GENERAL_AWAKE)
        for m in range(chain_len):
            if created > WORLD_END:
                break
            domain = rng.choices(["consumer", "disposable"], weights=[0.6, 0.4])[0]
            verification = rng.choices(["none", "email"], weights=[0.7, 0.3])[0]
            acct = W.new_account(created, domain, verification)
            W.add_membership(acct, cid)
            W.add_linkage(acct, created, "stable_device", device, "high")
            if link_kind in ("payment", "both"):
                W.add_linkage(acct, created, "payment_token", payment, "high")
            if link_kind in ("recovery", "both"):
                W.add_linkage(acct, created, "recovery_contact", recovery, "high")
            claim_t = clip(created + timedelta(minutes=rng.randint(1, 90)), created, WORLD_END)
            W.add_event(acct, claim_t, "claim_credits", 25, "low")
            W.cycle_count += 1
            exhaust_end = clip(claim_t + timedelta(days=rng.randint(3, 9)), claim_t, WORLD_END)
            remaining_credits = 25
            n_events = rng.randint(4, 8)
            for _ in range(n_events):
                if remaining_credits <= 0:
                    break
                amt = min(rng.randint(3, 8), remaining_credits)
                t = pick_datetime(rng, claim_t, exhaust_end, GENERAL_AWAKE)
                W.add_event(acct, t, "consume_credits", amt, "high")
                remaining_credits -= amt
            add_logins(rng, acct, claim_t, exhaust_end, GENERAL_AWAKE, n_range=(1, 4))
            reroll_gap = timedelta(days=rng.randint(25, 29), hours=rng.randint(0, 23))
            created = created + reroll_gap


# --------------------------------------------------------------------------
# Family 10: Credit Mule Ring (abusive)
# --------------------------------------------------------------------------

def fam_mule_ring(n_clusters=8):
    fam = "Credit Mule Ring"
    for i in range(n_clusters):
        cid = W.new_cluster(fam, "abusive",
                             "Distinct devices/networks per recruit; single shared payment/recovery link.")
        link_kind = rng.choice(["payment", "recovery"])
        shared_val = f"{'pay' if link_kind == 'payment' else 'rec'}-{h(cid, 'shared')}"
        size = rng.randint(4, 8)
        n_batches = rng.randint(3, 6)
        batch_windows = []
        for _ in range(n_batches):
            bstart = pick_datetime(rng, WORLD_START, WORLD_END - timedelta(days=2), GENERAL_AWAKE)
            batch_windows.append((bstart, bstart + timedelta(hours=rng.randint(1, 3))))
        accounts = []
        for m in range(size):
            created = pick_datetime(rng, WORLD_START, WORLD_END - timedelta(days=30), GENERAL_AWAKE)
            domain = rng.choices(["consumer", "disposable"], weights=[0.7, 0.3])[0]
            verification = rng.choices(["email", "none", "phone"], weights=[0.5, 0.3, 0.2])[0]
            acct = W.new_account(created, domain, verification)
            W.add_membership(acct, cid)
            W.add_linkage(acct, created, "stable_device", f"device-{h(cid, str(m))}", "high")
            W.add_linkage(acct, created, "network_group", f"net-{h(cid, str(m))}", "high")
            W.add_linkage(acct, created, "behavioral_template", f"behav-{h(cid, str(m))}", "medium")
            if link_kind == "payment":
                W.add_linkage(acct, created, "payment_token", shared_val, "high")
            else:
                W.add_linkage(acct, created, "recovery_contact", shared_val, "high")
            maybe_add_verification(rng, acct, created, verification)
            activity_end = clip(created + timedelta(days=rng.randint(200, 600)), created, WORLD_END)
            accounts.append((acct, created, activity_end))
        for acct, created, activity_end in accounts:
            # baseline light claims to keep credits internally consistent
            run_credit_cycles(rng, acct, created, activity_end, GENERAL_AWAKE,
                               consume_events_range=(0, 1), consume_amt_range=(1, 2),
                               max_fraction=0.15, claim_prob=1.0)
            # operator-directed spikes concentrated in shared batch windows
            for bstart, bend in batch_windows:
                if bstart < created or bstart > activity_end:
                    continue
                if rng.random() < 0.7:
                    n = rng.randint(2, 4)
                    for _ in range(n):
                        t = pick_datetime(rng, bstart, bend, BURST_TIGHT)
                        W.add_event(acct, t, "consume_credits", rng.randint(1, 3), "high")
                        W.add_event(acct, t, "login", 0, "low")


# --------------------------------------------------------------------------
# Family 11: Device Reseller Fleet (ambiguous)
# --------------------------------------------------------------------------

def fam_device_reseller(n_clusters=20):
    fam = "Device Reseller Fleet"
    for i in range(n_clusters):
        cid = W.new_cluster(fam, "ambiguous",
                             "Refurbished device resold to a sequence of unrelated owners, no overlap.")
        device = f"device-{h(cid, 'dev')}"
        size = rng.randint(2, 4)
        cursor = pick_datetime(rng, WORLD_START, WORLD_END - timedelta(days=500), GENERAL_AWAKE)
        for m in range(size):
            created = clip(cursor, WORLD_START, WORLD_END)
            domain = rng.choices(["consumer", "enterprise"], weights=[0.85, 0.15])[0]
            verification = rng.choices(["email", "phone", "none"], weights=[0.4, 0.3, 0.3])[0]
            acct = W.new_account(created, domain, verification)
            W.add_membership(acct, cid)
            W.add_linkage(acct, created, "stable_device", device, "high")
            W.add_linkage(acct, created, "network_group", f"net-{h(cid, str(m))}", "high")
            maybe_add_verification(rng, acct, created, verification)
            ownership = timedelta(days=rng.randint(60, 220))
            activity_end = clip(created + ownership, created, WORLD_END)
            run_credit_cycles(rng, acct, created, activity_end, GENERAL_AWAKE,
                               consume_events_range=(1, 5), consume_amt_range=(1, 4),
                               max_fraction=rng.uniform(0.3, 0.8))
            add_logins(rng, acct, created, activity_end, GENERAL_AWAKE, n_range=(4, 16))
            gap = timedelta(days=rng.randint(21, 90))
            cursor = created + ownership + gap


# --------------------------------------------------------------------------
# Family 12: Co-working Hot Desk (legitimate)
# --------------------------------------------------------------------------

def fam_hot_desk(n_clusters=20):
    fam = "Co-working Hot Desk"
    for i in range(n_clusters):
        cid = W.new_cluster(fam, "legitimate",
                             "Rotating freelancers on shared office wifi/loaner laptops; sustained heavy per-account use.")
        network = f"net-{h(cid, 'net')}"
        share_device = rng.random() < 0.4
        device = f"device-{h(cid, 'dev')}" if share_device else None
        size = rng.randint(5, 10)
        members = []
        for m in range(size):
            created = pick_datetime(rng, WORLD_START, WORLD_END - timedelta(days=200), BUSINESS_HOURS,
                                     weekdays=WEEKDAYS, weekday_boost=WEEKDAY_ONLY_BOOST)
            domain = rng.choices(["consumer", "enterprise"], weights=[0.7, 0.3])[0]
            verification = rng.choices(["email", "phone", "enhanced"], weights=[0.4, 0.4, 0.2])[0]
            acct = W.new_account(created, domain, verification)
            W.add_membership(acct, cid)
            W.add_linkage(acct, created, "network_group", network, "high")
            if share_device:
                W.add_linkage(acct, created, "stable_device", device, "medium")
            maybe_add_verification(rng, acct, created, verification)
            heaviness = rng.choice(["heavy", "occasional"])
            frac = rng.uniform(0.6, 1.0) if heaviness == "heavy" else rng.uniform(0.15, 0.4)
            activity_end = clip(created + timedelta(days=rng.randint(200, 650)), created, WORLD_END)
            run_credit_cycles(rng, acct, created, activity_end, BUSINESS_HOURS,
                               weekdays=WEEKDAYS, weekday_boost=WEEKDAY_ONLY_BOOST,
                               consume_events_range=(3, 8) if heaviness == "heavy" else (1, 3),
                               consume_amt_range=(1, 5), max_fraction=frac)
            add_logins(rng, acct, created, activity_end, BUSINESS_HOURS, weekdays=WEEKDAYS,
                       weekday_boost=WEEKDAY_ONLY_BOOST, n_range=(10, 35))
            members.append((acct, created, activity_end))
        for _ in range(rng.randint(1, 3)):
            a, ca, ea = rng.choice(members)
            b, cb, eb = rng.choice(members)
            if a == b:
                continue
            lo, hi = max(ca, cb), min(ea, eb)
            if hi <= lo:
                continue
            t = pick_datetime(rng, lo, hi, BUSINESS_HOURS, weekdays=WEEKDAYS,
                               weekday_boost=WEEKDAY_ONLY_BOOST)
            W.add_event(a, t, "login", 0, "low")
            W.add_event(b, t + timedelta(minutes=rng.randint(1, 15)), "login", 0, "low")


# --------------------------------------------------------------------------
# Family 13: Boarding-School / Long-Distance Family (ambiguous)
# --------------------------------------------------------------------------

def fam_boarding_school(n_clusters=20):
    fam = "Boarding-School / Long-Distance Family"
    profiles = [GENERAL_AWAKE, MORNING_SHIFT, NIGHT_SHIFT]
    for i in range(n_clusters):
        cid = W.new_cluster(fam, "ambiguous",
                             "Single shared payment token across geographically separate family members.")
        payment = f"pay-{h(cid, 'pay')}"
        size = rng.randint(2, 3)
        base = pick_datetime(rng, WORLD_START, WORLD_END - timedelta(days=400), GENERAL_AWAKE)
        for m in range(size):
            created = base if m == 0 else clip(base + timedelta(days=rng.randint(0, 300)),
                                                 WORLD_START, WORLD_END)
            domain = rng.choices(["consumer", "education"], weights=[0.6, 0.4])[0]
            verification = rng.choices(["email", "phone", "enhanced"], weights=[0.4, 0.4, 0.2])[0]
            acct = W.new_account(created, domain, verification)
            W.add_membership(acct, cid)
            W.add_linkage(acct, created, "payment_token", payment, "high")
            W.add_linkage(acct, created, "stable_device", f"device-{h(cid, str(m))}", "medium")
            W.add_linkage(acct, created, "network_group", f"net-{h(cid, str(m))}", "medium")
            maybe_add_verification(rng, acct, created, verification)
            profile = profiles[m % len(profiles)]
            activity_end = clip(created + timedelta(days=rng.randint(300, 700)), created, WORLD_END)
            run_credit_cycles(rng, acct, created, activity_end, profile,
                               consume_events_range=(1, 5), consume_amt_range=(1, 4),
                               max_fraction=rng.uniform(0.2, 0.8))
            add_logins(rng, acct, created, activity_end, profile, n_range=(5, 20))


# --------------------------------------------------------------------------
# Family 14: Referral / Viral Signup Burst (legitimate)
# --------------------------------------------------------------------------

def fam_referral_burst(n_clusters=15):
    fam = "Referral / Viral Signup Burst"
    for i in range(n_clusters):
        cid = W.new_cluster(fam, "legitimate",
                             "Independent strangers converge in a short window via referral; fans out after.")
        network = f"net-{h(cid, 'net')}"
        size = rng.randint(10, 20)
        burst_start = pick_datetime(rng, WORLD_START, WORLD_END - timedelta(days=200), BURST_TIGHT)
        burst_end = clip(burst_start + timedelta(days=2), burst_start, WORLD_END)
        for m in range(size):
            created = pick_datetime(rng, burst_start, burst_end, BURST_TIGHT)
            domain = rng.choices(["consumer", "disposable"], weights=[0.9, 0.1])[0]
            verification = rng.choices(["email", "phone", "none"], weights=[0.5, 0.3, 0.2])[0]
            acct = W.new_account(created, domain, verification)
            W.add_membership(acct, cid)
            if rng.random() < 0.5:
                W.add_linkage(acct, created, "network_group", network, rng.choice(["low", "medium"]))
            maybe_add_verification(rng, acct, created, verification)
            # each account fans out into its own idiosyncratic profile
            profile = rng.choice([GENERAL_AWAKE, HOUSEHOLD_EVENING, MORNING_SHIFT, BUSINESS_HOURS])
            activity_end = clip(created + timedelta(days=rng.randint(30, 500)), created, WORLD_END)
            run_credit_cycles(rng, acct, created, activity_end, profile,
                               consume_events_range=(1, 5), consume_amt_range=(1, 4),
                               max_fraction=rng.uniform(0.2, 0.9), claim_prob=rng.uniform(0.5, 1.0))
            add_logins(rng, acct, created, activity_end, profile, n_range=(2, 15))


# --------------------------------------------------------------------------
# Family 15: Verification-Avoidant Privacy Cluster (ambiguous)
# --------------------------------------------------------------------------

def fam_privacy_cluster(n_clusters=15):
    fam = "Verification-Avoidant Privacy Cluster"
    for i in range(n_clusters):
        cid = W.new_cluster(fam, "ambiguous",
                             "Privacy-seeking distinct individuals sharing a coarse VPN/carrier network.")
        network = f"net-{h(cid, 'net')}"
        size = rng.randint(4, 8)
        for m in range(size):
            created = pick_datetime(rng, WORLD_START, WORLD_END - timedelta(days=30), GENERAL_AWAKE)
            domain = rng.choices(["disposable", "consumer"], weights=[0.7, 0.3])[0]
            verification = "none"
            acct = W.new_account(created, domain, verification)
            W.add_membership(acct, cid)
            W.add_linkage(acct, created, "network_group", network, "medium")
            if rng.random() < 0.25:
                W.add_event(acct, created + timedelta(minutes=rng.randint(1, 90)),
                            "verification_failed", 0, "low")
            profile = rng.choice([GENERAL_AWAKE, HOUSEHOLD_EVENING, MORNING_SHIFT, NIGHT_SHIFT])
            activity_end = clip(created + timedelta(days=rng.randint(60, 500)), created, WORLD_END)
            run_credit_cycles(rng, acct, created, activity_end, profile,
                               consume_events_range=(1, 4), consume_amt_range=(1, 3),
                               max_fraction=rng.uniform(0.2, 0.6))
            add_logins(rng, acct, created, activity_end, profile, n_range=(3, 12))


# --------------------------------------------------------------------------
# Family 16: Slow-Burn Multi-Account Hoarder (abusive)
# --------------------------------------------------------------------------

def fam_slow_burn_hoarder(n_clusters=8):
    fam = "Slow-Burn Multi-Account Hoarder"
    for i in range(n_clusters):
        cid = W.new_cluster(fam, "abusive",
                             "One operator spaces out accounts on distinct devices; thin recovery/behavioral thread.")
        link_kind = rng.choice(["recovery", "behavioral"])
        shared_val = f"{'rec' if link_kind == 'recovery' else 'behav'}-{h(cid, 'shared')}"
        size = rng.randint(3, 5)
        cursor = pick_datetime(rng, WORLD_START, WORLD_END - timedelta(days=500), GENERAL_AWAKE)
        for m in range(size):
            created = clip(cursor, WORLD_START, WORLD_END)
            domain = rng.choices(["consumer", "enterprise"], weights=[0.8, 0.2])[0]
            verification = rng.choices(["email", "none", "phone"], weights=[0.5, 0.3, 0.2])[0]
            acct = W.new_account(created, domain, verification)
            W.add_membership(acct, cid)
            W.add_linkage(acct, created, "stable_device", f"device-{h(cid, str(m))}", "high")
            W.add_linkage(acct, created, "network_group", f"net-{h(cid, str(m))}", "high")
            if link_kind == "recovery":
                W.add_linkage(acct, created, "recovery_contact", shared_val, "high")
            else:
                W.add_linkage(acct, created, "behavioral_template", shared_val, "medium")
            maybe_add_verification(rng, acct, created, verification)
            activity_end = clip(created + timedelta(days=rng.randint(200, 650)), created, WORLD_END)
            run_credit_cycles(rng, acct, created, activity_end, GENERAL_AWAKE,
                               consume_events_range=(1, 3), consume_amt_range=(1, 3),
                               max_fraction=rng.uniform(0.1, 0.32))
            add_logins(rng, acct, created, activity_end, GENERAL_AWAKE, n_range=(3, 10))
            cursor = created + timedelta(days=rng.randint(75, 210))


# --------------------------------------------------------------------------
# Build the world
# --------------------------------------------------------------------------

def build_world():
    fam_household_shared_device(40)
    fam_sequential_household(30)
    fam_kiosk(15)
    fam_small_business(25)
    fam_recovery_migration(35)
    fam_roommate_turnover(25)
    fam_classroom(10)
    fam_trial_farm(10)
    fam_serial_reroll(12)
    fam_mule_ring(8)
    fam_device_reseller(20)
    fam_hot_desk(20)
    fam_boarding_school(20)
    fam_referral_burst(15)
    fam_privacy_cluster(15)
    fam_slow_burn_hoarder(8)


ACCEPTED_FAMILIES = [
    "Household Shared Device",
    "Sequential Household Users",
    "Shared Public Kiosk / Library Terminal",
    "Small Business Turnover",
    "Account Recovery Migration",
    "Roommate Turnover",
    "Classroom / Course Cohort",
    "Trial Credit Farm",
    "Serial Reroll on Expiry",
    "Credit Mule Ring",
    "Device Reseller Fleet",
    "Co-working Hot Desk",
    "Boarding-School / Long-Distance Family",
    "Referral / Viral Signup Burst",
    "Verification-Avoidant Privacy Cluster",
    "Slow-Burn Multi-Account Hoarder",
]


def compute_entitlement_cycles():
    """Derive the entitlement_cycles table purely from W.accounts/W.events
    (no rng calls — does not touch or perturb the existing random stream).

    Each claim_credits event opens a fixed 30-day cycle window
    [claim_time, claim_time + 30d). Every consume_credits event is assigned
    to the most recent preceding claim_credits event for that account (this
    exactly mirrors how the generators above produced consumption: always
    after that cycle's own claim and always before the next cycle's claim,
    since the next claim can only occur at or after the current cycle's
    grant window closes). This makes the assignment robust to boundary
    clipping and avoids any need to re-derive cycle boundaries from signup
    arithmetic.
    """
    events_by_acct: dict[str, list[dict]] = {}
    for e in W.events:
        events_by_acct.setdefault(e["account_id"], []).append(e)

    rows = []
    cyc_n = 0
    # iterate accounts in a fixed, deterministic order
    for acct in sorted(events_by_acct.keys()):
        evs = events_by_acct[acct]
        claims = sorted((e for e in evs if e["event_type"] == "claim_credits"),
                         key=lambda e: e["timestamp"])
        if not claims:
            continue
        consumes = sorted((e for e in evs if e["event_type"] == "consume_credits"),
                           key=lambda e: e["timestamp"])
        claim_times = [c["timestamp"] for c in claims]
        consumed_per_claim = [0] * len(claims)
        for x in consumes:
            idx = bisect.bisect_right(claim_times, x["timestamp"]) - 1
            if idx < 0:
                # consume_credits before any claim should not occur by
                # construction; skip defensively rather than misattribute.
                continue
            consumed_per_claim[idx] += x["credits_amount"]

        for i, claim in enumerate(claims):
            cycle_start = claim["timestamp"]
            cycle_end = cycle_start + timedelta(days=30)
            granted = 25
            consumed = consumed_per_claim[i]
            closed = cycle_end <= WORLD_END
            expired = (granted - consumed) if closed else 0
            cyc_n += 1
            rows.append(dict(
                cycle_id=f"cyc-{cyc_n:08d}",
                account_id=acct,
                cycle_start=cycle_start,
                cycle_end=cycle_end,
                credits_granted=granted,
                credits_consumed=consumed,
                credits_expired=expired,
            ))
    return rows


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    build_world()

    accounts_df = pl.DataFrame(W.accounts).sort("account_id")
    events_df = pl.DataFrame(W.events).sort("event_id")
    linkage_df = pl.DataFrame(W.linkage).sort("observation_id")
    gt_df = pl.DataFrame(W.ground_truth).sort("cluster_id")
    membership_df = pl.DataFrame(W.membership).sort(["account_id", "cluster_id"])
    cycles_rows = compute_entitlement_cycles()
    cycles_df = pl.DataFrame(cycles_rows).sort("cycle_id")

    accounts_path = OUT_DIR / "accounts.parquet"
    events_path = OUT_DIR / "account_events.parquet"
    linkage_path = OUT_DIR / "linkage_observations.parquet"
    gt_path = OUT_DIR / "ground_truth.parquet"
    membership_path = OUT_DIR / "cluster_membership.parquet"
    cycles_path = OUT_DIR / "entitlement_cycles.parquet"

    accounts_df.write_parquet(accounts_path)
    events_df.write_parquet(events_path)
    linkage_df.write_parquet(linkage_path)
    gt_df.write_parquet(gt_path)
    membership_df.write_parquet(membership_path)
    cycles_df.write_parquet(cycles_path)

    clusters_by_label = (
        gt_df.group_by("label").len().sort("label").to_dicts()
    )
    clusters_by_label = {row["label"]: row["len"] for row in clusters_by_label}

    manifest = {
        "schema_version": "tier2-brief-v1.0",
        "world_version": "world-v1",
        "author_model": (
            "anthropic/claude-sonnet — stand-in for qwen per "
            "docs/tier2-decisions.md; SAME model family as the detector "
            "designer, so the two-tier model-family coupling is NOT broken"
        ),
        "seed": SEED,
        "world_end": "2026-06-30T00:00:00Z",
        "accepted_families": ACCEPTED_FAMILIES,
        "counts": {
            "accounts": accounts_df.height,
            "clusters": gt_df.height,
            "events": events_df.height,
            "linkage_observations": linkage_df.height,
            "cycles": W.cycle_count,
            "entitlement_cycles": cycles_df.height,
            "clusters_by_label": clusters_by_label,
        },
        "checksums": {
            "accounts.parquet": sha256_file(accounts_path),
            "account_events.parquet": sha256_file(events_path),
            "linkage_observations.parquet": sha256_file(linkage_path),
            "ground_truth.parquet": sha256_file(gt_path),
            "cluster_membership.parquet": sha256_file(membership_path),
            "entitlement_cycles.parquet": sha256_file(cycles_path),
        },
    }

    manifest_path = OUT_DIR / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=False)
        f.write("\n")

    print(json.dumps(manifest["counts"], indent=2))
    print(json.dumps(manifest["checksums"], indent=2))


if __name__ == "__main__":
    main()
