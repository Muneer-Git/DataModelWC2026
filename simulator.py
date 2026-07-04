"""
Simulation engine extracted from notebook 03.
Cached with streamlit so it only builds the model once.
"""
import numpy as np
import pandas as pd
from collections import defaultdict
from pathlib import Path
import streamlit as st

_ROOT = Path(__file__).parent

np.random.seed(2026)

GROUP_LABELS = 'ABCDEFGHIJKL'
SKIP_TOURNAMENTS = {'Friendly', 'Kirin Cup', 'Intercontinental Cup'}

GROUP_TEAMS = {
    'A': ['Algeria', 'Argentina', 'Austria', 'Jordan'],
    'B': ['Australia', 'Paraguay', 'Turkey', 'United States'],
    'C': ['Belgium', 'Egypt', 'Iran', 'New Zealand'],
    'D': ['Bosnia and Herzegovina', 'Canada', 'Qatar', 'Switzerland'],
    'E': ['Brazil', 'Haiti', 'Morocco', 'Scotland'],
    'F': ['Cape Verde', 'Saudi Arabia', 'Spain', 'Uruguay'],
    'G': ['Colombia', 'DR Congo', 'Portugal', 'Uzbekistan'],
    'H': ['Croatia', 'England', 'Ghana', 'Panama'],
    'I': ['Curaçao', 'Ecuador', 'Germany', 'Ivory Coast'],
    'J': ['Czech Republic', 'Mexico', 'South Africa', 'South Korea'],
    'K': ['France', 'Iraq', 'Norway', 'Senegal'],
    'L': ['Japan', 'Netherlands', 'Sweden', 'Tunisia'],
}
ALL_TEAMS = [t for g in GROUP_TEAMS.values() for t in g]
GROUP_MAP = {team: label for label, teams in GROUP_TEAMS.items() for team in teams}


@st.cache_resource(show_spinner="Building team strength model…")
def build_model():
    results = pd.read_csv(_ROOT / 'data/raw/results.csv', parse_dates=['date'])
    elo_df = pd.read_csv(_ROOT / 'data/processed/current_elo_ratings.csv')
    elo_map = dict(zip(elo_df['team'], elo_df['current_elo']))

    CUTOFF = pd.Timestamp('2026-06-01')
    RECENT_START = CUTOFF - pd.DateOffset(years=4)

    competitive = results[
        (results['date'] >= RECENT_START) &
        (results['date'] < CUTOFF) &
        results['home_score'].notna() &
        ~results['tournament'].isin(SKIP_TOURNAMENTS)
    ].copy()

    rows = []
    for _, r in competitive.iterrows():
        rows.append({'team': r['home_team'], 'gf': r['home_score'], 'ga': r['away_score']})
        rows.append({'team': r['away_team'], 'gf': r['away_score'], 'ga': r['home_score']})
    t = pd.DataFrame(rows)
    recent_stats = t.groupby('team').agg(gf_mean=('gf','mean'), ga_mean=('ga','mean'), n=('gf','count'))
    league_avg_gf = recent_stats['gf_mean'].mean()
    league_avg_ga = recent_stats['ga_mean'].mean()
    recent_stats['attack']  = recent_stats['gf_mean'] / league_avg_gf
    recent_stats['defence'] = recent_stats['ga_mean'] / league_avg_ga

    wc_elos  = {t: elo_map.get(t, 1500) for t in ALL_TEAMS}
    max_elo  = max(wc_elos.values())

    records = []
    for team in ALL_TEAMS:
        elo      = wc_elos[team]
        elo_frac = (elo - 1300) / (max_elo - 1300)
        proxy_atk = 0.5 + elo_frac * 1.0
        proxy_dfn = 1.5 - elo_frac * 1.0
        if team in recent_stats.index and recent_stats.loc[team, 'n'] >= 5:
            raw_atk = recent_stats.loc[team, 'attack']
            raw_dfn = recent_stats.loc[team, 'defence']
            n       = int(recent_stats.loc[team, 'n'])
            blend   = min(n / (n + 30), 0.70)
            atk     = blend * raw_atk + (1 - blend) * proxy_atk
            dfn     = blend * raw_dfn + (1 - blend) * proxy_dfn
        else:
            atk, dfn, n = proxy_atk, proxy_dfn, 0
        records.append({'team': team, 'elo': elo, 'attack': atk, 'defence': dfn,
                        'group': GROUP_MAP[team], 'n_recent': n})

    strength       = pd.DataFrame(records).set_index('team')
    WC_BASE        = 1.15
    GLOBAL_ATK_AVG = strength['attack'].mean()
    return strength, WC_BASE, GLOBAL_ATK_AVG


RHO = -0.13

def dc_correction(goals_a, goals_b, mu_a, mu_b, rho=RHO):
    if   goals_a == 0 and goals_b == 0: return 1 - mu_a * mu_b * rho
    elif goals_a == 0 and goals_b == 1: return 1 + mu_a * rho
    elif goals_a == 1 and goals_b == 0: return 1 + mu_b * rho
    elif goals_a == 1 and goals_b == 1: return 1 - rho
    else: return 1.0


def expected_goals(team_a, team_b, strength, WC_BASE, GLOBAL_ATK_AVG):
    s_a     = strength.loc[team_a]
    s_b     = strength.loc[team_b]
    avg_dfn = strength['defence'].mean()
    mu_a = WC_BASE * (s_a['attack'] / GLOBAL_ATK_AVG) * (max(s_b['defence'], 0.1) / avg_dfn)
    mu_b = WC_BASE * (s_b['attack'] / GLOBAL_ATK_AVG) * (max(s_a['defence'], 0.1) / avg_dfn)
    elo_prob_a = 1 / (1 + 10 ** ((s_b['elo'] - s_a['elo']) / 400))
    elo_adj    = (elo_prob_a - 0.5) * 0.30
    mu_a = max(mu_a * (1 + elo_adj), 0.10)
    mu_b = max(mu_b * (1 - elo_adj), 0.10)
    return mu_a, mu_b


def _simulate_shootout(p_a):
    """
    Simulate a penalty shootout.
    p_a: Elo-derived probability that team A wins the shootout.
    Returns (pens_a, pens_b) — total penalties scored by each side.
    Bias is applied via per-kick conversion rates derived from p_a.
    """
    conv_a = min(max(0.76 + (p_a - 0.5) * 0.10, 0.62), 0.90)
    conv_b = min(max(0.76 - (p_a - 0.5) * 0.10, 0.62), 0.90)
    pa, pb = 0, 0
    for _ in range(5):                          # first 5 kicks each
        pa += int(np.random.random() < conv_a)
        pb += int(np.random.random() < conv_b)
    while pa == pb:                             # sudden death
        pa += int(np.random.random() < conv_a)
        pb += int(np.random.random() < conv_b)
    return pa, pb


def simulate_match(team_a, team_b, strength, WC_BASE, GLOBAL_ATK_AVG, knockout=False):
    """
    Returns (goals_a, goals_b, winner, pen_a, pen_b).
    pen_a / pen_b are None when there are no penalties.
    """
    mu_a, mu_b = expected_goals(team_a, team_b, strength, WC_BASE, GLOBAL_ATK_AVG)
    for _ in range(50):
        ga = np.random.poisson(mu_a)
        gb = np.random.poisson(mu_b)
        if np.random.random() < dc_correction(ga, gb, mu_a, mu_b):
            break
    if knockout and ga == gb:
        ga += np.random.poisson(mu_a * (30 / 90) * 0.65)
        gb += np.random.poisson(mu_b * (30 / 90) * 0.65)
    if knockout and ga == gb:
        p_a = 1 / (1 + 10 ** ((strength.loc[team_b, 'elo'] - strength.loc[team_a, 'elo']) / 800))
        pen_a, pen_b = _simulate_shootout(p_a)
        winner = team_a if pen_a > pen_b else team_b
    else:
        pen_a, pen_b = None, None
        winner = team_a if ga > gb else (team_b if gb > ga else None)
    return ga, gb, winner, pen_a, pen_b


def simulate_knockout_from_bracket(bracket, strength, WC_BASE, GLOBAL_ATK_AVG, n_sims=1000):
    """
    Simulate tournament from a given bracket (list of (teamA, teamB) matchups).
    Returns DataFrame with team → win/finalist/SF/QF probability.
    """
    teams_in = list({t for pair in bracket for t in pair})
    rounds_order = ['R1', 'QF', 'SF', 'Final', 'Winner']

    counts = {t: defaultdict(int) for t in teams_in}
    # teams_in is always 2× len(bracket); log2(teams) gives correct round count
    # (e.g. Final: 2 teams → 1 round; QF: 8 teams → 3 rounds)
    n_rounds = int(np.log2(len(teams_in)))

    for _ in range(n_sims):
        current_bracket = list(bracket)
        reached = {t: 0 for t in teams_in}
        for rnd in range(n_rounds):
            winners = []
            for a, b in current_bracket:
                _, _, w, _, _ = simulate_match(a, b, strength, WC_BASE, GLOBAL_ATK_AVG, knockout=True)
                winners.append(w)
                reached[a] = max(reached[a], rnd)
                reached[b] = max(reached[b], rnd)
            for w in winners:
                reached[w] = rnd + 1
            current_bracket = list(zip(winners[::2], winners[1::2]))

        for t, r in reached.items():
            counts[t][r] += 1

    rows = []
    for t in teams_in:
        total = n_sims
        rows.append({
            'team': t,
            'Win%': counts[t].get(n_rounds, 0) / total * 100,
            'Final%': (counts[t].get(n_rounds, 0) + counts[t].get(n_rounds - 1, 0)) / total * 100,
            **{f'R{n_rounds - i}%': sum(counts[t].get(r, 0) for r in range(i, n_rounds + 1)) / total * 100
               for i in range(1, n_rounds)},
        })
    return pd.DataFrame(rows).sort_values('Win%', ascending=False)
