import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from pathlib import Path
from simulator import (
    build_model, simulate_match, simulate_knockout_from_bracket,
    expected_goals, ALL_TEAMS, GROUP_TEAMS, GROUP_MAP,
)

ROOT = Path(__file__).parent

def _h(desktop: int, mobile: int = None) -> int:
    """Return chart height — Streamlit has no JS screen detection so we use
    a single responsive value capped for readability on all screen sizes."""
    return min(desktop, mobile or desktop)

def simulate_forward(starting_bracket, n_rounds, strength, WC_BASE, GLOBAL_ATK_AVG):
    """
    Simulate n knockout rounds from starting_bracket.
    Returns (round_summaries, next_bracket).
    round_summaries: list of lists of (home, away, home_goals, away_goals, winner, pen_a, pen_b)
    next_bracket:   list of (teamA, teamB) pairs for the next round
    """
    current = list(starting_bracket)
    summaries = []
    for _ in range(n_rounds):
        winners, results = [], []
        for a, b in current:
            ga, gb, w, pa, pb = simulate_match(a, b, strength, WC_BASE, GLOBAL_ATK_AVG, knockout=True)
            winners.append(w)
            results.append((a, b, ga, gb, w, pa, pb))
        summaries.append(results)
        current = list(zip(winners[::2], winners[1::2]))
    return summaries, current

st.set_page_config(
    page_title="WC 2026 — Predictor",
    page_icon="",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Dark theme CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
  /* ── Design system: Dark Editorial ──────────────────────────────────── */
  :root {
    --bg:      #080810;
    --surface: #0f0f1a;
    --card:    #14141f;
    --border:  #222235;
    --text:    #ededf8;
    --muted:   #7070a0;
    --accent:  #6366f1;
    --gold:    #f59e0b;
    --green:   #10b981;
    --red:     #ef4444;
  }

  /* ── Base ────────────────────────────────────────────────────────────── */
  .stApp { background:var(--bg) !important; color:var(--text); }
  * { font-family:-apple-system,'Segoe UI','Inter',sans-serif; }
  section[data-testid="stSidebar"] {
    background:var(--surface);
    border-right:1px solid var(--border);
  }
  .block-container { padding-top:1.5rem !important; max-width:1400px; }

  /* ── Typography ──────────────────────────────────────────────────────── */
  h1,h2,h3,h4 { color:var(--text) !important; letter-spacing:-0.02em; }
  .stMarkdown p { color:#b0b0d0; line-height:1.6; }

  /* ── Tabs — underline style ──────────────────────────────────────────── */
  .stTabs [data-baseweb="tab-list"] {
    background:transparent !important;
    border-bottom:1px solid var(--border);
    padding:0 !important;
    gap:0 !important;
    overflow-x:auto;
    -webkit-overflow-scrolling:touch;
    flex-wrap:nowrap;
  }
  .stTabs [data-baseweb="tab"] {
    background:transparent !important;
    color:var(--muted);
    border-radius:0 !important;
    padding:14px 22px !important;
    font-size:11px !important;
    font-weight:700 !important;
    letter-spacing:0.1em !important;
    text-transform:uppercase !important;
    white-space:nowrap;
    flex-shrink:0;
    border-bottom:2px solid transparent !important;
    transition:color 0.15s, border-color 0.15s;
  }
  .stTabs [aria-selected="true"] {
    background:transparent !important;
    color:var(--text) !important;
    border-bottom:2px solid var(--accent) !important;
  }
  .stTabs [data-baseweb="tab"]:hover { color:var(--text) !important; }

  /* ── Metric cards ────────────────────────────────────────────────────── */
  .metric-card {
    background:var(--card);
    border:1px solid var(--border);
    border-top:2px solid var(--accent);
    border-radius:8px;
    padding:22px 16px;
    text-align:center;
    margin-bottom:8px;
  }
  .metric-value {
    font-size:1.85rem;
    font-weight:800;
    color:var(--text);
    letter-spacing:-0.03em;
    line-height:1.1;
  }
  .metric-label {
    font-size:0.65rem;
    color:var(--muted);
    margin-top:6px;
    text-transform:uppercase;
    letter-spacing:0.12em;
    font-weight:600;
  }

  /* ── Buttons ─────────────────────────────────────────────────────────── */
  .stButton > button {
    background:transparent !important;
    border:1px solid var(--border) !important;
    color:var(--text) !important;
    border-radius:6px !important;
    font-size:12px !important;
    font-weight:600 !important;
    letter-spacing:0.04em !important;
    padding:8px 18px !important;
    transition:border-color 0.15s, color 0.15s !important;
  }
  .stButton > button:hover {
    border-color:var(--accent) !important;
    color:var(--accent) !important;
  }
  .stButton > button[kind="primary"] {
    background:var(--accent) !important;
    border-color:var(--accent) !important;
    color:#fff !important;
  }
  .stButton > button[kind="primary"]:hover {
    background:#4f52d9 !important;
    border-color:#4f52d9 !important;
  }

  /* ── Inputs ──────────────────────────────────────────────────────────── */
  div[data-testid="stSelectbox"] label,
  div[data-testid="stMultiSelect"] label,
  div[data-testid="stSlider"] label {
    color:var(--muted) !important;
    font-size:10px !important;
    font-weight:700 !important;
    letter-spacing:0.1em !important;
    text-transform:uppercase !important;
  }
  .stSelectbox > div > div {
    background:var(--card) !important;
    border-color:var(--border) !important;
    color:var(--text) !important;
  }

  /* ── Divider ─────────────────────────────────────────────────────────── */
  hr { border:none !important; border-top:1px solid var(--border) !important; margin:28px 0 !important; }

  /* ── Expander ────────────────────────────────────────────────────────── */
  [data-testid="stExpander"] {
    border:1px solid var(--border) !important;
    border-radius:8px !important;
    background:var(--card) !important;
  }
  [data-testid="stExpander"] summary {
    font-size:12px !important;
    font-weight:700 !important;
    letter-spacing:0.04em !important;
    color:var(--muted) !important;
    text-transform:uppercase !important;
  }

  /* ── Info boxes ──────────────────────────────────────────────────────── */
  [data-testid="stInfo"] {
    background:rgba(99,102,241,0.07) !important;
    border:1px solid rgba(99,102,241,0.2) !important;
    border-radius:6px !important;
    color:var(--text) !important;
  }

  /* ── Dataframes ──────────────────────────────────────────────────────── */
  [data-testid="stDataFrame"] { border:1px solid var(--border); border-radius:6px; overflow:hidden; }

  /* ── Caption ─────────────────────────────────────────────────────────── */
  .stCaption, [data-testid="stCaptionContainer"] p {
    color:var(--muted) !important;
    font-size:11px !important;
    letter-spacing:0.02em !important;
  }

  /* ── Tablet (≤768px) ─────────────────────────────────────────────────── */
  @media (max-width:768px) {
    [data-testid="stHorizontalBlock"] { flex-wrap:wrap !important; gap:6px !important; }
    [data-testid="column"] { min-width:calc(50% - 6px) !important; flex:1 1 calc(50% - 6px) !important; }
    .stTabs [data-baseweb="tab"] { font-size:9px !important; padding:10px 12px !important; }
    .metric-card { padding:14px 10px !important; }
    .metric-value { font-size:1.4rem !important; }
    .block-container { padding:1rem 0.6rem 2rem !important; }
    [data-testid="stDataFrame"] { overflow-x:auto !important; }
    h2 { font-size:1.1rem !important; }
    h3 { font-size:0.95rem !important; }
    .wc-title { font-size:1.8rem !important; }
  }

  /* ── Small phone (≤480px) ────────────────────────────────────────────── */
  @media (max-width:480px) {
    [data-testid="column"] { min-width:100% !important; flex:1 1 100% !important; }
    .stTabs [data-baseweb="tab"] { font-size:8px !important; padding:8px 10px !important; }
    .metric-value { font-size:1.2rem !important; }
    .wc-title { font-size:1.3rem !important; }
    .block-container { padding:0.5rem 0.3rem 2rem !important; }
  }
</style>
""", unsafe_allow_html=True)

# ── Confederation mapping & colours ──────────────────────────────────────────
CONF = {
    'Spain':'UEFA','France':'UEFA','England':'UEFA','Germany':'UEFA','Portugal':'UEFA',
    'Netherlands':'UEFA','Belgium':'UEFA','Croatia':'UEFA','Switzerland':'UEFA',
    'Austria':'UEFA','Turkey':'UEFA','Norway':'UEFA','Czech Republic':'UEFA',
    'Bosnia and Herzegovina':'UEFA','Scotland':'UEFA','Sweden':'UEFA',
    'Argentina':'CONMEBOL','Brazil':'CONMEBOL','Colombia':'CONMEBOL','Uruguay':'CONMEBOL',
    'Ecuador':'CONMEBOL','Paraguay':'CONMEBOL',
    'Algeria':'CAF','Morocco':'CAF','Senegal':'CAF','Egypt':'CAF','Ivory Coast':'CAF',
    'South Africa':'CAF','Ghana':'CAF','DR Congo':'CAF','Tunisia':'CAF','Cape Verde':'CAF',
    'Japan':'AFC','South Korea':'AFC','Iran':'AFC','Saudi Arabia':'AFC','Australia':'AFC',
    'Iraq':'AFC','Jordan':'AFC','Uzbekistan':'AFC','Qatar':'AFC',
    'United States':'CONCACAF','Mexico':'CONCACAF','Canada':'CONCACAF','Panama':'CONCACAF',
    'Haiti':'CONCACAF','Curaçao':'CONCACAF',
    'New Zealand':'OFC',
}
CONF_COLORS = {
    'UEFA':     '#6366f1',
    'CONMEBOL': '#10b981',
    'AFC':      '#8b5cf6',
    'CAF':      '#f59e0b',
    'CONCACAF': '#ef4444',
    'OFC':      '#06b6d4',
}

GROUP_TEAMS = {
    'A': ['Algeria','Argentina','Austria','Jordan'],
    'B': ['Australia','Paraguay','Turkey','United States'],
    'C': ['Belgium','Egypt','Iran','New Zealand'],
    'D': ['Bosnia and Herzegovina','Canada','Qatar','Switzerland'],
    'E': ['Brazil','Haiti','Morocco','Scotland'],
    'F': ['Cape Verde','Saudi Arabia','Spain','Uruguay'],
    'G': ['Colombia','DR Congo','Portugal','Uzbekistan'],
    'H': ['Croatia','England','Ghana','Panama'],
    'I': ['Curaçao','Ecuador','Germany','Ivory Coast'],
    'J': ['Czech Republic','Mexico','South Africa','South Korea'],
    'K': ['France','Iraq','Norway','Senegal'],
    'L': ['Japan','Netherlands','Sweden','Tunisia'],
}

# ── Live tournament data (WC 2026, updated July 4) ───────────────────────────
# Group labels here are the actual FIFA labels (A–L), different from notebook auto-labels
ACTUAL_GROUP_STANDINGS = {
    'A': [
        {'display':'Mexico',        'model':'Mexico',        'mp':3,'w':3,'d':0,'l':0,'gf':6, 'ga':0, 'gd':6,  'pts':9, 'q':'W'},
        {'display':'South Africa',  'model':'South Africa',  'mp':3,'w':1,'d':1,'l':1,'gf':2, 'ga':3, 'gd':-1, 'pts':4, 'q':'R'},
        {'display':'South Korea',   'model':'South Korea',   'mp':3,'w':1,'d':0,'l':2,'gf':2, 'ga':3, 'gd':-1, 'pts':3, 'q':''},
        {'display':'Czechia',       'model':'Czech Republic','mp':3,'w':0,'d':1,'l':2,'gf':2, 'ga':6, 'gd':-4, 'pts':1, 'q':''},
    ],
    'B': [
        {'display':'Switzerland',           'model':'Switzerland',           'mp':3,'w':2,'d':1,'l':0,'gf':7, 'ga':3, 'gd':4,  'pts':7, 'q':'W'},
        {'display':'Canada',                'model':'Canada',                'mp':3,'w':1,'d':1,'l':1,'gf':8, 'ga':3, 'gd':5,  'pts':4, 'q':'R'},
        {'display':'Bosnia and Herzegovina','model':'Bosnia and Herzegovina','mp':3,'w':1,'d':1,'l':1,'gf':5, 'ga':6, 'gd':-1, 'pts':4, 'q':'3'},
        {'display':'Qatar',                 'model':'Qatar',                 'mp':3,'w':0,'d':1,'l':2,'gf':2, 'ga':10,'gd':-8, 'pts':1, 'q':''},
    ],
    'C': [
        {'display':'Brazil',   'model':'Brazil',   'mp':3,'w':2,'d':1,'l':0,'gf':7,'ga':1,'gd':6, 'pts':7,'q':'W'},
        {'display':'Morocco',  'model':'Morocco',  'mp':3,'w':2,'d':1,'l':0,'gf':6,'ga':3,'gd':3, 'pts':7,'q':'R'},
        {'display':'Scotland', 'model':'Scotland', 'mp':3,'w':1,'d':0,'l':2,'gf':1,'ga':4,'gd':-3,'pts':3,'q':''},
        {'display':'Haiti',    'model':'Haiti',    'mp':3,'w':0,'d':0,'l':3,'gf':2,'ga':8,'gd':-6,'pts':0,'q':''},
    ],
    'D': [
        {'display':'USA',       'model':'United States','mp':3,'w':2,'d':0,'l':1,'gf':8,'ga':4,'gd':4, 'pts':6,'q':'W'},
        {'display':'Australia', 'model':'Australia',    'mp':3,'w':1,'d':1,'l':1,'gf':2,'ga':2,'gd':0, 'pts':4,'q':'R'},
        {'display':'Paraguay',  'model':'Paraguay',     'mp':3,'w':1,'d':1,'l':1,'gf':2,'ga':4,'gd':-2,'pts':4,'q':'3'},
        {'display':'Türkiye',   'model':'Turkey',       'mp':3,'w':1,'d':0,'l':2,'gf':3,'ga':5,'gd':-2,'pts':3,'q':''},
    ],
    'E': [
        {'display':'Germany',       'model':'Germany',      'mp':3,'w':2,'d':0,'l':1,'gf':10,'ga':4,'gd':6, 'pts':6,'q':'W'},
        {"display":"Côte d'Ivoire", 'model':'Ivory Coast',  'mp':3,'w':2,'d':0,'l':1,'gf':4, 'ga':2,'gd':2, 'pts':6,'q':'R'},
        {'display':'Ecuador',       'model':'Ecuador',      'mp':3,'w':1,'d':1,'l':1,'gf':2, 'ga':2,'gd':0, 'pts':4,'q':'3'},
        {'display':'Curaçao',       'model':'Curaçao',      'mp':3,'w':0,'d':1,'l':2,'gf':1, 'ga':9,'gd':-8,'pts':1,'q':''},
    ],
    'F': [
        {'display':'Netherlands','model':'Netherlands','mp':3,'w':2,'d':1,'l':0,'gf':10,'ga':4, 'gd':6,  'pts':7,'q':'W'},
        {'display':'Japan',      'model':'Japan',      'mp':3,'w':1,'d':2,'l':0,'gf':7, 'ga':3, 'gd':4,  'pts':5,'q':'R'},
        {'display':'Sweden',     'model':'Sweden',     'mp':3,'w':1,'d':1,'l':1,'gf':7, 'ga':7, 'gd':0,  'pts':4,'q':'3'},
        {'display':'Tunisia',    'model':'Tunisia',    'mp':3,'w':0,'d':0,'l':3,'gf':2, 'ga':12,'gd':-10,'pts':0,'q':''},
    ],
    'G': [
        {'display':'Belgium',     'model':'Belgium',     'mp':3,'w':1,'d':2,'l':0,'gf':6,'ga':2, 'gd':4, 'pts':5,'q':'W'},
        {'display':'Egypt',       'model':'Egypt',       'mp':3,'w':1,'d':2,'l':0,'gf':5,'ga':3, 'gd':2, 'pts':5,'q':'R'},
        {'display':'Iran',        'model':'Iran',        'mp':3,'w':0,'d':3,'l':0,'gf':3,'ga':3, 'gd':0, 'pts':3,'q':''},
        {'display':'New Zealand', 'model':'New Zealand', 'mp':3,'w':0,'d':1,'l':2,'gf':4,'ga':10,'gd':-6,'pts':1,'q':''},
    ],
    'H': [
        {'display':'Spain',       'model':'Spain',       'mp':3,'w':2,'d':1,'l':0,'gf':5,'ga':0,'gd':5, 'pts':7,'q':'W'},
        {'display':'Cabo Verde',  'model':'Cape Verde',  'mp':3,'w':0,'d':3,'l':0,'gf':2,'ga':2,'gd':0, 'pts':3,'q':'R'},
        {'display':'Uruguay',     'model':'Uruguay',     'mp':3,'w':0,'d':2,'l':1,'gf':3,'ga':4,'gd':-1,'pts':2,'q':''},
        {'display':'Saudi Arabia','model':'Saudi Arabia','mp':3,'w':0,'d':2,'l':1,'gf':1,'ga':5,'gd':-4,'pts':2,'q':''},
    ],
    'I': [
        {'display':'France',  'model':'France',  'mp':3,'w':3,'d':0,'l':0,'gf':10,'ga':2,'gd':8, 'pts':9,'q':'W'},
        {'display':'Norway',  'model':'Norway',  'mp':3,'w':2,'d':0,'l':1,'gf':8, 'ga':7,'gd':1, 'pts':6,'q':'R'},
        {'display':'Senegal', 'model':'Senegal', 'mp':3,'w':1,'d':0,'l':2,'gf':8, 'ga':6,'gd':2, 'pts':3,'q':'3'},
        {'display':'Iraq',    'model':'Iraq',    'mp':3,'w':0,'d':0,'l':3,'gf':1, 'ga':12,'gd':-11,'pts':0,'q':''},
    ],
    'J': [
        {'display':'Argentina',  'model':'Argentina',  'mp':3,'w':3,'d':0,'l':0,'gf':8,'ga':1,'gd':7, 'pts':9,'q':'W'},
        {'display':'Austria',    'model':'Austria',    'mp':3,'w':1,'d':1,'l':1,'gf':6,'ga':6,'gd':0, 'pts':4,'q':'R'},
        {'display':'Algeria',    'model':'Algeria',    'mp':3,'w':1,'d':1,'l':1,'gf':5,'ga':7,'gd':-2,'pts':4,'q':'3'},
        {'display':'Jordan',     'model':'Jordan',     'mp':3,'w':0,'d':0,'l':3,'gf':3,'ga':8,'gd':-5,'pts':0,'q':''},
    ],
    'K': [
        {'display':'Colombia', 'model':'Colombia', 'mp':3,'w':2,'d':1,'l':0,'gf':4,'ga':1,'gd':3, 'pts':7,'q':'W'},
        {'display':'Portugal', 'model':'Portugal', 'mp':3,'w':1,'d':2,'l':0,'gf':6,'ga':1,'gd':5, 'pts':5,'q':'R'},
        {'display':'DR Congo', 'model':'DR Congo', 'mp':3,'w':1,'d':1,'l':1,'gf':4,'ga':3,'gd':1, 'pts':4,'q':'3'},
        {'display':'Uzbekistan','model':'Uzbekistan','mp':3,'w':0,'d':0,'l':3,'gf':2,'ga':11,'gd':-9,'pts':0,'q':''},
    ],
    'L': [
        {'display':'England', 'model':'England', 'mp':3,'w':2,'d':1,'l':0,'gf':6,'ga':2,'gd':4,'pts':7,'q':'W'},
        {'display':'Croatia', 'model':'Croatia', 'mp':3,'w':2,'d':0,'l':1,'gf':5,'ga':5,'gd':0,'pts':6,'q':'R'},
        {'display':'Ghana',   'model':'Ghana',   'mp':3,'w':1,'d':1,'l':1,'gf':2,'ga':2,'gd':0,'pts':4,'q':'3'},
        {'display':'Panama',  'model':'Panama',  'mp':3,'w':0,'d':0,'l':3,'gf':0,'ga':4,'gd':-4,'pts':0,'q':''},
    ],
}

# Best 8 third-place teams (by pts → GD → GF): DR Congo, Sweden, Ecuador, Ghana, Bosnia, Algeria, Paraguay, Senegal
BEST_THIRDS = ['DR Congo','Sweden','Ecuador','Ghana','Bosnia and Herzegovina','Algeria','Paraguay','Senegal']

# Completed R32 results
ACTUAL_R32 = [
    {'home':'South Africa', 'away':'Canada',                'hs':0,  'as':1,  'pen':None,            'winner':'Canada'},
    {'home':'Netherlands',  'away':'Morocco',               'hs':1,  'as':1,  'pen':'Morocco 3-2',   'winner':'Morocco'},
    {'home':'Germany',      'away':'Paraguay',              'hs':1,  'as':1,  'pen':'Paraguay 4-3',  'winner':'Paraguay'},
    {'home':'France',       'away':'Sweden',                'hs':3,  'as':0,  'pen':None,            'winner':'France'},
    {'home':'Belgium',      'away':'Senegal',               'hs':3,  'as':2,  'pen':None,            'winner':'Belgium'},
    {'home':'United States','away':'Bosnia and Herzegovina','hs':2,  'as':0,  'pen':None,            'winner':'United States'},
    {'home':'Spain',        'away':'Austria',               'hs':3,  'as':0,  'pen':None,            'winner':'Spain'},
    {'home':'Portugal',     'away':'Croatia',               'hs':2,  'as':1,  'pen':None,            'winner':'Portugal'},
    {'home':'Brazil',       'away':'Japan',                 'hs':2,  'as':1,  'pen':None,            'winner':'Brazil'},
    {'home':'Ivory Coast',  'away':'Norway',                'hs':1,  'as':2,  'pen':None,            'winner':'Norway'},
    {'home':'Mexico',       'away':'Ecuador',               'hs':2,  'as':0,  'pen':None,            'winner':'Mexico'},
    {'home':'England',      'away':'DR Congo',              'hs':2,  'as':1,  'pen':None,            'winner':'England'},
    {'home':'Switzerland',  'away':'Algeria',               'hs':2,  'as':0,  'pen':None,            'winner':'Switzerland'},
    {'home':'Colombia',     'away':'Ghana',                 'hs':1,  'as':0,  'pen':None,            'winner':'Colombia'},
    {'home':'Australia',    'away':'Egypt',                 'hs':1,  'as':1,  'pen':'Egypt 4-2',     'winner':'Egypt'},
    {'home':'Argentina',    'away':'Cape Verde',            'hs':3,  'as':2,  'pen':None,            'winner':'Argentina'},
]

# Actual FIFA group assignments (model team names)
FIFA_GROUPS = {
    'A': ['Mexico',        'South Africa', 'South Korea',           'Czech Republic'],
    'B': ['Switzerland',   'Canada',       'Bosnia and Herzegovina', 'Qatar'],
    'C': ['Brazil',        'Morocco',      'Scotland',               'Haiti'],
    'D': ['United States', 'Australia',    'Paraguay',               'Turkey'],
    'E': ['Germany',       'Ivory Coast',  'Ecuador',                'Curaçao'],
    'F': ['Netherlands',   'Japan',        'Sweden',                 'Tunisia'],
    'G': ['Belgium',       'Egypt',        'Iran',                   'New Zealand'],
    'H': ['Spain',         'Cape Verde',   'Uruguay',                'Saudi Arabia'],
    'I': ['France',        'Norway',       'Senegal',                'Iraq'],
    'J': ['Argentina',     'Austria',      'Algeria',                'Jordan'],
    'K': ['Colombia',      'Portugal',     'DR Congo',               'Uzbekistan'],
    'L': ['England',       'Croatia',      'Ghana',                  'Panama'],
}

# Round of 16 bracket (upcoming — all scheduled)
ACTUAL_R16_BRACKET = [
    ('Canada',        'Morocco'),
    ('Paraguay',      'France'),
    ('United States', 'Belgium'),
    ('Portugal',      'Spain'),
    ('Brazil',        'Norway'),
    ('Mexico',        'England'),
    ('Switzerland',   'Colombia'),
    ('Argentina',     'Egypt'),
]

# ── Data loading ──────────────────────────────────────────────────────────────
@st.cache_data
def load_predictions():
    df = pd.read_csv(ROOT / 'models/wc2026_predictions.csv')
    df['confederation'] = df['team'].map(CONF).fillna('UEFA')
    df['conf_color'] = df['confederation'].map(CONF_COLORS)
    return df

@st.cache_data
def load_results():
    df = pd.read_csv(ROOT / 'data/raw/results.csv', parse_dates=['date'])
    df['year'] = df['date'].dt.year
    df['total_goals'] = df['home_score'] + df['away_score']
    return df

@st.cache_data
def load_elo():
    return pd.read_csv(ROOT / 'data/processed/current_elo_ratings.csv')

pred  = load_predictions()
hist  = load_results()
elo   = load_elo()

PLOT_LAYOUT = dict(
    paper_bgcolor='#080810', plot_bgcolor='#0f0f1a',
    font=dict(color='#ededf8', family='-apple-system, "Segoe UI", sans-serif', size=12),
    margin=dict(l=10, r=10, t=44, b=10),
    title_font=dict(size=13, color='#ededf8', family='-apple-system, "Segoe UI", sans-serif'),
)
AXIS_STYLE = dict(gridcolor='#1e1e30', linecolor='#222235', zerolinecolor='#222235',
                  tickfont=dict(color='#7070a0', size=11))

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="padding:32px 0 24px; border-bottom:1px solid #222235; margin-bottom:24px">
  <div style="font-size:10px;font-weight:700;letter-spacing:0.18em;color:#6366f1;
              text-transform:uppercase;margin-bottom:10px">
    World Cup 2026 &nbsp;·&nbsp; Monte Carlo Predictor
  </div>
  <div class="wc-title"
       style="font-size:2.6rem;font-weight:800;letter-spacing:-0.04em;
              color:#ededf8;line-height:1.05;margin-bottom:10px">
    FIFA World Cup 2026
  </div>
  <div style="font-size:0.82rem;color:#7070a0;letter-spacing:0.02em">
    Dixon-Coles Poisson model &nbsp;&nbsp;·&nbsp;&nbsp;
    Blended Elo + recent attack/defence ratings &nbsp;&nbsp;·&nbsp;&nbsp;
    150 years of international results
  </div>
</div>
""", unsafe_allow_html=True)

# ── Top KPI cards ─────────────────────────────────────────────────────────────
top1  = pred.iloc[0]
top3  = pred.head(3)
favs  = ", ".join(top3['team'].tolist())
wc_count = hist[hist['tournament'] == 'FIFA World Cup']['year'].nunique()

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown(f"""<div class="metric-card">
      <div class="metric-value">{top1['team']}</div>
      <div class="metric-label">Tournament favourite</div></div>""", unsafe_allow_html=True)
with c2:
    st.markdown(f"""<div class="metric-card">
      <div class="metric-value">{top1['Win%']:.1f}%</div>
      <div class="metric-label">{top1['team']} win probability</div></div>""", unsafe_allow_html=True)
with c3:
    st.markdown(f"""<div class="metric-card">
      <div class="metric-value">48</div>
      <div class="metric-label">Teams qualified</div></div>""", unsafe_allow_html=True)
with c4:
    st.markdown(f"""<div class="metric-card">
      <div class="metric-value">{len(hist):,}</div>
      <div class="metric-label">Historical matches analysed</div></div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "Win Odds", "Advancement", "Groups", "Strength", "History", "Simulate", "Live"
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Win Odds
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown("### Championship Probability — All 48 Teams")

    n_teams = st.slider("Show top N teams", 10, 48, 24, key="n_win")
    top_n = pred.head(n_teams).copy()

    fig = go.Figure()
    for conf, color in CONF_COLORS.items():
        subset = top_n[top_n['confederation'] == conf]
        if subset.empty:
            continue
        fig.add_trace(go.Bar(
            y=subset['team'],
            x=subset['Win%'],
            orientation='h',
            name=conf,
            marker_color=color,
            marker_line_color='#21262d',
            marker_line_width=0.5,
            text=[f"  {v:.1f}%  Elo {e}" for v, e in zip(subset['Win%'], subset['Elo'])],
            textposition='outside',
            textfont=dict(size=10, color='#8b949e'),
            hovertemplate=(
                "<b>%{y}</b><br>"
                "Win: %{x:.1f}%<br>"
                "Group: %{customdata[0]}<br>"
                "Elo: %{customdata[1]}<extra></extra>"
            ),
            customdata=subset[['Grp', 'Elo']].values,
        ))

    fig.update_layout(
        **PLOT_LAYOUT,
        height=min(max(420, n_teams * 26), 900),
        barmode='overlay',
        yaxis=dict(autorange='reversed', gridcolor='#21262d', linecolor='#30363d'),
        xaxis=dict(title='Championship probability (%)', gridcolor='#21262d'),
        legend=dict(
            orientation='h', yanchor='bottom', y=1.01, xanchor='right', x=1,
            bgcolor='#161b22', bordercolor='#30363d', borderwidth=1,
        ),
        title=dict(text='Win Probability by Confederation', font=dict(size=14)),
    )
    st.plotly_chart(fig, width="stretch")

    # Pie chart — win% share by confederation
    conf_win = pred.groupby('confederation')['Win%'].sum().reset_index()
    conf_win.columns = ['Confederation', 'Total Win%']
    conf_win = conf_win.sort_values('Total Win%', ascending=False)
    conf_win['color'] = conf_win['Confederation'].map(CONF_COLORS)

    col_a, col_b = st.columns([1, 1])
    with col_a:
        st.markdown("#### Confederation share of total win probability")
        fig_pie = go.Figure(go.Pie(
            labels=conf_win['Confederation'],
            values=conf_win['Total Win%'],
            marker_colors=conf_win['color'].tolist(),
            hole=0.45,
            textinfo='label+percent',
            textfont_size=12,
            hovertemplate="<b>%{label}</b><br>Combined win%: %{value:.1f}%<extra></extra>",
        ))
        fig_pie.update_layout(
            **PLOT_LAYOUT, height=380,
            legend=dict(bgcolor='#161b22', bordercolor='#30363d'),
        )
        st.plotly_chart(fig_pie, width="stretch")

    with col_b:
        st.markdown("#### Top 15 — full odds table")
        display = pred[['team','Grp','Elo','Win%','Final%','SF%','QF%','R16%','R32%']].head(15).copy()
        display.columns = ['Team','Grp','Elo','Win%','Final%','SF%','QF%','R16%','R32%']
        st.dataframe(
            display.style.background_gradient(subset=['Win%','Final%','SF%'], cmap='YlOrRd')
                         .format({'Elo': '{:.0f}', 'Win%': '{:.1f}%', 'Final%': '{:.1f}%',
                                  'SF%': '{:.1f}%', 'QF%': '{:.1f}%', 'R16%': '{:.1f}%',
                                  'R32%': '{:.1f}%'}),
            width="stretch", height=450,
        )

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Round-by-Round Heatmap
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("### Advancement Probability — Round by Round")

    rounds = ['R32%','R16%','QF%','SF%','Final%','Win%']
    round_labels = ['Round of 32','Round of 16','Quarter-final','Semi-final','Final','Winner']

    sort_by = st.selectbox("Sort by", options=rounds, index=5, key="hm_sort",
                           format_func=lambda x: round_labels[rounds.index(x)])
    n_hm = st.slider("Teams to show", 16, 48, 32, key="hm_n")

    hm_data = pred.sort_values(sort_by, ascending=False).head(n_hm)
    z = hm_data[rounds].values
    y_labels = [f"{t}  [{g}]" for t, g in zip(hm_data['team'], hm_data['Grp'])]

    fig_hm = go.Figure(go.Heatmap(
        z=z,
        x=round_labels,
        y=y_labels,
        colorscale='YlOrRd',
        zmin=0, zmax=100,
        text=[[f"{v:.0f}%" for v in row] for row in z],
        texttemplate="%{text}",
        textfont=dict(size=10),
        hovertemplate="<b>%{y}</b><br>%{x}: %{z:.1f}%<extra></extra>",
        colorbar=dict(title='%', ticksuffix='%', bgcolor='#161b22', bordercolor='#30363d'),
    ))
    fig_hm.update_layout(
        **PLOT_LAYOUT,
        height=min(max(500, n_hm * 22), 1000),
        yaxis=dict(autorange='reversed', tickfont=dict(size=10), gridcolor='#21262d'),
        xaxis=dict(side='top', tickfont=dict(size=11), gridcolor='#21262d'),
        title=dict(text=f'Round-by-round advancement probabilities (top {n_hm} teams)', font=dict(size=14)),
    )
    st.plotly_chart(fig_hm, width="stretch")

    # Stacked bar — expected round reached
    st.markdown("### Expected Round Reached — Top 24")
    top24 = pred.head(24).copy()
    increments = {
        'Group stage exit': 100 - top24['R32%'],
        'R32 exit':         top24['R32%'] - top24['R16%'],
        'R16 exit':         top24['R16%'] - top24['QF%'],
        'QF exit':          top24['QF%']  - top24['SF%'],
        'SF exit':          top24['SF%']  - top24['Final%'],
        'Runner-up':        top24['Final%'] - top24['Win%'],
        'Champion':         top24['Win%'],
    }
    stage_colors = ['#30363d','#484f58','#6e7681','#ffa657','#d2a8ff','#3fb950','#ffd700']

    fig_stack = go.Figure()
    for (label, vals), color in zip(increments.items(), stage_colors):
        fig_stack.add_trace(go.Bar(
            name=label,
            x=top24['team'],
            y=vals,
            marker_color=color,
            hovertemplate=f"<b>%{{x}}</b><br>{label}: %{{y:.1f}}%<extra></extra>",
        ))
    fig_stack.update_layout(
        **PLOT_LAYOUT, barmode='stack', height=420,
        xaxis=dict(tickangle=-35, gridcolor='#21262d'),
        yaxis=dict(title='Probability (%)', gridcolor='#21262d'),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1,
                    bgcolor='#161b22', bordercolor='#30363d'),
        title=dict(text='Expected round reached breakdown (top 24)', font=dict(size=14)),
    )
    st.plotly_chart(fig_stack, width="stretch")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Groups
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("### Group Stage Analysis")

    sel_group = st.selectbox(
        "Select a group", list(FIFA_GROUPS.keys()),
        format_func=lambda g: f"Group {g}  —  {', '.join(FIFA_GROUPS[g])}",
        key="grp_sel",
    )

    grp_teams = FIFA_GROUPS[sel_group]
    grp_data  = pred[pred['team'].isin(grp_teams)].copy()
    grp_data  = grp_data.sort_values('Win%', ascending=False)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"#### Group {sel_group} — Win & Knockout Odds")
        fig_grp = go.Figure()

        categories = ['Win%', 'Final%', 'SF%', 'QF%', 'R16%', 'R32%']
        cat_labels  = ['Win', 'Final', 'Semi', 'QF', 'R16', 'R32']

        for _, row in grp_data.iterrows():
            color = CONF_COLORS.get(CONF.get(row['team'], 'UEFA'), '#58a6ff')
            fig_grp.add_trace(go.Bar(
                name=row['team'],
                x=cat_labels,
                y=[row[c] for c in categories],
                marker_color=color,
                hovertemplate=f"<b>{row['team']}</b><br>%{{x}}: %{{y:.1f}}%<extra></extra>",
            ))

        fig_grp.update_layout(
            **PLOT_LAYOUT, barmode='group', height=380,
            xaxis=dict(title='Stage', gridcolor='#21262d'),
            yaxis=dict(title='Probability (%)', gridcolor='#21262d'),
            legend=dict(bgcolor='#161b22', bordercolor='#30363d'),
            title=dict(text=f'Group {sel_group} knockout probability by stage', font=dict(size=13)),
        )
        st.plotly_chart(fig_grp, width="stretch")

    with col2:
        st.markdown(f"#### Group {sel_group} — Qualification share (R32)")
        pie_colors = [CONF_COLORS.get(CONF.get(t, 'UEFA'), '#8b949e') for t in grp_data['team']]
        fig_pie2 = go.Figure(go.Pie(
            labels=grp_data['team'],
            values=grp_data['R32%'],
            marker_colors=pie_colors,
            hole=0.4,
            textinfo='label+percent',
            hovertemplate="<b>%{label}</b><br>Qualify probability: %{value:.1f}%<extra></extra>",
        ))
        fig_pie2.update_layout(
            **PLOT_LAYOUT, height=380,
            legend=dict(bgcolor='#161b22', bordercolor='#30363d'),
            title=dict(text='Share of R32 qualification probability', font=dict(size=13)),
        )
        st.plotly_chart(fig_pie2, width="stretch")

    # Stats table for group
    st.markdown(f"#### Group {sel_group} — detailed odds")
    grp_display = grp_data[['team','Elo','R32%','R16%','QF%','SF%','Final%','Win%']].copy()
    grp_display.columns = ['Team','Elo','Qualify','R16','QF','SF','Final','Win']
    st.dataframe(
        grp_display.style
            .background_gradient(subset=['Qualify','R16','QF','SF','Final','Win'], cmap='YlOrRd')
            .format({'Elo': '{:.0f}', 'Qualify': '{:.1f}%', 'R16': '{:.1f}%',
                     'QF': '{:.1f}%', 'SF': '{:.1f}%', 'Final': '{:.1f}%', 'Win': '{:.1f}%'}),
        width="stretch", height=200,
    )

    st.divider()
    st.markdown("### All 12 Groups — Win Probability Overview")

    fig_all = make_subplots(rows=3, cols=4, subplot_titles=[f"Group {g}" for g in FIFA_GROUPS],
                             vertical_spacing=0.12, horizontal_spacing=0.06)
    for idx, (g_label, g_teams) in enumerate(FIFA_GROUPS.items()):
        row, col = divmod(idx, 4)
        g_data = pred[pred['team'].isin(g_teams)].sort_values('Win%', ascending=False)
        for _, r in g_data.iterrows():
            color = CONF_COLORS.get(CONF.get(r['team'], 'UEFA'), '#8b949e')
            fig_all.add_trace(go.Bar(
                x=[r['team'].split()[-1]],
                y=[r['Win%']],
                marker_color=color,
                showlegend=False,
                hovertemplate=f"<b>{r['team']}</b><br>Win: {r['Win%']:.1f}%<br>Elo: {r['Elo']}<extra></extra>",
                name=r['team'],
            ), row=row+1, col=col+1)
        fig_all.update_xaxes(tickfont=dict(size=7), row=row+1, col=col+1,
                             gridcolor='#21262d', linecolor='#30363d')
        fig_all.update_yaxes(gridcolor='#21262d', linecolor='#30363d', row=row+1, col=col+1)

    fig_all.update_layout(
        **PLOT_LAYOUT, height=600, showlegend=False,
        title=dict(text='Win % per group (sorted by probability)', font=dict(size=14)),
    )
    st.plotly_chart(fig_all, width="stretch")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — Team Strength scatter
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown("### Team Strength Profile — Attack vs Defence vs Elo")

    elo_top = elo.sort_values('current_elo', ascending=False).head(48)

    col_a, col_b = st.columns([3, 1])
    with col_b:
        st.markdown("#### Elo Leaderboard (top 20)")
        elo_display = elo_top.head(20)[['team','current_elo']].copy()
        elo_display.columns = ['Team', 'Elo']
        elo_display['Elo'] = elo_display['Elo'].round(0).astype(int)
        elo_display = elo_display.reset_index(drop=True)
        elo_display.index += 1
        st.dataframe(elo_display, width="stretch", height=400)

    with col_a:
        # Merge elo with predictions
        merged = pred.merge(elo_top[['team','current_elo']], on='team', how='left')
        merged['conf'] = merged['team'].map(CONF).fillna('UEFA')
        merged['color'] = merged['conf'].map(CONF_COLORS)
        merged['size'] = 8 + merged['Win%'] * 4

        fig_sc = go.Figure()
        for conf, color in CONF_COLORS.items():
            sub = merged[merged['conf'] == conf]
            if sub.empty:
                continue
            fig_sc.add_trace(go.Scatter(
                x=sub['current_elo'],
                y=sub['Win%'],
                mode='markers+text',
                name=conf,
                text=sub['team'],
                textposition='top center',
                textfont=dict(size=8, color='#c9d1d9'),
                marker=dict(
                    color=color, size=sub['size'],
                    line=dict(color='#30363d', width=0.5),
                    opacity=0.85,
                ),
                hovertemplate=(
                    "<b>%{text}</b><br>"
                    "Elo: %{x:.0f}<br>"
                    "Win%: %{y:.1f}%<extra></extra>"
                ),
            ))

        fig_sc.update_layout(
            **PLOT_LAYOUT, height=480,
            xaxis=dict(title='Current Elo Rating', gridcolor='#21262d'),
            yaxis=dict(title='Win Probability (%)', gridcolor='#21262d'),
            legend=dict(bgcolor='#161b22', bordercolor='#30363d', borderwidth=1),
            title=dict(text='Elo vs Win Probability (size = win%, colour = confederation)', font=dict(size=14)),
        )
        st.plotly_chart(fig_sc, width="stretch")

    # Elo distribution by confederation (violin/box)
    st.markdown("### Elo Distribution by Confederation")
    elo_all = elo.copy()
    elo_all['conf'] = elo_all['team'].map(CONF)
    elo_conf = elo_all.dropna(subset=['conf'])

    fig_box = go.Figure()
    for conf, color in CONF_COLORS.items():
        sub = elo_conf[elo_conf['conf'] == conf]['current_elo']
        if sub.empty:
            continue
        fig_box.add_trace(go.Violin(
            y=sub, name=conf,
            fillcolor=color, line_color=color,
            opacity=0.7,
            box_visible=True, meanline_visible=True,
            points='all',
            hovertemplate=f"<b>{conf}</b><br>Elo: %{{y:.0f}}<extra></extra>",
        ))
    fig_box.update_layout(
        **PLOT_LAYOUT, height=420,
        yaxis=dict(title='Elo Rating', gridcolor='#21262d'),
        legend=dict(bgcolor='#161b22', bordercolor='#30363d'),
        title=dict(text='Elo distribution across confederations (all teams)', font=dict(size=14)),
    )
    st.plotly_chart(fig_box, width="stretch")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — Historical Stats
# ══════════════════════════════════════════════════════════════════════════════
with tab5:
    st.markdown("### Historical International Football — Since 1872")

    wc = hist[hist['tournament'] == 'FIFA World Cup'].dropna(subset=['home_score'])

    # ── Row 1: goals distribution + home advantage ─────────────────────────
    c1, c2 = st.columns(2)

    with c1:
        st.markdown("#### Goals per match distribution (all matches)")
        hist_clean = hist.dropna(subset=['total_goals'])
        counts = hist_clean['total_goals'].value_counts().sort_index()
        counts = counts[counts.index <= 12]
        fig_hist = go.Figure(go.Bar(
            x=counts.index.astype(int),
            y=counts.values,
            marker_color='#58a6ff',
            marker_line_color='#21262d',
            marker_line_width=0.5,
            hovertemplate="Goals: %{x}<br>Matches: %{y:,}<extra></extra>",
        ))
        fig_hist.update_layout(
            **PLOT_LAYOUT, height=340,
            xaxis=dict(title='Total goals in match', dtick=1, gridcolor='#21262d'),
            yaxis=dict(title='Number of matches', gridcolor='#21262d'),
            title=dict(text='Goal distribution (all 49k+ international matches)', font=dict(size=12)),
        )
        st.plotly_chart(fig_hist, width="stretch")

    with c2:
        st.markdown("#### Home advantage (non-neutral venues)")
        non_neutral = hist[(hist['neutral'] == False) & hist['home_score'].notna()]
        hw = (non_neutral['home_score'] > non_neutral['away_score']).sum()
        d  = (non_neutral['home_score'] == non_neutral['away_score']).sum()
        aw = (non_neutral['home_score'] < non_neutral['away_score']).sum()
        fig_pie3 = go.Figure(go.Pie(
            labels=['Home win', 'Draw', 'Away win'],
            values=[hw, d, aw],
            marker_colors=['#3fb950','#ffa657','#f78166'],
            hole=0.45,
            textinfo='label+percent+value',
            textfont_size=11,
            hovertemplate="<b>%{label}</b><br>Count: %{value:,}<br>Share: %{percent}<extra></extra>",
        ))
        fig_pie3.update_layout(
            **PLOT_LAYOUT, height=340,
            legend=dict(bgcolor='#161b22', bordercolor='#30363d'),
            title=dict(text='Result distribution — home matches only', font=dict(size=12)),
        )
        st.plotly_chart(fig_pie3, width="stretch")

    # ── Row 2: avg goals per era ───────────────────────────────────────────
    st.markdown("#### Average goals per match by decade")
    hist_clean2 = hist.dropna(subset=['total_goals']).copy()
    hist_clean2['decade'] = (hist_clean2['year'] // 10) * 10
    by_decade = hist_clean2.groupby('decade').agg(
        avg_goals=('total_goals','mean'), n=('total_goals','count')
    ).reset_index()
    by_decade = by_decade[by_decade['decade'] >= 1870]

    fig_line = go.Figure()
    fig_line.add_trace(go.Scatter(
        x=by_decade['decade'], y=by_decade['avg_goals'],
        mode='lines+markers',
        line=dict(color='#58a6ff', width=2.5),
        marker=dict(size=7, color='#58a6ff'),
        fill='tozeroy', fillcolor='rgba(88,166,255,0.12)',
        hovertemplate="Decade: %{x}s<br>Avg goals: %{y:.2f}<extra></extra>",
        name='Avg goals/match',
    ))
    fig_line.update_layout(
        **PLOT_LAYOUT, height=350,
        xaxis=dict(title='Decade', dtick=10, gridcolor='#21262d'),
        yaxis=dict(title='Average goals per match', gridcolor='#21262d'),
        title=dict(text='Goals per match trend across 150 years', font=dict(size=13)),
    )
    st.plotly_chart(fig_line, width="stretch")

    # ── Row 3: WC goals per edition + top scorers (by team) ───────────────
    c3, c4 = st.columns(2)

    with c3:
        st.markdown("#### Total goals per World Cup edition")
        wc_by_year = wc.groupby('year').agg(
            total=('total_goals','sum'), matches=('total_goals','count')
        ).reset_index()
        wc_by_year['avg'] = wc_by_year['total'] / wc_by_year['matches']

        fig_wc = go.Figure()
        fig_wc.add_trace(go.Bar(
            x=wc_by_year['year'], y=wc_by_year['total'],
            marker_color='#ffa657',
            hovertemplate="WC %{x}<br>Total goals: %{y}<extra></extra>",
            name='Total goals',
        ))
        fig_wc.add_trace(go.Scatter(
            x=wc_by_year['year'], y=wc_by_year['avg'],
            mode='lines+markers',
            line=dict(color='#58a6ff', width=2),
            marker=dict(size=6),
            yaxis='y2',
            hovertemplate="WC %{x}<br>Avg/match: %{y:.2f}<extra></extra>",
            name='Avg/match',
        ))
        fig_wc.update_layout(
            **PLOT_LAYOUT, height=360,
            xaxis=dict(title='Year', dtick=4, gridcolor='#21262d'),
            yaxis=dict(title='Total goals', gridcolor='#21262d'),
            yaxis2=dict(title='Avg goals/match', overlaying='y', side='right',
                        gridcolor='#21262d', showgrid=False),
            legend=dict(bgcolor='#161b22', bordercolor='#30363d'),
            title=dict(text='World Cup goals — total & per-match average', font=dict(size=12)),
        )
        st.plotly_chart(fig_wc, width="stretch")

    with c4:
        st.markdown("#### Most matches played — all-time")
        all_apps = pd.concat([
            hist[['home_team']].rename(columns={'home_team':'team'}),
            hist[['away_team']].rename(columns={'away_team':'team'}),
        ])
        top_teams_count = all_apps['team'].value_counts().head(15).reset_index()
        top_teams_count.columns = ['team','matches']
        top_teams_count['color'] = top_teams_count['team'].map(
            lambda t: CONF_COLORS.get(CONF.get(t,'UEFA'),'#8b949e')
        )

        fig_teams = go.Figure(go.Bar(
            x=top_teams_count['matches'],
            y=top_teams_count['team'],
            orientation='h',
            marker_color=top_teams_count['color'].tolist(),
            marker_line_color='#21262d', marker_line_width=0.5,
            hovertemplate="<b>%{y}</b><br>Matches: %{x:,}<extra></extra>",
        ))
        fig_teams.update_layout(
            **PLOT_LAYOUT, height=360,
            yaxis=dict(autorange='reversed', gridcolor='#21262d'),
            xaxis=dict(title='Total international matches', gridcolor='#21262d'),
            title=dict(text='Most active international teams (all time)', font=dict(size=12)),
        )
        st.plotly_chart(fig_teams, width="stretch")

    # ── Row 4: matches per year trend ─────────────────────────────────────
    st.markdown("#### International football activity — matches per year")
    matches_per_year = hist.groupby('year').size().reset_index(name='matches')
    matches_per_year = matches_per_year[matches_per_year['year'] <= 2025]

    fig_activity = go.Figure(go.Scatter(
        x=matches_per_year['year'],
        y=matches_per_year['matches'],
        mode='lines',
        line=dict(color='#3fb950', width=1.5),
        fill='tozeroy', fillcolor='rgba(63,185,80,0.10)',
        hovertemplate="Year: %{x}<br>Matches: %{y:,}<extra></extra>",
    ))
    fig_activity.update_layout(
        **PLOT_LAYOUT, height=300,
        xaxis=dict(title='Year', gridcolor='#21262d'),
        yaxis=dict(title='Matches played', gridcolor='#21262d'),
        title=dict(text='International football matches per year — 1872 to 2025', font=dict(size=13)),
    )
    st.plotly_chart(fig_activity, width="stretch")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 6 — Bracket Simulator
# ══════════════════════════════════════════════════════════════════════════════
with tab6:
    st.markdown("### Interactive Bracket Simulator")
    st.markdown(
        "Select a round, set the bracket, and run the simulation. "
        "Use Auto-fill to seed teams by Elo rating, or load the actual WC 2026 bracket."
    )

    # ── Live load button MUST be before the selectbox so st.rerun() ──────────
    # applies before chosen_round is evaluated.
    _l, _mid, _r = st.columns([1, 2, 1])
    with _mid:
        if st.button("Load WC 2026 Round of 16", key="load_r16", width="stretch"):
            st.session_state["sim_round"] = "Round of 16  (8 matches)"
            st.session_state["bracket_r16"] = list(ACTUAL_R16_BRACKET)
            for i, (a, b) in enumerate(ACTUAL_R16_BRACKET):
                st.session_state[f"ma_r16_{i}"] = a
                st.session_state[f"mb_r16_{i}"] = b
            st.rerun()

    strength, WC_BASE, GLOBAL_ATK_AVG = build_model()

    ROUND_CONFIG = {
        "Round of 32  (16 matches)":   {"n_matches": 16, "n_teams": 32, "key": "r32"},
        "Round of 16  (8 matches)":    {"n_matches":  8, "n_teams": 16, "key": "r16"},
        "Quarter-finals  (4 matches)": {"n_matches":  4, "n_teams":  8, "key": "qf"},
        "Semi-finals  (2 matches)":    {"n_matches":  2, "n_teams":  4, "key": "sf"},
        "Final  (1 match)":            {"n_matches":  1, "n_teams":  2, "key": "final"},
    }

    chosen_round = st.selectbox(
        "Starting round", list(ROUND_CONFIG.keys()), key="sim_round"
    )
    cfg = ROUND_CONFIG[chosen_round]
    n_matches = cfg["n_matches"]
    n_teams   = cfg["n_teams"]

    top_by_elo = sorted(ALL_TEAMS, key=lambda t: -strength.loc[t, 'elo'])[:n_teams]

    col_ctrl1, col_ctrl2 = st.columns([1, 1])
    with col_ctrl1:
        n_sims = st.selectbox("Simulations", [200, 500, 1000, 2000], index=1, key="sim_n")
    with col_ctrl2:
        smart_fill = st.button("Auto-fill by Elo", key="smart_fill")

    # ── Forward-simulation fill (QF / SF) ─────────────────────────────────────
    _FWD_CONFIG = {
        "qf":    {"n_rounds": 1, "chain": "Round of 16",        "round_names": ["Round of 16"]},
        "sf":    {"n_rounds": 2, "chain": "R16 → QF",           "round_names": ["Round of 16", "Quarter-finals"]},
        "final": {"n_rounds": 3, "chain": "R16 → QF → SF",      "round_names": ["Round of 16", "Quarter-finals", "Semi-finals"]},
    }
    if cfg["key"] in _FWD_CONFIG:
        fwd = _FWD_CONFIG[cfg["key"]]
        with st.expander(f"🔗 Auto-fill teams by simulating {fwd['chain']} first (WC 2026 bracket)", expanded=True):
            st.caption(
                f"Runs a single simulated {fwd['chain']} starting from the actual WC 2026 R16 fixture list, "
                f"then fills this bracket with the simulated winners."
            )
            _fc1, _fc2 = st.columns([1, 3])
            with _fc1:
                if st.button(f"Simulate {fwd['chain']} — fill {cfg['key'].upper()}", key="fill_fwd"):
                    _summaries, _next = simulate_forward(
                        ACTUAL_R16_BRACKET, fwd["n_rounds"], strength, WC_BASE, GLOBAL_ATK_AVG
                    )
                    _next = list(_next)
                    st.session_state[f"bracket_{cfg['key']}"] = _next
                    st.session_state["_fwd_summaries"] = _summaries
                    st.session_state["_fwd_round_key"] = cfg["key"]
                    st.session_state["_fwd_round_names"] = fwd["round_names"]
                    # Write team names directly into selectbox keys so they
                    # take effect on rerun (pop alone is not sufficient).
                    for i, (a, b) in enumerate(_next):
                        st.session_state[f"ma_{cfg['key']}_{i}"] = a
                        st.session_state[f"mb_{cfg['key']}_{i}"] = b
                    st.rerun()

            # Show results of the most recent forward simulation for this round
            if (st.session_state.get("_fwd_round_key") == cfg["key"]
                    and "_fwd_summaries" in st.session_state):
                with _fc2:
                    _rnames = st.session_state.get("_fwd_round_names", [])
                    for rnd_idx, rnd_results in enumerate(st.session_state["_fwd_summaries"]):
                        st.markdown(f"**{_rnames[rnd_idx] if rnd_idx < len(_rnames) else 'Round'} results:**")
                        _rcols = st.columns(2)
                        for mi, (a, b, ga, gb, w, pa, pb) in enumerate(rnd_results):
                            with _rcols[mi % 2]:
                                won_a = w == a
                                pen_str = f" <span style='color:#ffa657'>({pa}–{pb} pens)</span>" if pa is not None else ""
                                st.markdown(
                                    f"<div style='background:#161b22;border:1px solid #30363d;"
                                    f"border-radius:6px;padding:6px 10px;margin:2px 0;font-size:12px'>"
                                    f"<span style='color:{'#3fb950' if won_a else '#8b949e'};font-weight:{'700' if won_a else '400'}'>{a}</span>"
                                    f" <span style='color:#58a6ff'>{ga}–{gb}</span>{pen_str} "
                                    f"<span style='color:{'#3fb950' if not won_a else '#8b949e'};font-weight:{'700' if not won_a else '400'}'>{b}</span>"
                                    f"</div>",
                                    unsafe_allow_html=True,
                                )

    st.markdown("---")
    st.markdown(f"#### Set the {n_matches} bracket match-ups ({n_teams} teams total)")
    st.caption("Each pair is one match. Winner advances to the next round.")

    bracket_key = f"bracket_{cfg['key']}"
    if bracket_key not in st.session_state or smart_fill:
        pairs = []
        for i in range(0, n_teams, 2):
            pairs.append((top_by_elo[i], top_by_elo[i + 1] if i + 1 < n_teams else top_by_elo[0]))
        st.session_state[bracket_key] = pairs
        # Write directly into selectbox keys so Streamlit picks up new values
        for i, (a, b) in enumerate(pairs):
            st.session_state[f"ma_{cfg['key']}_{i}"] = a
            st.session_state[f"mb_{cfg['key']}_{i}"] = b

    stored_pairs = st.session_state[bracket_key]

    # Render match-up selectors in a grid (2 matches per row)
    bracket = []
    for match_idx in range(n_matches):
        if match_idx % 2 == 0:
            cols = st.columns(2)
        with cols[match_idx % 2]:
            with st.container():
                st.markdown(f"**Match {match_idx + 1}**")
                default_a = stored_pairs[match_idx][0] if match_idx < len(stored_pairs) else top_by_elo[match_idx * 2]
                default_b = stored_pairs[match_idx][1] if match_idx < len(stored_pairs) else top_by_elo[match_idx * 2 + 1]
                c_a, c_vs, c_b = st.columns([5, 1, 5])
                with c_a:
                    team_a = st.selectbox(
                        "Team A", ALL_TEAMS,
                        index=ALL_TEAMS.index(default_a) if default_a in ALL_TEAMS else 0,
                        key=f"ma_{cfg['key']}_{match_idx}",
                        label_visibility="collapsed",
                    )
                with c_vs:
                    st.markdown("<div style='text-align:center;padding-top:6px;color:#8b949e'>vs</div>",
                                unsafe_allow_html=True)
                with c_b:
                    team_b = st.selectbox(
                        "Team B", ALL_TEAMS,
                        index=ALL_TEAMS.index(default_b) if default_b in ALL_TEAMS else 1,
                        key=f"mb_{cfg['key']}_{match_idx}",
                        label_visibility="collapsed",
                    )
                # Mini Elo badges
                elo_a = int(strength.loc[team_a, 'elo'])
                elo_b = int(strength.loc[team_b, 'elo'])
                st.markdown(
                    f"<small style='color:#58a6ff'>Elo {elo_a}</small>"
                    f"&nbsp;&nbsp;&nbsp;"
                    f"<small style='color:#58a6ff'>Elo {elo_b}</small>",
                    unsafe_allow_html=True,
                )
                bracket.append((team_a, team_b))

    # Save current state
    st.session_state[bracket_key] = bracket

    st.markdown("---")
    run_sim = st.button(f"Run {n_sims:,} Simulations", type="primary", key="run_sim")

    if run_sim:
        # Validate no duplicate teams
        all_selected = [t for pair in bracket for t in pair]
        if len(all_selected) != len(set(all_selected)):
            dupes = [t for t in set(all_selected) if all_selected.count(t) > 1]
            st.error(f"Duplicate teams in bracket: {', '.join(dupes)}. Each team must appear exactly once.")
        elif len(bracket) & (len(bracket) - 1) != 0:
            st.error("Number of matches must be a power of 2 (2, 4, 8, 16).")
        else:
            with st.spinner(f"Simulating {n_sims:,} tournaments…"):
                results_df = simulate_knockout_from_bracket(
                    bracket, strength, WC_BASE, GLOBAL_ATK_AVG, n_sims=n_sims
                )

            st.success(f"Done! {n_sims:,} simulations complete.")
            st.markdown("#### Simulation Results")

            # Merge with confederation colours
            results_df['conf'] = results_df['team'].map(CONF).fillna('UEFA')
            results_df['color'] = results_df['conf'].map(CONF_COLORS)

            # Win probability bar chart
            fig_sim = go.Figure()
            for conf, color in CONF_COLORS.items():
                sub = results_df[results_df['conf'] == conf]
                if sub.empty:
                    continue
                fig_sim.add_trace(go.Bar(
                    y=sub['team'],
                    x=sub['Win%'],
                    orientation='h',
                    name=conf,
                    marker_color=color,
                    marker_line_color='#21262d', marker_line_width=0.5,
                    text=[f"  {v:.1f}%" for v in sub['Win%']],
                    textposition='outside',
                    textfont=dict(size=10, color='#8b949e'),
                    hovertemplate="<b>%{y}</b><br>Win: %{x:.1f}%<extra></extra>",
                ))
            fig_sim.update_layout(
                **PLOT_LAYOUT,
                height=max(350, len(results_df) * 28),
                barmode='overlay',
                yaxis=dict(autorange='reversed', gridcolor='#21262d'),
                xaxis=dict(title='Win probability (%)', gridcolor='#21262d'),
                legend=dict(orientation='h', yanchor='bottom', y=1.01, xanchor='right', x=1,
                            bgcolor='#161b22', bordercolor='#30363d'),
                title=dict(text=f'Win probability from {chosen_round}', font=dict(size=13)),
            )
            st.plotly_chart(fig_sim, width="stretch")

            # Show first match predicted winners
            st.markdown("#### Predicted first-round winners (match-by-match)")
            match_cols = st.columns(min(4, n_matches))
            for i, (a, b) in enumerate(bracket):
                mu_a, mu_b = expected_goals(a, b, strength, WC_BASE, GLOBAL_ATK_AVG)
                p_a = mu_a / (mu_a + mu_b) * 100
                p_b = 100 - p_a
                fav = a if p_a >= p_b else b
                fav_pct = max(p_a, p_b)
                with match_cols[i % min(4, n_matches)]:
                    st.markdown(f"""
<div style="background:#161b22;border:1px solid #30363d;border-radius:8px;padding:10px;margin:4px 0;text-align:center">
  <div style="font-size:0.75rem;color:#8b949e">Match {i+1}</div>
  <div style="font-size:0.9rem;color:#e6edf3">{a} <span style="color:#30363d">vs</span> {b}</div>
  <div style="font-size:0.8rem;color:#3fb950;margin-top:4px">
    Favoured: <b>{fav}</b> ({fav_pct:.0f}%)
  </div>
  <div style="font-size:0.7rem;color:#8b949e">xG {mu_a:.2f} – {mu_b:.2f}</div>
</div>""", unsafe_allow_html=True)

            # Full results table
            st.markdown("#### Full simulation results table")
            display_cols = [c for c in results_df.columns if c.endswith('%')]
            table = results_df[['team'] + display_cols].copy()
            table.columns = ['Team'] + [c.replace('%','') for c in display_cols]
            st.dataframe(
                table.style.background_gradient(subset=table.columns[1:], cmap='YlOrRd')
                           .format({c: '{:.1f}%' for c in table.columns[1:]}),
                width="stretch",
            )

# ══════════════════════════════════════════════════════════════════════════════
# TAB 7 — Live Bracket
# ══════════════════════════════════════════════════════════════════════════════
with tab7:
    st.markdown("## WC 2026 — Live Tournament Tracker")
    st.caption("Data current as of 4 July 2026 · Round of 16 fixtures set · QF draw pending")

    STATUS_COLOR = {'W': '#10b981', 'R': '#6366f1', '3': '#f59e0b', '': '#333350'}
    STATUS_LABEL = {'W': 'Winner', 'R': 'Runner-up', '3': 'Best 3rd', '': 'Eliminated'}

    # ── Group stage tables ─────────────────────────────────────────────────
    st.markdown("### Group Stage — Final Standings")
    grp_cols_per_row = 3
    grp_keys = list(ACTUAL_GROUP_STANDINGS.keys())

    for row_start in range(0, 12, grp_cols_per_row):
        cols = st.columns(grp_cols_per_row)
        for ci, g in enumerate(grp_keys[row_start:row_start + grp_cols_per_row]):
            with cols[ci]:
                st.markdown(f"**Group {g}**")
                rows_html = ""
                for t in ACTUAL_GROUP_STANDINGS[g]:
                    q = t['q']
                    bg = '#1a2f1a' if q == 'W' else ('#1a2040' if q == 'R' else ('#2a2010' if q == '3' else '#1c1c1c'))
                    badge_color = STATUS_COLOR[q]
                    badge = f"<span style='background:{badge_color};color:#0d1117;font-size:9px;padding:1px 5px;border-radius:3px;font-weight:700'>{STATUS_LABEL[q]}</span>" if q else "<span style='color:#484f58;font-size:9px'>Out</span>"
                    rows_html += f"""
<tr style="background:{bg}">
  <td style="padding:4px 6px;font-size:12px;color:#e6edf3">{t['display']}</td>
  <td style="padding:4px 6px;font-size:12px;color:#8b949e;text-align:center">{t['pts']}</td>
  <td style="padding:4px 6px;font-size:12px;color:#8b949e;text-align:center">{t['gd']:+d}</td>
  <td style="padding:4px 6px;font-size:11px;text-align:right">{badge}</td>
</tr>"""
                st.markdown(f"""
<table style="width:100%;border-collapse:collapse;border:1px solid #30363d;border-radius:6px;overflow:hidden;margin-bottom:8px">
  <tr style="background:#21262d">
    <th style="padding:4px 6px;font-size:11px;color:#8b949e;text-align:left">Team</th>
    <th style="padding:4px 6px;font-size:11px;color:#8b949e;text-align:center">Pts</th>
    <th style="padding:4px 6px;font-size:11px;color:#8b949e;text-align:center">GD</th>
    <th style="padding:4px 6px;font-size:11px;color:#8b949e"></th>
  </tr>{rows_html}
</table>""", unsafe_allow_html=True)

    # Best 3rd-place teams banner
    st.markdown("### Best 8 Third-Place Teams — All Qualified")
    thirds_html = "".join([
        f"<span style='background:#2a2010;border:1px solid #ffa657;border-radius:6px;padding:5px 10px;margin:4px;display:inline-block;color:#ffa657;font-size:13px'>{t}</span>"
        for t in BEST_THIRDS
    ])
    st.markdown(f"<div style='line-height:2.5'>{thirds_html}</div>", unsafe_allow_html=True)

    st.divider()

    # ── R32 results ────────────────────────────────────────────────────────
    st.markdown("### Round of 32 — Results")
    r32_cols = st.columns(2)
    for i, m in enumerate(ACTUAL_R32):
        pen_note = f"  *(pens {m['pen']})*" if m['pen'] else ""
        is_pen = m['pen'] is not None
        home_won = m['winner'] == m['home']
        away_won = m['winner'] == m['away']
        home_col = '#3fb950' if home_won else '#e6edf3'
        away_col = '#3fb950' if away_won else '#e6edf3'
        score_note = f"({m['hs']}–{m['as']}{'  pens' if is_pen else ''})"
        with r32_cols[i % 2]:
            st.markdown(f"""
<div style="background:#161b22;border:1px solid #30363d;border-radius:8px;padding:10px 14px;margin:4px 0">
  <div style="display:flex;justify-content:space-between;align-items:center">
    <span style="font-size:13px;font-weight:{'700' if home_won else '400'};color:{home_col}">{m['home']}</span>
    <span style="font-size:11px;color:#8b949e">{score_note}</span>
    <span style="font-size:13px;font-weight:{'700' if away_won else '400'};color:{away_col}">{m['away']}</span>
  </div>
  <div style="text-align:center;margin-top:2px">
    <span style="font-size:10px;color:#3fb950">→ {m['winner']} advance</span>
  </div>
</div>""", unsafe_allow_html=True)

    st.divider()

    # ── R16 fixtures ───────────────────────────────────────────────────────
    st.markdown("### Round of 16 — Upcoming Fixtures")
    st.caption("All matches scheduled for 5–8 July 2026")

    r16_cols = st.columns(2)
    # Use pre-computed win probabilities from predictions CSV for each matchup
    for i, (a, b) in enumerate(ACTUAL_R16_BRACKET):
        win_a = pred.loc[pred['team'] == a, 'Win%'].values
        win_b = pred.loc[pred['team'] == b, 'Win%'].values
        pct_a = float(win_a[0]) if len(win_a) else 0.0
        pct_b = float(win_b[0]) if len(win_b) else 0.0
        total  = pct_a + pct_b if (pct_a + pct_b) > 0 else 1
        bar_a  = int(pct_a / total * 100)
        bar_b  = 100 - bar_a
        fav    = a if pct_a >= pct_b else b
        fav_pct= max(pct_a, pct_b) / total * 100
        conf_a = CONF_COLORS.get(CONF.get(a,'UEFA'), '#8b949e')
        conf_b = CONF_COLORS.get(CONF.get(b,'UEFA'), '#8b949e')
        with r16_cols[i % 2]:
            st.markdown(f"""
<div style="background:#161b22;border:1px solid #30363d;border-radius:10px;padding:14px 16px;margin:6px 0">
  <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px">
    <span style="font-size:14px;font-weight:600;color:#e6edf3">{a}</span>
    <span style="font-size:11px;color:#8b949e;background:#21262d;padding:2px 8px;border-radius:4px">R16</span>
    <span style="font-size:14px;font-weight:600;color:#e6edf3">{b}</span>
  </div>
  <div style="display:flex;gap:2px;border-radius:4px;overflow:hidden;height:8px">
    <div style="width:{bar_a}%;background:{conf_a}"></div>
    <div style="width:{bar_b}%;background:{conf_b}"></div>
  </div>
  <div style="display:flex;justify-content:space-between;margin-top:4px">
    <span style="font-size:10px;color:#8b949e">{pct_a:.1f}% win odds</span>
    <span style="font-size:10px;color:#ffa657">Fav: {fav} ({fav_pct:.0f}%)</span>
    <span style="font-size:10px;color:#8b949e">{pct_b:.1f}% win odds</span>
  </div>
</div>""", unsafe_allow_html=True)

    # ── R16 win probability bar chart ──────────────────────────────────────
    st.markdown("### R16 Teams — Championship Odds (pre-tournament model)")
    r16_teams = list({t for pair in ACTUAL_R16_BRACKET for t in pair})
    r16_pred = pred[pred['team'].isin(r16_teams)].sort_values('Win%', ascending=False).copy()
    r16_pred['conf'] = r16_pred['team'].map(CONF).fillna('UEFA')

    fig_r16 = go.Figure()
    for conf, color in CONF_COLORS.items():
        sub = r16_pred[r16_pred['conf'] == conf]
        if sub.empty:
            continue
        fig_r16.add_trace(go.Bar(
            y=sub['team'], x=sub['Win%'], orientation='h', name=conf,
            marker_color=color, marker_line_color='#21262d', marker_line_width=0.5,
            text=[f"  {v:.1f}%" for v in sub['Win%']],
            textposition='outside', textfont=dict(size=10, color='#8b949e'),
            hovertemplate="<b>%{y}</b><br>Win%: %{x:.1f}%<extra></extra>",
        ))
    fig_r16.update_layout(
        **PLOT_LAYOUT, height=420, barmode='overlay',
        yaxis=dict(autorange='reversed', **AXIS_STYLE),
        xaxis=dict(title='Pre-tournament championship probability (%)', **AXIS_STYLE),
        legend=dict(orientation='h', yanchor='bottom', y=1.01, xanchor='right', x=1,
                    bgcolor='#161b22', bordercolor='#30363d'),
        title=dict(text='Round of 16 teams — pre-tournament model win odds', font=dict(size=13)),
    )
    st.plotly_chart(fig_r16, width="stretch")

    st.info("Open the Simulate tab and click **Load WC 2026 Round of 16** to run Monte Carlo simulations from the current bracket.")

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.markdown("""
<div style="text-align:center; color:#484f58; font-size:0.75rem; padding:10px">
  Model: Dixon-Coles Poisson · Blended Elo + Recent Attack/Defence · Monte Carlo simulation<br>
  Data: International results 1872–2026 · Kaggle International Football Results
</div>
""", unsafe_allow_html=True)
