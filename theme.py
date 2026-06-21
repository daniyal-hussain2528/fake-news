"""
theme.py
--------
Visual theme for the Fake News Identification System.

Design language: "Newsroom Verification Desk" — a dark editorial backdrop
(ink navy) with two competing signal colors that map directly to the
classification task: ALERT RED for Fake, VERIFIED GREEN/CYAN for Real,
plus AMBER for neutral data and VIOLET for the ML/tech layer.

Everything is delivered as a single CSS injection so app.py stays simple.
Cards use layered box-shadows (not just blur) to read as physically
raised/pressed — that's the "3D" component — and the prediction badge
uses a CSS keyframe stamp animation as the page's signature element.
"""

import streamlit as st

# ---- Token system -------------------------------------------------
COLORS = {
    "bg_void": "#070912",
    "bg_panel": "#10131F",
    "bg_panel_raised": "#161A2C",
    "ink_line": "#262B40",
    "text_primary": "#ECEFF9",
    "text_dim": "#8B91AC",
    "alert_red": "#FF3B5C",
    "alert_red_glow": "rgba(255, 59, 92, 0.45)",
    "verified_green": "#00E5A0",
    "verified_glow": "rgba(0, 229, 160, 0.45)",
    "amber": "#FFB627",
    "amber_glow": "rgba(255, 182, 39, 0.35)",
    "violet": "#8B5CF6",
    "violet_glow": "rgba(139, 92, 246, 0.40)",
}


def inject_theme():
    st.markdown(
        f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Archivo+Black&family=Archivo:wght@400;600;800&family=JetBrains+Mono:wght@400;600;700&display=swap');

    :root {{
        --bg-void: {COLORS['bg_void']};
        --bg-panel: {COLORS['bg_panel']};
        --bg-panel-raised: {COLORS['bg_panel_raised']};
        --ink-line: {COLORS['ink_line']};
        --text-primary: {COLORS['text_primary']};
        --text-dim: {COLORS['text_dim']};
        --alert: {COLORS['alert_red']};
        --alert-glow: {COLORS['alert_red_glow']};
        --verified: {COLORS['verified_green']};
        --verified-glow: {COLORS['verified_glow']};
        --amber: {COLORS['amber']};
        --amber-glow: {COLORS['amber_glow']};
        --violet: {COLORS['violet']};
        --violet-glow: {COLORS['violet_glow']};
    }}

    /* ---------- Base canvas ---------- */
    .stApp {{
        background:
            radial-gradient(circle at 15% 0%, rgba(139,92,246,0.12), transparent 45%),
            radial-gradient(circle at 85% 15%, rgba(0,229,160,0.10), transparent 40%),
            var(--bg-void);
        color: var(--text-primary);
        font-family: 'Archivo', sans-serif;
    }}

    /* faint moving scanline texture across the whole app, newsroom-monitor feel */
    .stApp::before {{
        content: "";
        position: fixed;
        inset: 0;
        pointer-events: none;
        background: repeating-linear-gradient(
            to bottom,
            rgba(255,255,255,0.012) 0px,
            rgba(255,255,255,0.012) 1px,
            transparent 1px,
            transparent 3px
        );
        z-index: 0;
        animation: scan-drift 9s linear infinite;
    }}
    @keyframes scan-drift {{
        0% {{ background-position: 0 0; }}
        100% {{ background-position: 0 120px; }}
    }}

    /* ---------- Headline / display type ---------- */
    h1 {{
        font-family: 'Archivo Black', sans-serif !important;
        letter-spacing: -0.01em;
        text-transform: uppercase;
        font-size: 2.6rem !important;
        background: linear-gradient(100deg, var(--text-primary) 40%, var(--violet) 75%, var(--verified) 100%);
        -webkit-background-clip: text;
        background-clip: text;
        -webkit-text-fill-color: transparent;
        text-shadow: 0 0 40px rgba(139,92,246,0.25);
        margin-bottom: 0.1rem !important;
    }}

    h2, h3 {{
        font-family: 'Archivo', sans-serif !important;
        font-weight: 800 !important;
        color: var(--text-primary) !important;
        letter-spacing: -0.005em;
    }}

    h2::before {{
        content: "// ";
        color: var(--violet);
        font-family: 'JetBrains Mono', monospace;
    }}

    p, label, .stMarkdown {{
        color: var(--text-dim);
    }}

    /* ---------- Top nav buttons: ticker-tape pill bar ---------- */
    div[data-testid="column"] .stButton > button {{
        background: linear-gradient(180deg, var(--bg-panel-raised), var(--bg-panel));
        color: var(--text-primary);
        border: 1px solid var(--ink-line);
        border-radius: 10px;
        padding: 0.7rem 0.4rem;
        font-family: 'JetBrains Mono', monospace;
        font-weight: 700;
        letter-spacing: 0.02em;
        text-transform: uppercase;
        font-size: 0.8rem;
        box-shadow:
            0 1px 0 rgba(255,255,255,0.04) inset,
            0 8px 18px -8px rgba(0,0,0,0.7),
            0 0 0 0 rgba(139,92,246,0);
        transition: transform 0.15s ease, box-shadow 0.25s ease, border-color 0.25s ease;
    }}
    div[data-testid="column"] .stButton > button:hover {{
        transform: translateY(-3px);
        border-color: var(--violet);
        box-shadow:
            0 1px 0 rgba(255,255,255,0.06) inset,
            0 14px 24px -10px rgba(0,0,0,0.8),
            0 0 26px var(--violet-glow);
        color: #fff;
    }}
    div[data-testid="column"] .stButton > button:active {{
        transform: translateY(0px) scale(0.98);
    }}

    /* ---------- Generic 3D panel for grouped content ---------- */
    .panel-3d {{
        background: linear-gradient(160deg, var(--bg-panel-raised) 0%, var(--bg-panel) 100%);
        border: 1px solid var(--ink-line);
        border-radius: 16px;
        padding: 1.4rem 1.6rem;
        box-shadow:
            0 1px 0 rgba(255,255,255,0.05) inset,
            0 18px 40px -20px rgba(0,0,0,0.85),
            0 0 0 1px rgba(255,255,255,0.02);
        margin-bottom: 1.1rem;
        position: relative;
        overflow: hidden;
    }}
    .panel-3d::after {{
        content: "";
        position: absolute;
        top: -60%;
        left: -20%;
        width: 60%;
        height: 220%;
        background: linear-gradient(75deg, transparent, rgba(255,255,255,0.035), transparent);
        transform: rotate(8deg);
        pointer-events: none;
    }}

    .eyebrow {{
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.72rem;
        letter-spacing: 0.14em;
        text-transform: uppercase;
        color: var(--violet);
        font-weight: 700;
        margin-bottom: 0.3rem;
        display: block;
    }}

    /* ---------- Prediction badge: the signature stamp element ---------- */
    .verdict-stage {{
        display: flex;
        justify-content: center;
        padding: 1.8rem 0 0.6rem 0;
    }}
    .verdict-badge {{
        font-family: 'Archivo Black', sans-serif;
        font-size: 2.1rem;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        padding: 1.1rem 2.6rem;
        border-radius: 14px;
        position: relative;
        transform: rotate(-3deg) scale(1);
        animation: stamp-in 0.45s cubic-bezier(.2,1.4,.4,1) both;
        border: 3px solid currentColor;
    }}
    .verdict-badge.real {{
        color: var(--verified);
        background: rgba(0,229,160,0.08);
        box-shadow:
            0 0 0 6px rgba(0,229,160,0.06),
            0 18px 50px -12px var(--verified-glow),
            0 1px 0 rgba(255,255,255,0.15) inset;
        text-shadow: 0 0 24px var(--verified-glow);
    }}
    .verdict-badge.fake {{
        color: var(--alert);
        background: rgba(255,59,92,0.08);
        box-shadow:
            0 0 0 6px rgba(255,59,92,0.06),
            0 18px 50px -12px var(--alert-glow),
            0 1px 0 rgba(255,255,255,0.15) inset;
        text-shadow: 0 0 24px var(--alert-glow);
    }}
    @keyframes stamp-in {{
        0%   {{ transform: rotate(-3deg) scale(2.6); opacity: 0; filter: blur(6px); }}
        60%  {{ transform: rotate(-3deg) scale(0.92); opacity: 1; filter: blur(0); }}
        100% {{ transform: rotate(-3deg) scale(1); }}
    }}

    /* ---------- Metric cards (st.metric) ---------- */
    div[data-testid="stMetric"] {{
        background: linear-gradient(160deg, var(--bg-panel-raised), var(--bg-panel));
        border: 1px solid var(--ink-line);
        border-radius: 14px;
        padding: 0.9rem 1rem 0.7rem 1rem;
        box-shadow:
            0 1px 0 rgba(255,255,255,0.05) inset,
            0 14px 28px -16px rgba(0,0,0,0.8);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }}
    div[data-testid="stMetric"]:hover {{
        transform: translateY(-4px);
        border-color: var(--amber);
    }}
    div[data-testid="stMetricLabel"] {{
        font-family: 'JetBrains Mono', monospace;
        color: var(--text-dim) !important;
        text-transform: uppercase;
        font-size: 0.72rem !important;
        letter-spacing: 0.08em;
    }}
    div[data-testid="stMetricValue"] {{
        font-family: 'Archivo Black', sans-serif;
        color: var(--amber) !important;
        text-shadow: 0 0 20px var(--amber-glow);
    }}

    /* ---------- DataFrames / tables ---------- */
    div[data-testid="stDataFrame"] {{
        border-radius: 12px;
        overflow: hidden;
        border: 1px solid var(--ink-line);
        box-shadow: 0 16px 32px -18px rgba(0,0,0,0.8);
    }}

    /* ---------- Text input / textarea ---------- */
    .stTextArea textarea, .stSelectbox div[data-baseweb="select"] > div {{
        background: var(--bg-panel) !important;
        border: 1px solid var(--ink-line) !important;
        color: var(--text-primary) !important;
        border-radius: 10px !important;
        font-family: 'JetBrains Mono', monospace !important;
    }}
    .stTextArea textarea:focus {{
        border-color: var(--violet) !important;
        box-shadow: 0 0 0 3px var(--violet-glow) !important;
    }}

    /* ---------- Primary action button (Predict / Run pipeline) ---------- */
    .stButton > button[kind="primary"], .stButton > button:has(span:contains("Predict")) {{
        background: linear-gradient(135deg, var(--violet), #5B3FD6) !important;
    }}

    /* generic primary-style override applied via container class below */
    .pulse-cta .stButton > button {{
        background: linear-gradient(135deg, var(--violet) 0%, #5B3FD6 100%) !important;
        border: none !important;
        color: #fff !important;
        font-family: 'Archivo', sans-serif !important;
        font-weight: 800 !important;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        padding: 0.85rem !important;
        box-shadow:
            0 1px 0 rgba(255,255,255,0.2) inset,
            0 16px 34px -14px var(--violet-glow) !important;
    }}
    .pulse-cta .stButton > button:hover {{
        transform: translateY(-3px) scale(1.01);
        box-shadow:
            0 1px 0 rgba(255,255,255,0.25) inset,
            0 20px 42px -14px var(--violet-glow) !important;
    }}

    /* ---------- Expanders (ML Pipeline steps) ---------- */
    .streamlit-expanderHeader {{
        background: var(--bg-panel-raised) !important;
        border: 1px solid var(--ink-line) !important;
        border-radius: 10px !important;
        font-family: 'JetBrains Mono', monospace !important;
        color: var(--text-primary) !important;
    }}

    /* ---------- File uploader ---------- */
    section[data-testid="stFileUploaderDropzone"] {{
        background: var(--bg-panel) !important;
        border: 2px dashed var(--ink-line) !important;
        border-radius: 14px !important;
    }}
    section[data-testid="stFileUploaderDropzone"]:hover {{
        border-color: var(--verified) !important;
    }}

    /* ---------- Misc cleanup ---------- */
    hr {{ border-color: var(--ink-line) !important; }}
    code {{ color: var(--verified) !important; background: rgba(0,229,160,0.08) !important; }}
    </style>
    """,
        unsafe_allow_html=True,
    )


def render_verdict_badge(prediction: str):
    """Render the signature 3D 'press stamp' verdict badge."""
    css_class = "real" if prediction == "Real" else "fake"
    icon = "✅" if prediction == "Real" else "🚫"
    st.markdown(
        f"""
        <div class="verdict-stage">
            <div class="verdict-badge {css_class}">{icon} {prediction.upper()}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def panel_start(eyebrow: str | None = None):
    """Open a 3D panel div. Must be paired with panel_end()."""
    eyebrow_html = f'<span class="eyebrow">{eyebrow}</span>' if eyebrow else ""
    st.markdown(f'<div class="panel-3d">{eyebrow_html}', unsafe_allow_html=True)


def panel_end():
    st.markdown("</div>", unsafe_allow_html=True)
