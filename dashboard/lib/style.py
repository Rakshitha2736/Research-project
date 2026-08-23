"""Shared Streamlit chrome / CSS for the dashboard (premium analytics theme)."""

from __future__ import annotations

import streamlit as st

from .paths import (
    FORECAST_HORIZONS,
    N_PRIMARY_MODELS,
    N_USABLE_STATIONS,
    TEST_PERIOD_END,
    TEST_PERIOD_START,
)

DISCLAIMER_TEXT = (
    "This dashboard replays and explores historical forecasts from the "
    f"locked test period ({TEST_PERIOD_START} to {TEST_PERIOD_END}). It does NOT "
    "generate live/current forecasts — the underlying model requires 30 days of "
    "complete observed weather data, including rainfall, which is not "
    "available in real-time in this deployment."
)

# Consistent source labels (Fix 2)
LABEL_SEED42 = "Seed-42 (replay prediction)"
LABEL_3SEED = "3-seed mean ± std (research result)"
DUAL_NUMBER_NOTE = (
    "This page shows two distinct types of numbers: individual seed-42 "
    "predictions used for interactive replay, and aggregated 3-seed research "
    "statistics used for the project's formal conclusions. These serve "
    "different purposes and are not directly interchangeable."
)

# Design system — midnight navy analytics (visual language only)
DS = {
    "bg": "#0b1220",
    "bg_elevated": "#0f172a",
    "sidebar": "#080e1a",
    "card": "#121a2b",
    "card_alt": "#152033",
    "card_border": "#1e2a3f",
    "card_border_strong": "#2a3a55",
    "ink": "#e8eef7",
    "muted": "#8b9bb4",
    "faint": "#5c6b82",
    "accent": "#3b82f6",
    "accent_soft": "#1e3a5f",
    "accent_glow": "rgba(59, 130, 246, 0.18)",
    "success": "#22c55e",
    "success_soft": "#14532d",
    "warn": "#f59e0b",
    "warn_bg": "#1f1808",
    "warn_border": "#d97706",
    "warn_text": "#fde68a",
    "danger": "#ef4444",
    "purple": "#a78bfa",
    "teal": "#14b8a6",
    "orange": "#f97316",
    "radius": "12px",
    "radius_sm": "8px",
    "pad": "1rem 1.15rem",
    "space_1": "0.4rem",
    "space_2": "0.85rem",
    "space_3": "1.35rem",
    "font_header": "1.75rem",
    "font_subheader": "1.05rem",
    "font_body": "0.95rem",
    "font_caption": "0.8rem",
}

_SS_CSS_RUN = "_dash_css_run_id"
_SS_BADGE_RUN = "_dash_badge_run_id"
_SS_SIDEBAR_RUN = "_dash_sidebar_run_id"


def _script_run_id() -> str:
    """Stable id for the current Streamlit script run (for idempotent chrome)."""
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx

        ctx = get_script_run_ctx()
        if ctx is not None and getattr(ctx, "id", None):
            return str(ctx.id)
    except Exception:
        pass
    return "no-ctx"


def inject_base_css() -> None:
    """Inject design-system CSS at most once per script run."""
    run_id = _script_run_id()
    if st.session_state.get(_SS_CSS_RUN) == run_id:
        return

    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,400;0,9..40,500;0,9..40,600;0,9..40,700;1,9..40,400&family=JetBrains+Mono:wght@500;600&display=swap');

        :root {{
          --bg: {DS["bg"]};
          --bg-elevated: {DS["bg_elevated"]};
          --sidebar: {DS["sidebar"]};
          --card: {DS["card"]};
          --card-alt: {DS["card_alt"]};
          --card-border: {DS["card_border"]};
          --card-border-strong: {DS["card_border_strong"]};
          --ink: {DS["ink"]};
          --muted: {DS["muted"]};
          --faint: {DS["faint"]};
          --accent: {DS["accent"]};
          --accent-soft: {DS["accent_soft"]};
          --accent-glow: {DS["accent_glow"]};
          --success: {DS["success"]};
          --success-soft: {DS["success_soft"]};
          --warn: {DS["warn"]};
          --warn-bg: {DS["warn_bg"]};
          --warn-border: {DS["warn_border"]};
          --warn-text: {DS["warn_text"]};
          --danger: {DS["danger"]};
          --purple: {DS["purple"]};
          --teal: {DS["teal"]};
          --orange: {DS["orange"]};
          --radius: {DS["radius"]};
          --radius-sm: {DS["radius_sm"]};
          --pad: {DS["pad"]};
          --space-1: {DS["space_1"]};
          --space-2: {DS["space_2"]};
          --space-3: {DS["space_3"]};
          --font-header: {DS["font_header"]};
          --font-subheader: {DS["font_subheader"]};
          --font-body: {DS["font_body"]};
          --font-caption: {DS["font_caption"]};
        }}

        html, body, [class*="css"] {{
          font-family: "DM Sans", "Segoe UI", sans-serif;
        }}

        .stApp {{
          background:
            radial-gradient(1200px 600px at 12% -10%, rgba(59,130,246,0.10), transparent 55%),
            radial-gradient(900px 500px at 90% 0%, rgba(20,184,166,0.06), transparent 50%),
            var(--bg);
          color: var(--ink);
        }}

        .block-container {{
          padding-top: 1rem !important;
          padding-bottom: 2rem !important;
          padding-left: 1.5rem !important;
          padding-right: 1.5rem !important;
          max-width: 1480px;
        }}

        /* Hide default Streamlit chrome clutter */
        #MainMenu {{ visibility: hidden; }}
        header[data-testid="stHeader"] {{
          background: transparent;
        }}
        div[data-testid="stToolbar"] {{
          display: none;
        }}
        footer {{ visibility: hidden; }}

        h1, h2, h3, h4 {{
          color: var(--ink) !important;
          letter-spacing: -0.025em;
          font-weight: 650 !important;
        }}
        h1 {{ font-size: var(--font-header) !important; margin-bottom: 0.15rem !important; }}
        h2 {{ font-size: 1.2rem !important; }}
        h3, h4 {{ font-size: var(--font-subheader) !important; }}
        p, li {{ font-size: var(--font-body); color: var(--ink); }}

        /* ========== SIDEBAR ========== */
        section[data-testid="stSidebar"] {{
          background: linear-gradient(180deg, #0a101c 0%, #0b1324 100%) !important;
          border-right: 1px solid var(--card-border) !important;
        }}
        section[data-testid="stSidebar"] > div {{
          background: transparent !important;
        }}
        section[data-testid="stSidebar"] [data-testid="stSidebarNav"] {{
          padding-top: 0.25rem;
        }}
        /* Group labels: OVERVIEW / ANALYSIS */
        section[data-testid="stSidebar"] [data-testid="stSidebarNav"] span[data-testid="stMarkdownContainer"] p,
        section[data-testid="stSidebar"] [data-testid="stSidebarNav"] li[data-testid="stSidebarNavSeparator"] {{
          color: var(--faint) !important;
          font-size: 0.68rem !important;
          font-weight: 700 !important;
          letter-spacing: 0.12em !important;
          text-transform: uppercase !important;
        }}
        /* Nav links */
        section[data-testid="stSidebar"] [data-testid="stSidebarNav"] a {{
          border-radius: 10px !important;
          margin: 0.18rem 0.35rem !important;
          padding: 0.55rem 0.75rem !important;
          border: 1px solid transparent !important;
          transition: background 0.15s ease, border-color 0.15s ease, box-shadow 0.15s ease;
          color: var(--muted) !important;
          font-weight: 500 !important;
        }}
        section[data-testid="stSidebar"] [data-testid="stSidebarNav"] a:hover {{
          background: rgba(59,130,246,0.08) !important;
          border-color: rgba(59,130,246,0.2) !important;
          color: var(--ink) !important;
        }}
        /* Active page */
        section[data-testid="stSidebar"] [data-testid="stSidebarNav"] a[aria-selected="true"],
        section[data-testid="stSidebar"] [data-testid="stSidebarNavLink"][aria-selected="true"],
        section[data-testid="stSidebar"] a[aria-current="page"] {{
          background: linear-gradient(90deg, rgba(59,130,246,0.22), rgba(59,130,246,0.08)) !important;
          border: 1px solid rgba(59,130,246,0.35) !important;
          box-shadow: inset 3px 0 0 var(--accent);
          color: var(--ink) !important;
          font-weight: 650 !important;
        }}
        section[data-testid="stSidebar"] [data-testid="stSidebarNav"] svg {{
          color: inherit !important;
        }}

        .sidebar-brand {{
          display: flex;
          align-items: center;
          gap: 0.75rem;
          padding: 0.35rem 0.4rem 1rem 0.4rem;
          margin-bottom: 0.35rem;
          border-bottom: 1px solid var(--card-border);
        }}
        .sidebar-brand .mark {{
          width: 40px;
          height: 40px;
          border-radius: 11px;
          background: linear-gradient(145deg, #2563eb, #0ea5e9);
          display: flex;
          align-items: center;
          justify-content: center;
          box-shadow: 0 4px 14px rgba(37, 99, 235, 0.35);
          flex-shrink: 0;
        }}
        .sidebar-brand .mark {{
          overflow: hidden;
        }}
        .sidebar-brand .mark svg {{
          width: 22px !important;
          height: 22px !important;
          max-width: 22px !important;
          max-height: 22px !important;
          display: block;
        }}
        .sidebar-brand .titles {{
          min-width: 0;
        }}
        .sidebar-brand .titles .name {{
          font-size: 0.98rem;
          font-weight: 700;
          color: var(--ink);
          line-height: 1.2;
          letter-spacing: -0.02em;
        }}
        .sidebar-brand .titles .sub {{
          font-size: 0.72rem;
          color: var(--muted);
          margin-top: 0.12rem;
          line-height: 1.3;
        }}
        .sidebar-disclaimer {{
          margin: 0.85rem 0.35rem 0.5rem 0.35rem;
          padding: 0.7rem 0.8rem;
          background: var(--warn-bg);
          border: 1px solid var(--warn-border);
          border-radius: var(--radius-sm);
          color: var(--warn-text);
          font-size: 0.72rem;
          line-height: 1.45;
        }}
        .sidebar-disclaimer .lbl {{
          display: block;
          font-size: 0.65rem;
          letter-spacing: 0.1em;
          text-transform: uppercase;
          color: var(--warn);
          font-weight: 700;
          margin-bottom: 0.3rem;
        }}

        /* ========== PAGE HEADER ========== */
        .page-hero {{
          display: flex;
          flex-wrap: wrap;
          justify-content: space-between;
          align-items: flex-start;
          gap: 1rem;
          margin: 0 0 1rem 0;
          padding-bottom: 0.85rem;
          border-bottom: 1px solid var(--card-border);
        }}
        .page-hero .hero-text {{
          flex: 1 1 280px;
          min-width: 0;
        }}
        .page-hero .eyebrow {{
          font-size: 0.72rem;
          font-weight: 700;
          letter-spacing: 0.1em;
          text-transform: uppercase;
          color: var(--accent);
          margin-bottom: 0.35rem;
        }}
        .page-hero h1 {{
          margin: 0 !important;
          font-size: 1.65rem !important;
          line-height: 1.2 !important;
        }}
        .page-hero .subtitle {{
          margin-top: 0.35rem;
          color: var(--muted);
          font-size: 0.92rem;
          line-height: 1.45;
          max-width: 42rem;
        }}
        .status-chip-row {{
          display: flex;
          flex-wrap: wrap;
          gap: 0.5rem;
          justify-content: flex-end;
          align-items: stretch;
        }}
        .status-chip {{
          background: var(--card);
          border: 1px solid var(--card-border);
          border-radius: var(--radius-sm);
          padding: 0.45rem 0.7rem;
          min-width: 7.5rem;
          transition: border-color 0.15s ease;
        }}
        .status-chip:hover {{
          border-color: var(--accent);
        }}
        .status-chip .k {{
          display: block;
          font-size: 0.65rem;
          text-transform: uppercase;
          letter-spacing: 0.08em;
          color: var(--muted);
          font-weight: 650;
          margin-bottom: 0.15rem;
        }}
        .status-chip .v {{
          display: block;
          font-size: 0.82rem;
          font-weight: 650;
          color: var(--ink);
          font-family: "JetBrains Mono", "DM Sans", monospace;
        }}

        /* ========== KPI / STAT CARDS ========== */
        .kpi-row {{
          display: grid;
          grid-template-columns: repeat(5, 1fr);
          gap: 0.75rem;
          margin: 0.25rem 0 1rem 0;
        }}
        @media (max-width: 1200px) {{
          .kpi-row {{ grid-template-columns: repeat(3, 1fr); }}
        }}
        @media (max-width: 800px) {{
          .kpi-row {{ grid-template-columns: repeat(2, 1fr); }}
        }}
        .kpi-card {{
          background: linear-gradient(180deg, var(--card-alt) 0%, var(--card) 100%);
          border: 1px solid var(--card-border);
          border-radius: var(--radius);
          padding: 0.95rem 1rem;
          position: relative;
          overflow: hidden;
          min-height: 6.4rem;
          transition: border-color 0.15s ease, box-shadow 0.15s ease, transform 0.15s ease;
        }}
        .kpi-card::before {{
          content: "";
          position: absolute;
          left: 0; top: 0; bottom: 0;
          width: 3px;
          background: var(--kpi-accent, var(--accent));
          opacity: 0.95;
        }}
        .kpi-card:hover {{
          border-color: var(--card-border-strong);
          box-shadow: 0 8px 24px rgba(0,0,0,0.25);
          transform: translateY(-1px);
        }}
        .kpi-card .kpi-top {{
          display: flex;
          justify-content: space-between;
          align-items: flex-start;
          margin-bottom: 0.55rem;
        }}
        .kpi-card .kpi-label {{
          font-size: 0.72rem;
          text-transform: uppercase;
          letter-spacing: 0.07em;
          color: var(--muted);
          font-weight: 650;
        }}
        .kpi-card .kpi-icon {{
          width: 30px;
          height: 30px;
          min-width: 30px;
          min-height: 30px;
          max-width: 30px;
          max-height: 30px;
          border-radius: 8px;
          background: rgba(59,130,246,0.12);
          color: var(--kpi-accent, var(--accent));
          display: flex;
          align-items: center;
          justify-content: center;
          overflow: hidden;
          flex-shrink: 0;
        }}
        /* Hard cap — prevents escaped SVGs from filling the viewport */
        .kpi-card .kpi-icon svg,
        .kpi-icon svg,
        .sidebar-brand .mark svg {{
          width: 16px !important;
          height: 16px !important;
          max-width: 22px !important;
          max-height: 22px !important;
        }}
        .sidebar-brand .mark svg {{
          width: 22px !important;
          height: 22px !important;
          max-width: 22px !important;
          max-height: 22px !important;
        }}
        /* Nuclear safety: any decorative SVG leaking into markdown/html blocks */
        div[data-testid="stMarkdownContainer"] > svg,
        div[data-testid="stMarkdownContainer"] svg:not(.js-plotly-plot svg),
        .stHtml svg {{
          max-width: 40px !important;
          max-height: 40px !important;
        }}
        .kpi-card .kpi-value {{
          font-size: 1.55rem;
          font-weight: 700;
          color: var(--ink);
          line-height: 1.1;
          font-family: "JetBrains Mono", "DM Sans", monospace;
          letter-spacing: -0.03em;
        }}
        .kpi-card .kpi-value.accent {{ color: var(--kpi-accent, var(--accent)); }}
        .kpi-card .kpi-sub {{
          margin-top: 0.35rem;
          font-size: 0.78rem;
          color: var(--faint);
          line-height: 1.4;
        }}
        .kpi-card .kpi-delta {{
          display: inline-block;
          margin-top: 0.35rem;
          font-size: 0.78rem;
          font-weight: 650;
          color: var(--success);
        }}
        .section-block {{
          margin: 0.85rem 0 0.55rem 0;
        }}
        .section-title {{
          font-size: 1.15rem;
          font-weight: 650;
          color: var(--ink);
          letter-spacing: -0.02em;
        }}
        .section-sub {{
          margin-top: 0.2rem;
          font-size: 0.85rem;
          color: var(--muted);
          line-height: 1.4;
        }}
        .insight-card {{
          border: 1px solid var(--card-border);
          border-radius: var(--radius-sm);
          padding: 0.75rem 0.85rem;
          background: var(--bg-elevated);
          margin: 0.4rem 0;
        }}
        .insight-card .insight-title {{
          font-size: 0.78rem;
          font-weight: 700;
          letter-spacing: 0.06em;
          text-transform: uppercase;
          margin-bottom: 0.3rem;
        }}
        .insight-card .insight-body {{
          font-size: 0.9rem;
          line-height: 1.45;
          color: var(--ink);
        }}
        .insight-ok {{ border-left: 3px solid var(--success); }}
        .insight-ok .insight-title {{ color: var(--success); }}
        .insight-warn {{ border-left: 3px solid var(--warn); }}
        .insight-warn .insight-title {{ color: var(--warn); }}
        .insight-caveat {{ border-left: 3px solid var(--accent); }}
        .insight-caveat .insight-title {{ color: #93c5fd; }}
        .finding-panel {{
          border: 1px solid var(--card-border);
          border-radius: var(--radius);
          background: var(--card);
          padding: 0.9rem 1rem;
        }}
        .finding-title {{
          font-size: 0.95rem;
          font-weight: 650;
          color: var(--ink);
          margin-bottom: 0.65rem;
        }}
        .finding-row {{
          display: flex;
          justify-content: space-between;
          gap: 0.75rem;
          padding: 0.4rem 0;
          border-bottom: 1px solid var(--card-border);
          font-size: 0.88rem;
        }}
        .finding-row:last-child {{ border-bottom: none; }}
        .finding-row .fk {{ color: var(--muted); }}
        .finding-row .fv {{
          color: var(--ink);
          font-weight: 650;
          font-family: "JetBrains Mono", "DM Sans", monospace;
          text-align: right;
        }}
        .finding-foot {{
          margin-top: 0.55rem;
          font-size: 0.78rem;
          color: var(--faint);
          line-height: 1.4;
        }}

        /* Legacy aliases used by older page fragments */
        .stat-row {{
          display: grid;
          grid-template-columns: repeat(4, 1fr);
          gap: var(--space-2);
          margin-top: var(--space-1);
        }}
        .stat-card {{
          background: var(--card);
          border: 1px solid var(--card-border);
          border-left: 3px solid var(--accent);
          padding: var(--pad);
          border-radius: var(--radius);
        }}
        .stat-card .value {{
          font-size: 1.45rem;
          font-weight: 700;
          color: var(--ink);
          line-height: 1.1;
          font-family: "JetBrains Mono", monospace;
        }}
        .stat-card .label {{
          font-size: var(--font-caption);
          color: var(--muted);
          margin-top: 0.25rem;
        }}
        .stat-card .sublabel {{
          font-size: 0.72rem;
          color: var(--faint);
          margin-top: 0.2rem;
          line-height: 1.35;
        }}

        /* ========== SECTION CARDS ========== */
        /* Streamlit bordered containers → premium analytics cards */
        div[data-testid="stVerticalBlockBorderWrapper"] {{
          background: linear-gradient(180deg, var(--card-alt) 0%, var(--card) 100%) !important;
          border: 1px solid var(--card-border) !important;
          border-radius: var(--radius) !important;
          padding: 0.85rem 1rem 0.95rem 1rem !important;
          box-shadow: 0 1px 0 rgba(255,255,255,0.025) inset, 0 8px 28px rgba(0,0,0,0.18);
          margin-bottom: 0.35rem;
          transition: border-color 0.15s ease, box-shadow 0.15s ease;
        }}
        div[data-testid="stVerticalBlockBorderWrapper"]:hover {{
          border-color: var(--card-border-strong) !important;
        }}
        .card-head {{
          display: flex;
          justify-content: space-between;
          align-items: flex-start;
          gap: 0.75rem;
          margin-bottom: 0.55rem;
        }}
        .card-title {{
          font-size: 0.98rem;
          font-weight: 650;
          color: var(--ink);
          letter-spacing: -0.015em;
          margin: 0;
        }}
        .card-caption {{
          font-size: 0.75rem;
          color: var(--muted);
          margin-top: 0.2rem;
          line-height: 1.4;
        }}
        .card-link {{
          font-size: 0.78rem;
          color: var(--accent);
          font-weight: 600;
          text-decoration: none;
          white-space: nowrap;
        }}
        .card-footer-link {{
          margin-top: 0.55rem;
          padding-top: 0.55rem;
          border-top: 1px solid var(--card-border);
          font-size: 0.8rem;
          color: var(--accent);
          font-weight: 600;
        }}
        .dash-card {{
          background: var(--card);
          border: 1px solid var(--card-border);
          border-radius: var(--radius);
          padding: 1rem 1.1rem 1.05rem 1.1rem;
          margin-bottom: 0.85rem;
        }}

        /* ========== DISCLAIMER / NOTES ========== */
        .disclaimer-box {{
          background: linear-gradient(90deg, rgba(217,119,6,0.12), rgba(217,119,6,0.04));
          border: 1px solid rgba(217,119,6,0.45);
          border-radius: var(--radius);
          padding: 0.85rem 1rem;
          color: var(--warn-text);
          font-size: 0.88rem;
          line-height: 1.5;
          font-weight: 500;
          margin: 0 0 1rem 0;
        }}
        .disclaimer-box .label {{
          display: block;
          font-size: 0.7rem;
          letter-spacing: 0.1em;
          text-transform: uppercase;
          margin-bottom: 0.3rem;
          color: var(--warn);
          font-weight: 700;
        }}
        .honesty-box {{
          background: linear-gradient(90deg, rgba(59,130,246,0.12), rgba(59,130,246,0.04));
          border: 1px solid rgba(59,130,246,0.4);
          border-left: 4px solid var(--accent);
          border-radius: var(--radius);
          padding: 0.9rem 1rem;
          color: #d7e6f2;
          font-size: 0.9rem;
          line-height: 1.5;
          font-weight: 500;
          margin: 0.5rem 0 1rem 0;
        }}
        .honesty-box .label {{
          display: block;
          font-size: 0.7rem;
          letter-spacing: 0.1em;
          text-transform: uppercase;
          margin-bottom: 0.35rem;
          color: #93c5fd;
          font-weight: 700;
        }}
        .extreme-flag-box {{
          background: linear-gradient(90deg, rgba(239,68,68,0.12), rgba(239,68,68,0.04));
          border: 1px solid rgba(239,68,68,0.4);
          border-radius: var(--radius);
          padding: 0.85rem 1rem;
          color: #fecaca;
          font-weight: 500;
          margin: 0.5rem 0 1rem 0;
          line-height: 1.45;
          font-size: 0.9rem;
        }}
        .extreme-flag-box .label {{
          display: block;
          font-size: 0.7rem;
          letter-spacing: 0.08em;
          text-transform: uppercase;
          color: #f87171;
          margin-bottom: 0.3rem;
          font-weight: 700;
        }}

        /* ========== HIGHLIGHTS / LISTS ========== */
        .highlights-panel ul {{
          margin: 0;
          padding-left: 0;
          list-style: none;
        }}
        .highlights-panel li {{
          position: relative;
          padding-left: 1.35rem;
          margin-bottom: 0.65rem;
          line-height: 1.45;
          font-size: 0.88rem;
          color: var(--ink);
        }}
        .highlights-panel li::before {{
          content: "";
          position: absolute;
          left: 0;
          top: 0.35rem;
          width: 0.55rem;
          height: 0.55rem;
          border-radius: 50%;
          background: var(--success);
          box-shadow: 0 0 0 3px rgba(34,197,94,0.18);
        }}
        .highlight-item {{
          display: flex;
          gap: 0.65rem;
          align-items: flex-start;
          padding: 0.55rem 0;
          border-bottom: 1px solid var(--card-border);
          font-size: 0.86rem;
          line-height: 1.45;
          color: var(--ink);
        }}
        .highlight-item:last-child {{ border-bottom: none; }}
        .highlight-item .dot {{
          flex-shrink: 0;
          width: 18px;
          height: 18px;
          margin-top: 0.1rem;
          border-radius: 50%;
          background: rgba(34,197,94,0.15);
          color: var(--success);
          display: flex;
          align-items: center;
          justify-content: center;
          font-size: 0.7rem;
          font-weight: 700;
        }}
        .highlight-item.warn .dot {{
          background: rgba(245,158,11,0.15);
          color: var(--warn);
        }}
        .highlight-item.caveat .dot {{
          background: rgba(59,130,246,0.15);
          color: var(--accent);
        }}

        /* ========== ACTION / NAV CARDS ========== */
        .action-list {{
          display: flex;
          flex-direction: column;
          gap: 0.35rem;
        }}
        .action-row {{
          display: flex;
          align-items: center;
          justify-content: space-between;
          padding: 0.65rem 0.75rem;
          border-radius: var(--radius-sm);
          border: 1px solid var(--card-border);
          background: var(--bg-elevated);
          transition: border-color 0.15s ease, background 0.15s ease;
        }}
        .action-row:hover {{
          border-color: rgba(59,130,246,0.4);
          background: rgba(59,130,246,0.08);
        }}
        .action-row .action-label {{
          font-size: 0.88rem;
          font-weight: 600;
          color: var(--ink);
        }}
        .action-row .action-hint {{
          font-size: 0.72rem;
          color: var(--muted);
          margin-top: 0.1rem;
        }}
        .action-row .chev {{
          color: var(--accent);
          font-weight: 700;
          font-size: 1rem;
        }}
        .nav-card-row {{
          display: grid;
          grid-template-columns: repeat(4, 1fr);
          gap: var(--space-2);
          margin-top: var(--space-1);
        }}
        @media (max-width: 900px) {{
          .nav-card-row {{ grid-template-columns: repeat(2, 1fr); }}
        }}
        .nav-card {{
          background: var(--card);
          border: 1px solid var(--card-border);
          border-radius: var(--radius);
          padding: var(--pad);
          transition: border-color 0.15s ease, box-shadow 0.15s ease;
          min-height: 5.2rem;
        }}
        .nav-card:hover {{
          border-color: rgba(59,130,246,0.45);
          box-shadow: 0 4px 16px rgba(0,0,0,0.2);
        }}

        /* ========== EXTREME / MINI METRIC TILES ========== */
        .mini-metric-row {{
          display: grid;
          grid-template-columns: repeat(3, 1fr);
          gap: 0.55rem;
        }}
        .mini-metric {{
          border-radius: var(--radius-sm);
          padding: 0.7rem 0.75rem;
          border: 1px solid var(--card-border);
          background: var(--bg-elevated);
        }}
        .mini-metric .mm-label {{
          font-size: 0.68rem;
          text-transform: uppercase;
          letter-spacing: 0.06em;
          color: var(--muted);
          font-weight: 650;
        }}
        .mini-metric .mm-value {{
          font-size: 1.05rem;
          font-weight: 700;
          margin-top: 0.25rem;
          font-family: "JetBrains Mono", monospace;
        }}
        .mini-metric.red {{ border-color: rgba(239,68,68,0.35); }}
        .mini-metric.red .mm-value {{ color: #f87171; }}
        .mini-metric.amber {{ border-color: rgba(245,158,11,0.35); }}
        .mini-metric.amber .mm-value {{ color: #fbbf24; }}
        .mini-metric.green {{ border-color: rgba(34,197,94,0.35); }}
        .mini-metric.green .mm-value {{ color: #4ade80; }}

        /* ========== MISC ========== */
        .page-context {{
          font-size: var(--font-caption);
          color: var(--muted);
          letter-spacing: 0.02em;
          margin: 0 0 var(--space-2) 0;
          padding-bottom: 0.35rem;
          border-bottom: 1px solid var(--card-border);
        }}
        .page-context .sep {{
          margin: 0 0.35rem;
          color: var(--card-border);
        }}
        .empty-state {{
          background: var(--bg-elevated);
          border: 1px dashed var(--card-border);
          border-radius: var(--radius);
          padding: var(--pad);
          color: var(--muted);
          font-size: var(--font-body);
          line-height: 1.45;
          margin: var(--space-2) 0;
        }}
        .empty-state .label {{
          display: block;
          font-size: 0.75rem;
          letter-spacing: 0.08em;
          text-transform: uppercase;
          color: var(--accent);
          font-weight: 700;
          margin-bottom: 0.35rem;
        }}
        .secondary-gnn-note {{
          color: var(--muted);
          font-size: 0.9rem;
          margin-bottom: 0.5rem;
        }}
        .sig-yes {{ color: var(--success); font-weight: 700; }}
        .sig-no {{ color: var(--muted); font-weight: 600; }}
        .caveat-chip {{
          display: inline-block;
          background: var(--accent-soft);
          border: 1px solid var(--accent);
          color: var(--ink);
          border-radius: 6px;
          padding: 0.2rem 0.5rem;
          font-size: var(--font-caption);
          font-weight: 600;
        }}
        .home-lede {{
          font-size: 0.95rem;
          line-height: 1.55;
          color: var(--muted);
          max-width: 52rem;
          margin-bottom: 0.75rem;
        }}
        .section-label {{
          font-size: 0.72rem;
          font-weight: 700;
          letter-spacing: 0.1em;
          text-transform: uppercase;
          color: var(--faint);
          margin: 0.35rem 0 0.55rem 0;
        }}
        .station-pill {{
          display: inline-flex;
          align-items: center;
          gap: 0.45rem;
          padding: 0.4rem 0.75rem;
          background: rgba(34,197,94,0.1);
          border: 1px solid rgba(34,197,94,0.35);
          border-radius: 999px;
          color: #86efac;
          font-size: 0.85rem;
          font-weight: 600;
          margin-bottom: 0.75rem;
        }}
        .station-pill.empty {{
          background: rgba(139,155,180,0.08);
          border-color: var(--card-border);
          color: var(--muted);
        }}
        .detail-grid {{
          display: grid;
          grid-template-columns: repeat(2, 1fr);
          gap: 0.55rem;
        }}
        .detail-cell {{
          background: var(--bg-elevated);
          border: 1px solid var(--card-border);
          border-radius: var(--radius-sm);
          padding: 0.65rem 0.75rem;
        }}
        .detail-cell .dk {{
          font-size: 0.68rem;
          text-transform: uppercase;
          letter-spacing: 0.06em;
          color: var(--muted);
          font-weight: 650;
        }}
        .detail-cell .dv {{
          margin-top: 0.2rem;
          font-size: 0.95rem;
          font-weight: 650;
          color: var(--ink);
          font-family: "JetBrains Mono", "DM Sans", monospace;
        }}

        div[data-testid="stCaption"] {{ color: var(--muted) !important; }}
        div[data-testid="stMetric"] {{
          background: var(--card);
          border: 1px solid var(--card-border);
          border-radius: var(--radius-sm);
          padding: 0.65rem 0.8rem;
        }}
        div[data-testid="stMetric"] label {{ color: var(--muted) !important; }}
        div[data-testid="stMetric"] [data-testid="stMetricValue"] {{
          color: var(--ink) !important;
          font-family: "JetBrains Mono", monospace;
        }}

        /* Dataframes */
        div[data-testid="stDataFrame"] {{
          border: 1px solid var(--card-border);
          border-radius: var(--radius-sm);
          overflow: hidden;
        }}

        /* Buttons / inputs */
        div[data-testid="stButton"] > button {{
          border-radius: 8px;
          border: 1px solid var(--card-border);
          background: var(--card);
          color: var(--ink);
          font-size: var(--font-caption);
          transition: border-color 0.15s ease, background-color 0.15s ease;
        }}
        div[data-testid="stButton"] > button:hover {{
          border-color: var(--accent);
          background: var(--bg-elevated);
          color: var(--ink);
        }}
        div[data-testid="stTextInput"] input,
        div[data-testid="stSelectbox"] > div,
        div[data-baseweb="select"] > div {{
          border-radius: 8px !important;
        }}

        /* Info / warning Streamlit boxes */
        div[data-testid="stAlert"] {{
          border-radius: var(--radius-sm);
          border: 1px solid var(--card-border);
        }}

        /* Plotly container breathing room */
        div[data-testid="stPlotlyChart"] {{
          border-radius: var(--radius-sm);
        }}

        /* Tighter dividers */
        hr {{
          border: none !important;
          border-top: 1px solid var(--card-border) !important;
          margin: 0.85rem 0 !important;
        }}

        /* Compact radio / tabs feel */
        div[role="radiogroup"] label {{
          background: var(--card) !important;
          border: 1px solid var(--card-border) !important;
          border-radius: 8px !important;
          padding: 0.25rem 0.65rem !important;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.session_state[_SS_CSS_RUN] = run_id


def render_sidebar_brand() -> None:
    """Branded sidebar header + compact historical-replay note (Option C)."""
    run_id = _script_run_id()
    if st.session_state.get(_SS_SIDEBAR_RUN) == run_id:
        return

    # Explicit px sizing — prevents giant-icon blowups if CSS fails to bind
    cloud_svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" width="22" height="22" viewBox="0 0 24 24" '
        'fill="none" style="width:22px;height:22px;max-width:22px;max-height:22px;display:block;">'
        '<path d="M7.5 19h9.2a4.3 4.3 0 0 0 .4-8.58A6 6 0 0 0 6.2 12.1 3.7 3.7 0 0 0 7.5 19Z" '
        'stroke="#fff" stroke-width="1.7" stroke-linejoin="round"/>'
        '<path d="M9 21v1M12 21v2M15 21v1" stroke="#bfdbfe" stroke-width="1.7" '
        'stroke-linecap="round"/>'
        "</svg>"
    )
    brand_html = (
        '<div class="sidebar-brand">'
        f'<div class="mark">{cloud_svg}</div>'
        '<div class="titles">'
        '<div class="name">Rainfall Prediction</div>'
        '<div class="sub">Historical Forecast Analytics</div>'
        "</div></div>"
        '<div class="sidebar-disclaimer" role="note">'
        '<span class="lbl">Historical Replay Only</span>'
        f"Locked test period {TEST_PERIOD_START} → {TEST_PERIOD_END}. "
        "No live forecasts.</div>"
    )
    with st.sidebar:
        if hasattr(st, "html"):
            st.html(brand_html)
        else:
            st.markdown(brand_html, unsafe_allow_html=True)
    st.session_state[_SS_SIDEBAR_RUN] = run_id


def render_empty_state(title: str, body: str) -> None:
    """Styled empty/unavailable panel (no new data — presentation only)."""
    from .ui_components import render_html
    import html as html_lib

    render_html(
        '<div class="empty-state" role="status">'
        f'<span class="label">{html_lib.escape(title)}</span>'
        f"{html_lib.escape(body)}</div>"
    )


def render_header_badge_bar() -> None:
    """Persistent scope badges (test period / stations / horizons). Idempotent per run."""
    run_id = _script_run_id()
    if st.session_state.get(_SS_BADGE_RUN) == run_id:
        return

    from .ui_components import render_html

    horizons = ", ".join(f"h={h}" for h in FORECAST_HORIZONS)
    render_html(
        '<div class="status-chip-row" role="region" aria-label="Evaluation scope" '
        'style="justify-content:flex-start;margin-bottom:0.85rem;">'
        f'<div class="status-chip"><span class="k">Test period</span>'
        f'<span class="v">{TEST_PERIOD_START} → {TEST_PERIOD_END}</span></div>'
        f'<div class="status-chip"><span class="k">Stations</span>'
        f'<span class="v">{N_USABLE_STATIONS} usable</span></div>'
        f'<div class="status-chip"><span class="k">Horizons</span>'
        f'<span class="v">{horizons}</span></div>'
        '<div class="status-chip"><span class="k">Models</span>'
        f'<span class="v">{N_PRIMARY_MODELS} primary</span></div></div>'
    )
    st.session_state[_SS_BADGE_RUN] = run_id


def render_page_context(trail: str) -> None:
    """Small breadcrumb under header chrome (navigational only)."""
    from .ui_components import render_html
    import html as html_lib

    parts = [p.strip() for p in trail.split(">") if p.strip()]
    if not parts:
        return
    inner = '<span class="sep">›</span>'.join(html_lib.escape(p) for p in parts)
    render_html(f'<div class="page-context" aria-label="Page context">{inner}</div>')


def render_page_chrome(page_context: str | None = None, *, show_badges: bool = True) -> None:
    """Top-of-page chrome: design-system CSS (once/run) + optional badge bar."""
    inject_base_css()
    if show_badges:
        render_header_badge_bar()
    if page_context:
        render_page_context(page_context)


def render_disclaimer() -> None:
    from .ui_components import render_html

    render_html(
        '<div class="disclaimer-box" role="note">'
        '<span class="label">Historical Forecast Replay</span>'
        f"{DISCLAIMER_TEXT}</div>"
    )


def render_dual_number_note() -> None:
    """Explain seed-42 replay vs 3-seed research aggregates when both appear."""
    from .ui_components import render_html

    render_html(
        '<div class="honesty-box" role="note">'
        '<span class="label">Two number systems on this page</span>'
        f"{DUAL_NUMBER_NOTE}</div>"
    )
