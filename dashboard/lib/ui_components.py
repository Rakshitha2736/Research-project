"""Reusable UI building blocks for the rainfall forecast dashboard.

CRITICAL: Complex HTML must go through ``render_html()`` (uses ``st.html`` when
available). Never pass multi-line nested HTML through ``st.markdown`` — Streamlit's
Markdown parser can split tags across blocks and show raw ``<div>`` text, and
unconstrained SVGs can expand to full-page size.
"""

from __future__ import annotations

import html as html_lib
from typing import Iterable, Sequence

import plotly.graph_objects as go
import streamlit as st

from .paths import (
    FORECAST_HORIZONS,
    N_PRIMARY_MODELS,
    N_USABLE_STATIONS,
    TEST_PERIOD_END,
    TEST_PERIOD_START,
)
from .style import DS


def _esc(text: object) -> str:
    """Escape text for safe insertion into HTML attribute/body content."""
    return html_lib.escape("" if text is None else str(text), quote=True)


def render_html(markup: str) -> None:
    """Render raw HTML without Markdown parsing (prevents visible tag leakage)."""
    # Collapse accidental blank lines that would break Markdown fallbacks.
    compact = "\n".join(line for line in markup.splitlines() if line.strip() != "")
    if hasattr(st, "html"):
        st.html(compact)
    else:
        # Last resort: single-line markdown (still safer than multi-paragraph HTML)
        st.markdown(compact.replace("\n", ""), unsafe_allow_html=True)


# ---- Inline SVG icons: ALWAYS include explicit width/height (px) ----
def _svg(paths: str, *, size: int = 16) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" '
        f'viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" '
        f'stroke-linecap="round" stroke-linejoin="round" '
        f'style="width:{size}px;height:{size}px;max-width:{size}px;max-height:{size}px;'
        f'display:block;flex-shrink:0;">'
        f"{paths}</svg>"
    )


_ICON_PATHS: dict[str, str] = {
    "database": (
        '<ellipse cx="12" cy="5" rx="8" ry="3"/>'
        '<path d="M4 5v6c0 1.7 3.6 3 8 3s8-1.3 8-3V5"/>'
        '<path d="M4 11v6c0 1.7 3.6 3 8 3s8-1.3 8-3v-6"/>'
    ),
    "map-pin": (
        '<path d="M12 21s-7-5.4-7-11a7 7 0 1 1 14 0c0 5.6-7 11-7 11z"/>'
        '<circle cx="12" cy="10" r="2.5"/>'
    ),
    "clock": '<circle cx="12" cy="12" r="9"/><path d="M12 7v5l3 2"/>',
    "layers": (
        '<path d="m12 3 9 5-9 5-9-5 9-5z"/>'
        '<path d="m3 12 9 5 9-5"/>'
        '<path d="m3 17 9 5 9-5"/>'
    ),
    "trending-up": '<path d="m3 17 6-6 4 4 7-7"/><path d="M14 8h6v6"/>',
    "activity": '<path d="M22 12h-4l-3 9L9 3l-3 9H2"/>',
    "alert": (
        '<path d="M10.3 3.9 1.8 18a2 2 0 0 0 1.7 3h17a2 2 0 0 0 1.7-3L13.7 3.9a2 2 0 0 0-3.4 0z"/>'
        '<path d="M12 9v4"/><path d="M12 17h.01"/>'
    ),
    "brain": (
        '<path d="M12 5a4 4 0 0 0-4 4v1a3 3 0 0 0-2 2.7V16a3 3 0 0 0 3 3h1"/>'
        '<path d="M12 5a4 4 0 0 1 4 4v1a3 3 0 0 1 2 2.7V16a3 3 0 0 1-3 3h-1"/>'
        '<path d="M9 12h6"/><path d="M9 16h6"/>'
    ),
    "cloud": (
        '<path d="M7.5 19h9.2a4.3 4.3 0 0 0 .4-8.58A6 6 0 0 0 6.2 12.1 3.7 3.7 0 0 0 7.5 19Z"/>'
        '<path d="M9 21v1M12 21v2M15 21v1"/>'
    ),
}


def icon_svg(name: str, *, size: int = 16) -> str:
    paths = _ICON_PATHS.get(name, _ICON_PATHS["activity"])
    return _svg(paths, size=size)


def apply_plotly_theme(fig: go.Figure, *, height: int | None = None) -> go.Figure:
    """Apply consistent dark analytics theme to a Plotly figure (layout only)."""
    layout_kwargs = dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=DS["ink"], family="DM Sans, Segoe UI, sans-serif", size=13),
        margin=dict(l=52, r=18, t=40, b=44),
        legend=dict(
            bgcolor="rgba(18,26,43,0.85)",
            bordercolor=DS["card_border"],
            borderwidth=1,
            font=dict(size=12),
        ),
        hoverlabel=dict(
            bgcolor=DS["card"],
            bordercolor=DS["card_border"],
            font=dict(color=DS["ink"], size=12),
        ),
    )
    if height is not None:
        layout_kwargs["height"] = height
    fig.update_layout(**layout_kwargs)
    fig.update_xaxes(
        gridcolor="rgba(42,58,85,0.45)",
        zerolinecolor="rgba(42,58,85,0.55)",
        linecolor=DS["card_border"],
        tickfont=dict(size=12, color=DS["muted"]),
        title_font=dict(size=12, color=DS["muted"]),
    )
    fig.update_yaxes(
        gridcolor="rgba(42,58,85,0.45)",
        zerolinecolor="rgba(42,58,85,0.55)",
        linecolor=DS["card_border"],
        tickfont=dict(size=12, color=DS["muted"]),
        title_font=dict(size=12, color=DS["muted"]),
    )
    return fig


def render_page_header(
    title: str,
    *,
    subtitle: str | None = None,
    eyebrow: str | None = None,
    show_status_chips: bool = True,
) -> None:
    """Professional page header with optional compact status chips."""
    chips = ""
    if show_status_chips:
        horizons = ", ".join(f"h={h}" for h in FORECAST_HORIZONS)
        chips = (
            '<div class="status-chip-row">'
            f'<div class="status-chip"><span class="k">Test Period</span>'
            f'<span class="v">{_esc(TEST_PERIOD_START)} → {_esc(TEST_PERIOD_END)}</span></div>'
            f'<div class="status-chip"><span class="k">Stations</span>'
            f'<span class="v">{N_USABLE_STATIONS} usable</span></div>'
            f'<div class="status-chip"><span class="k">Horizons</span>'
            f'<span class="v">{_esc(horizons)}</span></div>'
            f'<div class="status-chip"><span class="k">Models</span>'
            f'<span class="v">{N_PRIMARY_MODELS} primary</span></div>'
            "</div>"
        )

    eyebrow_html = (
        f'<div class="eyebrow">{_esc(eyebrow)}</div>' if eyebrow else ""
    )
    subtitle_html = (
        f'<div class="subtitle">{_esc(subtitle)}</div>' if subtitle else ""
    )
    render_html(
        '<div class="page-hero">'
        f'<div class="hero-text">{eyebrow_html}<h1>{_esc(title)}</h1>{subtitle_html}</div>'
        f"{chips}</div>"
    )


def render_kpi_card(
    *,
    label: str,
    value: str,
    sublabel: str = "",
    icon: str = "activity",
    accent: str | None = None,
    value_accent: bool = False,
    delta: str = "",
) -> str:
    """Return a single-line HTML fragment for one KPI card (compose into a row)."""
    color = accent or DS["accent"]
    val_class = "kpi-value accent" if value_accent else "kpi-value"
    delta_html = f'<div class="kpi-delta">{_esc(delta)}</div>' if delta else ""
    sub_html = f'<div class="kpi-sub">{_esc(sublabel)}</div>' if sublabel else ""
    # Use direct border-left (not CSS custom props) — more reliable across sanitizers
    return (
        f'<div class="kpi-card" style="border-left:3px solid {_esc(color)};">'
        f'<div class="kpi-top">'
        f'<div class="kpi-label">{_esc(label)}</div>'
        f'<div class="kpi-icon" style="color:{_esc(color)};background:rgba(59,130,246,0.12);">'
        f"{icon_svg(icon, size=16)}</div></div>"
        f'<div class="{val_class}" style="{"color:" + _esc(color) + ";" if value_accent else ""}">'
        f"{_esc(value)}</div>"
        f"{delta_html}{sub_html}</div>"
    )


def render_kpi_row(
    cards_html: Iterable[str],
    *,
    columns: int | None = None,
) -> None:
    """Render a KPI row. ``columns`` overrides the default 5-column grid."""
    cards = list(cards_html)
    n = columns if columns is not None else max(len(cards), 1)
    style = f' style="grid-template-columns:repeat({n},minmax(0,1fr));"'
    render_html(f'<div class="kpi-row"{style}>{"".join(cards)}</div>')


def render_card_header(title: str, caption: str = "", link_label: str = "") -> None:
    """Card title bar (pair with ``with card_container():`` for bordered panels)."""
    link = f'<span class="card-link">{_esc(link_label)}</span>' if link_label else ""
    caption_html = (
        f'<div class="card-caption">{_esc(caption)}</div>' if caption else ""
    )
    render_html(
        '<div class="card-head"><div>'
        f'<div class="card-title">{_esc(title)}</div>{caption_html}'
        f"</div>{link}</div>"
    )


def card_container():
    """Bordered Streamlit container styled as a dash card."""
    return st.container(border=True)


def open_card(title: str, caption: str = "", link_label: str = "") -> None:
    render_card_header(title, caption=caption, link_label=link_label)


def close_card(footer: str = "") -> None:
    if footer:
        render_html(f'<div class="card-footer-link">{_esc(footer)}</div>')


def render_section_label(text: str) -> None:
    render_html(f'<div class="section-label">{_esc(text)}</div>')


def render_section_header(title: str, subtitle: str = "") -> None:
    """Larger section heading used between major page blocks."""
    sub = f'<div class="section-sub">{_esc(subtitle)}</div>' if subtitle else ""
    render_html(
        f'<div class="section-block"><div class="section-title">{_esc(title)}</div>{sub}</div>'
    )


def render_highlight_items(items: Sequence[tuple[str, str]]) -> None:
    """items: list of (tone, text) where tone in {{ok, warn, caveat}}."""
    blocks: list[str] = []
    for tone, text in items:
        cls = "highlight-item"
        mark = "✓"
        if tone == "warn":
            cls += " warn"
            mark = "!"
        elif tone == "caveat":
            cls += " caveat"
            mark = "i"
        blocks.append(
            f'<div class="{cls}"><div class="dot">{mark}</div>'
            f"<div>{_esc(text)}</div></div>"
        )
    render_html(f'<div class="highlights-panel">{"".join(blocks)}</div>')


def render_mini_metrics(items: Sequence[tuple[str, str, str]]) -> None:
    """items: (tone, label, value) tone in {{red, amber, green}}."""
    cells = []
    for tone, label, value in items:
        cells.append(
            f'<div class="mini-metric {_esc(tone)}">'
            f'<div class="mm-label">{_esc(label)}</div>'
            f'<div class="mm-value">{_esc(value)}</div></div>'
        )
    render_html(f'<div class="mini-metric-row">{"".join(cells)}</div>')


def render_detail_grid(pairs: Sequence[tuple[str, str]]) -> None:
    cells = "".join(
        f'<div class="detail-cell"><div class="dk">{_esc(k)}</div>'
        f'<div class="dv">{_esc(v)}</div></div>'
        for k, v in pairs
    )
    render_html(f'<div class="detail-grid">{cells}</div>')


def render_station_pill(text: str, *, empty: bool = False) -> None:
    cls = "station-pill empty" if empty else "station-pill"
    render_html(f'<div class="{cls}">{_esc(text)}</div>')


def render_status_panel(label: str, body: str, *, tone: str = "info") -> None:
    """Research status / caveat / disclaimer panel."""
    cls = {
        "info": "honesty-box",
        "warn": "disclaimer-box",
        "danger": "extreme-flag-box",
    }.get(tone, "honesty-box")
    render_html(
        f'<div class="{cls}" role="note">'
        f'<span class="label">{_esc(label)}</span>{_esc(body)}</div>'
    )


def render_insight_card(title: str, body: str, *, tone: str = "ok") -> None:
    tone_cls = {"ok": "insight-ok", "warn": "insight-warn", "caveat": "insight-caveat"}.get(
        tone, "insight-ok"
    )
    render_html(
        f'<div class="insight-card {tone_cls}">'
        f'<div class="insight-title">{_esc(title)}</div>'
        f'<div class="insight-body">{_esc(body)}</div></div>'
    )


def render_finding_block(
    title: str,
    rows: Sequence[tuple[str, str]],
    *,
    footnote: str = "",
) -> None:
    """Compact research finding panel (label/value rows)."""
    row_html = "".join(
        f'<div class="finding-row"><span class="fk">{_esc(k)}</span>'
        f'<span class="fv">{_esc(v)}</span></div>'
        for k, v in rows
    )
    foot = f'<div class="finding-foot">{_esc(footnote)}</div>' if footnote else ""
    render_html(
        f'<div class="finding-panel"><div class="finding-title">{_esc(title)}</div>'
        f"{row_html}{foot}</div>"
    )


def render_action_links(actions: Sequence[tuple[str, str, str]]) -> None:
    """actions: (page_path, label, hint). Uses st.page_link inside styled rows."""
    for path, label, hint in actions:
        cols = st.columns([12, 1])
        with cols[0]:
            try:
                st.page_link(path, label=label)
            except Exception:
                st.markdown(f"**{label}**")
            st.caption(hint)
        with cols[1]:
            render_html('<div class="chev">→</div>')
