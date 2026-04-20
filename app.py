"""
app.py  —  Extreme Precipitation Indices Dashboard
===================================================
Single-file Streamlit app. Deploy straight to Streamlit Community Cloud.

Run locally:
    pip install streamlit pandas numpy matplotlib pymannkendall scipy
    streamlit run app.py
"""

import io
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.colors import LinearSegmentedColormap
import streamlit as st

warnings.filterwarnings("ignore")

try:
    import pymannkendall as mk
    from scipy.stats import theilslopes
    TREND_AVAILABLE = True
except ImportError:
    TREND_AVAILABLE = False

# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

INDEX_META = {
    "Rx1day":  {"unit": "mm",     "category": "Flood",    "desc": "Max 1-day precipitation per year"},
    "Rx5day":  {"unit": "mm",     "category": "Flood",    "desc": "Max consecutive 5-day precipitation"},
    "R95pTOT": {"unit": "mm",     "category": "Flood",    "desc": "Annual total on very wet days (>95th pct)"},
    "N95":     {"unit": "days",   "category": "Flood",    "desc": "Days above 95th percentile"},
    "R10mm":   {"unit": "days",   "category": "Sediment", "desc": "Heavy precipitation days (≥10 mm)"},
    "R20mm":   {"unit": "days",   "category": "Sediment", "desc": "Very heavy precipitation days (≥20 mm)"},
    "SDII":    {"unit": "mm/day", "category": "Sediment", "desc": "Simple daily intensity index"},
    "CDD":     {"unit": "days",   "category": "Storage",  "desc": "Max consecutive dry days (<1 mm)"},
    "CWD":     {"unit": "days",   "category": "Storage",  "desc": "Max consecutive wet days (≥1 mm)"},
    "PRCPTOT": {"unit": "mm",     "category": "Storage",  "desc": "Annual total wet-day precipitation"},
}

CATEGORY_COLORS = {"Flood": "#63b3ed", "Sediment": "#f6ad55", "Storage": "#68d391"}
CATEGORY_INDICES = {
    "Flood":    ["Rx1day", "Rx5day", "R95pTOT", "N95"],
    "Sediment": ["R10mm", "R20mm", "SDII"],
    "Storage":  ["CDD", "CWD", "PRCPTOT"],
}

MARKERS    = ["o", "s", "^", "D", "v", "P", "X", "*", "None"]
LINESTYLES = {"Solid": "-", "Dashed": "--", "Dotted": ":", "Dash-dot": "-."}
COLORMAPS  = ["coolwarm", "RdYlBu", "viridis", "plasma", "magma",
              "inferno", "Spectral", "Blues", "Greens", "PuOr"]
PALETTE    = ["#63b3ed","#f6ad55","#68d391","#fc8181","#b794f4",
              "#76e4f7","#fbb6ce","#c3dafe","#fbd38d","#9ae6b4"]

# ══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG  (must be first Streamlit call)
# ══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Extreme Precipitation Indices",
    page_icon="🌧️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════════════════
# CSS
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;600&family=Syne:wght@400;700&display=swap');

html, body, [class*="css"] { font-family: 'Syne', sans-serif; }

.stApp { background-color: #080c12; color: #dde3ed; }

[data-testid="stSidebar"] {
    background: linear-gradient(180deg,#0d1220 0%,#0a0f1a 100%);
    border-right: 1px solid #1e2d45;
}

h1 { font-family:'JetBrains Mono',monospace !important; color:#5eb8ff !important; letter-spacing:-2px; }
h2,h3 { font-family:'JetBrains Mono',monospace !important; color:#7ecfff !important; }

[data-testid="metric-container"] {
    background: linear-gradient(135deg,#0f1e32,#0d1828);
    border: 1px solid #1e3a5f; border-radius:10px; padding:14px 18px;
}
[data-testid="metric-container"] label { color:#4a7fa5 !important; font-size:10px !important; text-transform:uppercase; letter-spacing:1.5px; font-family:'JetBrains Mono',monospace !important; }
[data-testid="stMetricValue"] { color:#5eb8ff !important; font-family:'JetBrains Mono',monospace !important; font-size:1.5rem !important; }

.stTabs [data-baseweb="tab-list"] { background:#0d1220; border-bottom:1px solid #1e2d45; gap:2px; }
.stTabs [data-baseweb="tab"] { background:transparent; color:#4a6080; font-family:'JetBrains Mono',monospace; font-size:11px; padding:10px 18px; border-radius:6px 6px 0 0; }
.stTabs [aria-selected="true"] { background:#0f1e32 !important; color:#5eb8ff !important; border-bottom:2px solid #5eb8ff !important; }

.stSelectbox label,.stMultiSelect label,.stSlider label,.stNumberInput label,.stCheckbox label,.stColorPicker label {
    color:#4a7fa5 !important; font-size:10px !important; text-transform:uppercase; letter-spacing:1.2px; font-family:'JetBrains Mono',monospace !important;
}

.index-desc { font-size:12px; color:#4a7fa5; font-family:'JetBrains Mono',monospace; border-left:3px solid #1e3a5f; padding-left:10px; margin-bottom:14px; }

.stDownloadButton button { background:#0f1e32 !important; border:1px solid #1e3a5f !important; color:#5eb8ff !important; font-family:'JetBrains Mono',monospace !important; font-size:11px !important; border-radius:6px !important; }
.stDownloadButton button:hover { border-color:#5eb8ff !important; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# MATPLOTLIB DARK THEME
# ══════════════════════════════════════════════════════════════════════════════

plt.rcParams.update({
    "figure.facecolor": "#080c12",
    "axes.facecolor":   "#0d1220",
    "axes.edgecolor":   "#1e2d45",
    "axes.labelcolor":  "#718096",
    "axes.titlecolor":  "#7ecfff",
    "xtick.color":      "#718096",
    "ytick.color":      "#718096",
    "grid.color":       "#1e2d45",
    "grid.linestyle":   "--",
    "grid.linewidth":   0.6,
    "text.color":       "#dde3ed",
    "font.family":      "monospace",
    "figure.dpi":       110,
})

# ══════════════════════════════════════════════════════════════════════════════
# DATA FUNCTIONS  (pure, no Streamlit)
# ══════════════════════════════════════════════════════════════════════════════

def compute_r95(df, baseline):
    tmp = df.copy()
    tmp["year"] = tmp["date"].dt.year
    base = tmp[(tmp["year"] >= baseline[0]) & (tmp["year"] <= baseline[1])]
    wet = base["rain"].dropna()
    wet = wet[wet >= 1]
    return float(np.percentile(wet if len(wet) else base["rain"].dropna(), 95))


def compute_indices(df, r95):
    df = df.copy()
    df["year"] = df["date"].dt.year
    rows = []
    for yr, g in df.groupby("year"):
        rain = g["rain"]
        wet_days = rain[rain >= 1]

        dry = rain < 1
        grps = (dry != dry.shift()).cumsum()
        lens = dry.groupby(grps).sum(); lens = lens[lens > 0]
        cdd = int(lens.max()) if len(lens) else 0

        wet = rain >= 1
        grps = (wet != wet.shift()).cumsum()
        lens = wet.groupby(grps).sum(); lens = lens[lens > 0]
        cwd = int(lens.max()) if len(lens) else 0

        rows.append({
            "year":    yr,
            "Rx1day":  rain.max(),
            "Rx5day":  rain.rolling(5).sum().max(),
            "R95pTOT": rain[rain > r95].sum(),
            "N95":     int((rain > r95).sum()),
            "R10mm":   int((rain >= 10).sum()),
            "R20mm":   int((rain >= 20).sum()),
            "SDII":    wet_days.sum() / len(wet_days) if len(wet_days) else np.nan,
            "CDD":     cdd,
            "CWD":     cwd,
            "PRCPTOT": wet_days.sum(),
        })
    return pd.DataFrame(rows)


@st.cache_data(show_spinner="Computing indices…")
def load_and_compute(raw_bytes, b_start, b_end):
    df = pd.read_csv(io.BytesIO(raw_bytes))
    date_col = next((c for c in df.columns if c.lower() == "date"), None)
    if not date_col:
        raise ValueError("No 'Date' column found.")
    df[date_col] = pd.to_datetime(df[date_col], dayfirst=True)
    df = df.rename(columns={date_col: "date"})
    df.columns = df.columns.str.strip().str.replace(r"[ (),]", "_", regex=True)

    df_long = df.melt(id_vars="date", var_name="station", value_name="rain")
    df_long["rain"] = pd.to_numeric(df_long["rain"], errors="coerce")
    df_long = df_long.sort_values(["station", "date"]).reset_index(drop=True)

    parts = []
    for st in df_long["station"].unique():
        d = df_long[df_long["station"] == st].copy()
        r95 = compute_r95(d, (b_start, b_end))
        idx = compute_indices(d, r95)
        idx["station"] = st
        parts.append(idx)
    return pd.concat(parts, ignore_index=True)


def compute_trend(series):
    s = series.dropna()
    if not TREND_AVAILABLE or len(s) < 10:
        return np.nan, np.nan, "n/a"
    res = mk.original_test(s)
    slope, *_ = theilslopes(s)
    return float(res.p), float(slope), ("↑" if slope > 0 else "↓")

# ══════════════════════════════════════════════════════════════════════════════
# PLOT STYLE CONTROLS  (returns a dict of style params)
# ══════════════════════════════════════════════════════════════════════════════

def plot_controls(key, default_color="#63b3ed", show_colormap=False):
    """Renders an expander with all visual knobs. Returns a style dict."""
    with st.expander("🎨  Plot style", expanded=False):
        r1c1, r1c2, r1c3, r1c4 = st.columns(4)
        line_color  = r1c1.color_picker("Line colour",  default_color,  key=f"{key}_lc")
        ls_label    = r1c2.selectbox("Line style", list(LINESTYLES), key=f"{key}_ls")
        line_width  = r1c3.slider("Line width",  0.5, 6.0, 2.2, 0.25, key=f"{key}_lw")
        fig_height  = r1c4.slider("Plot height", 2.0, 10.0, 4.0, 0.5,  key=f"{key}_fh")

        r2c1, r2c2, r2c3, r2c4 = st.columns(4)
        marker      = r2c1.selectbox("Marker", MARKERS, key=f"{key}_mk")
        marker_size = r2c2.slider("Marker size", 0.0, 14.0, 5.0, 0.5, key=f"{key}_ms")
        fill_color  = r2c3.color_picker("Fill colour", default_color,   key=f"{key}_fc")
        fill_alpha  = r2c4.slider("Fill opacity", 0.0, 1.0, 0.15, 0.05, key=f"{key}_fa")

        r3c1, r3c2, r3c3, r3c4 = st.columns(4)
        show_fill   = r3c1.checkbox("Show fill",  True,  key=f"{key}_sf")
        show_trend  = r3c2.checkbox("Trend line", True,  key=f"{key}_st")
        grid_alpha  = r3c3.slider("Grid opacity", 0.0, 1.0, 0.25, 0.05, key=f"{key}_ga")
        colormap    = r3c4.selectbox("Colormap", COLORMAPS, key=f"{key}_cm") if show_colormap \
                      else "coolwarm"

        trend_color, trend_alpha, trend_width = "#ffffff", 0.45, 1.2
        if show_trend:
            r4c1, r4c2, r4c3, _ = st.columns(4)
            trend_color = r4c1.color_picker("Trend colour", "#ffffff", key=f"{key}_tc")
            trend_alpha = r4c2.slider("Trend opacity", 0.0, 1.0, 0.45, 0.05, key=f"{key}_ta")
            trend_width = r4c3.slider("Trend width",  0.5, 4.0, 1.2, 0.25,  key=f"{key}_tw")

    return dict(
        line_color=line_color, fill_color=fill_color, fill_alpha=fill_alpha,
        line_width=line_width, marker=None if marker == "None" else marker,
        marker_size=marker_size, linestyle=LINESTYLES[ls_label],
        show_fill=show_fill, show_trend=show_trend,
        trend_color=trend_color, trend_alpha=trend_alpha, trend_width=trend_width,
        grid_alpha=grid_alpha, fig_height=fig_height, colormap=colormap,
    )

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE BUILDERS  (return fig, never display)
# ══════════════════════════════════════════════════════════════════════════════

def _trend_line(ax, years, vals, s):
    valid = ~np.isnan(vals.astype(float))
    if valid.sum() < 3:
        return
    z = np.polyfit(years[valid], vals[valid].astype(float), 1)
    ax.plot(years, np.poly1d(z)(years), "--",
            color=s["trend_color"], alpha=s["trend_alpha"],
            linewidth=s["trend_width"], label=f"Trend ({z[0]:+.3f}/yr)")


def fig_single(df, station, index, s):
    d = df[df["station"] == station].sort_values("year")
    years, vals = d["year"].values, d[index].values.astype(float)
    fig, ax = plt.subplots(figsize=(11, s["fig_height"]))
    ax.grid(True, axis="y", alpha=s["grid_alpha"])
    if s["show_fill"]:
        ax.fill_between(years, vals, alpha=s["fill_alpha"], color=s["fill_color"])
    ax.plot(years, vals, color=s["line_color"], linewidth=s["line_width"],
            linestyle=s["linestyle"], marker=s["marker"],
            markersize=s["marker_size"], label=station)
    if s["show_trend"]:
        _trend_line(ax, years, vals, s)
        ax.legend(fontsize=9, framealpha=0.2)
    ax.set_title(f"{station}  —  {index}", fontsize=13)
    ax.set_xlabel("Year"); ax.set_ylabel(f"{index} ({INDEX_META[index]['unit']})")
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    fig.tight_layout()
    return fig


def fig_multi(df, stations, index, s):
    fig, ax = plt.subplots(figsize=(11, s["fig_height"]))
    ax.grid(True, axis="y", alpha=s["grid_alpha"])
    for i, st in enumerate(stations):
        d = df[df["station"] == st].sort_values("year")
        clr = PALETTE[i % len(PALETTE)]
        ax.plot(d["year"], d[index], color=clr, linewidth=s["line_width"],
                linestyle=s["linestyle"], marker=s["marker"],
                markersize=s["marker_size"], label=st)
        if s["show_fill"]:
            ax.fill_between(d["year"], d[index], alpha=s["fill_alpha"] * 0.7, color=clr)
    ax.set_title(f"{index}  —  all selected stations", fontsize=13)
    ax.set_xlabel("Year"); ax.set_ylabel(f"{index} ({INDEX_META[index]['unit']})")
    ax.legend(fontsize=9, framealpha=0.2)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    fig.tight_layout()
    return fig


def fig_trend_heatmap(trend_df, indices, stations, sig_alpha, s):
    pivot = trend_df.pivot_table(index="index", columns="station", values="slope")
    pivot = pivot.reindex(index=[i for i in indices if i in pivot.index],
                          columns=[st for st in stations if st in pivot.columns])
    nr, nc = pivot.shape
    fig, ax = plt.subplots(figsize=(max(6, nc * 1.6), nr * 0.75 + 1.4))
    vmax = np.nanmax(np.abs(pivot.values)) if pivot.size else 1
    im = ax.imshow(pivot.values, cmap=s["colormap"], vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(nc)); ax.set_xticklabels(pivot.columns, fontsize=10)
    ax.set_yticks(range(nr)); ax.set_yticklabels(pivot.index, fontsize=10)
    for i in range(nr):
        for j in range(nc):
            val = pivot.values[i, j]
            if np.isnan(val):
                continue
            row = trend_df[(trend_df["index"] == pivot.index[i]) &
                           (trend_df["station"] == pivot.columns[j])]
            sig = not row.empty and not np.isnan(row["p_value"].values[0]) \
                  and row["p_value"].values[0] < sig_alpha
            ax.text(j, i, f"{val:+.2f}", ha="center", va="center", fontsize=9,
                    color="#fff" if abs(val) > vmax * 0.4 else "#90cdf4",
                    fontweight="bold" if sig else "normal")
    plt.colorbar(im, ax=ax, label="Theil-Sen slope (unit/yr)", shrink=0.8)
    ax.set_title(f"Trend slopes  (bold = significant at α={sig_alpha})", fontsize=12)
    fig.tight_layout()
    return fig


def fig_index_heatmap(df, station, indices, s):
    d = df[df["station"] == station].sort_values("year")
    pivot = d.set_index("year")[indices].T.astype(float)
    norm  = pivot.copy()
    for idx in indices:
        row = pivot.loc[idx]; rng = row.max() - row.min()
        norm.loc[idx] = (row - row.min()) / rng if rng > 0 else 0.5
    nr, nc = norm.shape
    fig, ax = plt.subplots(figsize=(max(9, nc * 0.32), nr * 0.72 + 1.2))
    im = ax.imshow(norm.values, cmap=s["colormap"], aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(nc)); ax.set_xticklabels(pivot.columns, rotation=90, fontsize=8)
    ax.set_yticks(range(nr)); ax.set_yticklabels(indices, fontsize=10)
    ax.set_title(f"{station}  —  normalised index heatmap", fontsize=12)
    plt.colorbar(im, ax=ax, label="0 = min · 1 = max", shrink=0.8)
    fig.tight_layout()
    return fig


def fig_sparklines(df, station, indices, category, s):
    d = df[df["station"] == station].sort_values("year")
    clr = CATEGORY_COLORS.get(category, s["line_color"])
    fig, axes = plt.subplots(len(indices), 1, figsize=(4, len(indices) * 1.6), sharex=True)
    if len(indices) == 1:
        axes = [axes]
    for ax, idx in zip(axes, indices):
        row = d[["year", idx]].dropna()
        if s["show_fill"]:
            ax.fill_between(row["year"], row[idx], alpha=s["fill_alpha"] * 1.5, color=clr)
        ax.plot(row["year"], row[idx], color=clr,
                linewidth=s["line_width"] * 0.8,
                marker=s["marker"], markersize=s["marker_size"] * 0.7)
        ax.set_ylabel(idx, fontsize=8)
        ax.grid(True, axis="y", alpha=s["grid_alpha"])
    axes[-1].set_xlabel("Year", fontsize=8)
    fig.tight_layout(pad=0.5)
    return fig

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown(
        "<div style='font-family:JetBrains Mono,monospace;color:#5eb8ff;"
        "font-size:1.1rem;font-weight:700;letter-spacing:-1px;margin-bottom:4px;'>"
        "🌧 Extreme Indices</div>"
        "<div style='color:#4a6080;font-size:10px;letter-spacing:1px;"
        "text-transform:uppercase;margin-bottom:16px;'>ETCCDI · Dam Management</div>",
        unsafe_allow_html=True,
    )

    uploaded = st.file_uploader("Upload rainfall CSV", type=["csv"])

    with st.expander("⚙️ Baseline period", expanded=True):
        bc1, bc2 = st.columns(2)
        b_start = bc1.number_input("Start", 1900, 2100, 1981, 1)
        b_end   = bc2.number_input("End",   1900, 2100, 2010, 1)

    st.markdown("---")
    st.markdown(
        "<div style='color:#4a7fa5;font-size:10px;text-transform:uppercase;"
        "letter-spacing:1.2px;font-family:JetBrains Mono,monospace;"
        "margin-bottom:8px;'>Index categories</div>",
        unsafe_allow_html=True,
    )
    vis_cats = []
    if st.checkbox("🔵 Flood / Overtopping", True):  vis_cats.append("Flood")
    if st.checkbox("🟠 Sedimentation",        True):  vis_cats.append("Sediment")
    if st.checkbox("🟢 Storage Reliability",  True):  vis_cats.append("Storage")

    vis_indices = [k for k, v in INDEX_META.items() if v["category"] in vis_cats]

    st.markdown("---")
    sig_alpha = st.slider("Trend significance (α)", 0.01, 0.20, 0.05, 0.01, format="%.2f")

# ══════════════════════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("# Extreme Precipitation Indices")
st.markdown(
    "<div style='color:#4a6080;font-size:13px;margin-bottom:20px;'>"
    "ETCCDI climate indices · dam & reservoir management · upload your station CSV to begin"
    "</div>",
    unsafe_allow_html=True,
)

# ══════════════════════════════════════════════════════════════════════════════
# LANDING PAGE
# ══════════════════════════════════════════════════════════════════════════════

if uploaded is None:
    st.info("👈  Upload a CSV in the sidebar to get started.")
    with st.expander("Expected CSV format"):
        st.code("Date,Station_A,Station_B\n01/01/1981,2.3,0.0\n02/01/1981,0.0,1.2", language="text")
    ref = pd.DataFrame([
        {"Index": k, "Unit": v["unit"], "Category": v["category"], "Description": v["desc"]}
        for k, v in INDEX_META.items()
    ])
    st.markdown("### Index reference")
    st.dataframe(ref, use_container_width=True, hide_index=True)
    st.stop()

# ══════════════════════════════════════════════════════════════════════════════
# LOAD DATA
# ══════════════════════════════════════════════════════════════════════════════

df_all = load_and_compute(uploaded.read(), b_start, b_end)
all_stations = sorted(df_all["station"].unique())
yr_min, yr_max = int(df_all["year"].min()), int(df_all["year"].max())

# ── Filter controls (below header, above tabs) ─────────────────────────────────
fc1, fc2, fc3 = st.columns([2, 2, 2])
selected_stations = fc1.multiselect("Stations", all_stations, default=all_stations[:min(4, len(all_stations))])
year_range        = fc2.slider("Year range", yr_min, yr_max, (yr_min, yr_max))
view_mode         = fc3.selectbox("View mode (Time Series)", ["Single station", "Multi-station overlay"])

if not selected_stations:
    st.warning("Select at least one station.")
    st.stop()

df = df_all[
    df_all["station"].isin(selected_stations) &
    df_all["year"].between(*year_range)
].copy()

# ══════════════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════════════

t1, t2, t3, t4 = st.tabs(["📈  Time Series", "📊  Trend Analysis", "🟥  Heatmap", "🗂  Raw Data"])

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 – Time Series
# ─────────────────────────────────────────────────────────────────────────────
with t1:
    if not vis_indices:
        st.warning("No index category selected."); st.stop()

    tc1, tc2 = st.columns([2, 2])
    sel_index = tc1.selectbox(
        "Index", vis_indices,
        format_func=lambda x: f"{x}  ·  {INDEX_META[x]['desc']}",
        key="ts_idx",
    )
    if view_mode == "Single station" and len(selected_stations) > 1:
        station = tc2.selectbox("Station", selected_stations, key="ts_st")
    else:
        station = selected_stations[0]

    meta = INDEX_META[sel_index]
    cat_clr = CATEGORY_COLORS[meta["category"]]
    st.markdown(
        f"<div class='index-desc'>{meta['desc']} · unit: <b>{meta['unit']}</b> · "
        f"<span style='color:{cat_clr}'>{meta['category']}</span></div>",
        unsafe_allow_html=True,
    )

    # style controls
    s = plot_controls("ts", default_color=cat_clr)

    # metrics
    mcols = st.columns(len(selected_stations))
    for i, st_name in enumerate(selected_stations):
        vals = df[df["station"] == st_name][sel_index].dropna()
        if len(vals):
            mcols[i].metric(st_name, f"{vals.mean():.2f} {meta['unit']}", f"max {vals.max():.1f}")

    # plot
    if view_mode == "Single station":
        fig = fig_single(df, station, sel_index, s)
    else:
        fig = fig_multi(df, selected_stations, sel_index, s)

    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 – Trend Analysis
# ─────────────────────────────────────────────────────────────────────────────
with t2:
    if not TREND_AVAILABLE:
        st.error("Install `pymannkendall` and `scipy`:\n```\npip install pymannkendall scipy\n```")
        st.stop()
    if not vis_indices:
        st.warning("No index category selected."); st.stop()

    s_tr = plot_controls("tr", show_colormap=True)

    # compute trends
    trend_rows = []
    for st_name in selected_stations:
        d = df[df["station"] == st_name]
        for idx in vis_indices:
            p, slope, direction = compute_trend(d[idx])
            trend_rows.append({"station": st_name, "index": idx,
                                "category": INDEX_META[idx]["category"],
                                "p_value": p, "slope": slope, "direction": direction})
    trend_df = pd.DataFrame(trend_rows)

    fig = fig_trend_heatmap(trend_df, vis_indices, selected_stations, sig_alpha, s_tr)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    with st.expander("📋  Full trend table"):
        disp = trend_df.rename(columns={"p_value":"p-value","slope":"Slope (unit/yr)","direction":"Direction"})
        def _style(row):
            sig = not np.isnan(row["p-value"]) and row["p-value"] < sig_alpha
            clr = "#1a3d2b" if sig and row["Direction"]=="↑" else "#3d1a1a" if sig and row["Direction"]=="↓" else ""
            return [f"background-color:{clr}" if clr else ""]*len(row)
        st.dataframe(
            disp.style.apply(_style, axis=1)
                      .format({"p-value":"{:.4f}","Slope (unit/yr)":"{:+.4f}"}),
            use_container_width=True, hide_index=True,
        )
        st.download_button("⬇  Download trend CSV",
                           trend_df.to_csv(index=False).encode(),
                           "trend_results.csv", "text/csv")

# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 – Heatmap
# ─────────────────────────────────────────────────────────────────────────────
with t3:
    if not vis_indices:
        st.warning("No index category selected."); st.stop()

    hc1, hc2 = st.columns([2, 3])
    heat_station  = hc1.selectbox("Station", selected_stations, key="hm_st")
    heat_indices  = hc2.multiselect("Indices", vis_indices,
                                    default=vis_indices[:min(6, len(vis_indices))],
                                    key="hm_idx")

    if not heat_indices:
        st.warning("Select at least one index."); st.stop()

    s_hm = plot_controls("hm", show_colormap=True)

    fig = fig_index_heatmap(df, heat_station, heat_indices, s_hm)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    # category sparkline panels
    st.markdown("### Category panels")
    panel_cats = [c for c in vis_cats if c in CATEGORY_COLORS]
    pcols = st.columns(len(panel_cats)) if panel_cats else []
    for col_ui, cat in zip(pcols, panel_cats):
        cat_idx = [i for i in CATEGORY_INDICES.get(cat, []) if i in heat_indices]
        with col_ui:
            st.markdown(
                f"<div style='color:{CATEGORY_COLORS[cat]};font-family:JetBrains Mono,monospace;"
                f"font-size:11px;border-bottom:1px solid {CATEGORY_COLORS[cat]};"
                f"padding-bottom:4px;margin-bottom:8px;'>▪ {cat}</div>",
                unsafe_allow_html=True,
            )
            if cat_idx:
                spark = fig_sparklines(df, heat_station, cat_idx, cat, s_hm)
                st.pyplot(spark, use_container_width=True)
                plt.close(spark)
            else:
                st.caption("None selected.")

# ─────────────────────────────────────────────────────────────────────────────
# TAB 4 – Raw Data
# ─────────────────────────────────────────────────────────────────────────────
with t4:
    cols = ["station", "year"] + [c for c in vis_indices if c in df.columns]
    disp = df[cols].sort_values(["station", "year"]).reset_index(drop=True)
    st.dataframe(disp, use_container_width=True, height=480)

    st.download_button("⬇  Download filtered CSV",
                       disp.to_csv(index=False).encode(),
                       "extreme_indices_filtered.csv", "text/csv")

    with st.expander("📊  Summary statistics"):
        st.dataframe(
            disp.drop(columns=["station"]).groupby(disp["station"]).describe().T,
            use_container_width=True,
        )
