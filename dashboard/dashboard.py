import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="GreenLens — ESG Greenwashing Detector",
    page_icon="☘️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CUSTOM CSS
# ============================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@300;400;500;600&display=swap');

    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

    .stApp { background-color: #0e1a12; color: #e8f0e9; }

    [data-testid="stSidebar"] {
        background-color: #142019;
        border-right: 1px solid #2a3d2e;
    }
    [data-testid="metric-container"] {
        background-color: #1a2e1e;
        border: 1px solid #2a3d2e;
        border-radius: 12px;
        padding: 16px;
    }
    h1, h2, h3 { font-family: 'DM Serif Display', serif; color: #a8d5a2; }

    [data-testid="stDataFrame"] { border-radius: 12px; overflow: hidden; }

    .stSelectbox > div > div {
        background-color: #1a2e1e;
        border-color: #2a3d2e;
        color: #e8f0e9;
    }
    .badge-high    { background:#4a1515; color:#ff6b6b; padding:4px 12px; border-radius:20px; font-weight:600; font-size:0.85rem; border:1px solid #ff6b6b40; }
    .badge-medium  { background:#3d3015; color:#ffd166; padding:4px 12px; border-radius:20px; font-weight:600; font-size:0.85rem; border:1px solid #ffd16640; }
    .badge-low     { background:#153d20; color:#06d6a0; padding:4px 12px; border-radius:20px; font-weight:600; font-size:0.85rem; border:1px solid #06d6a040; }
    .badge-credible{ background:#152a3d; color:#74b9ff; padding:4px 12px; border-radius:20px; font-weight:600; font-size:0.85rem; border:1px solid #74b9ff40; }

    .info-box {
        background: #1a2e1e;
        border: 1px solid #2a3d2e;
        border-left: 3px solid #a8d5a2;
        border-radius: 8px;
        padding: 16px 20px;
        margin: 12px 0;
        font-size: 0.92rem;
        color: #c8e6c9;
    }
    .score-card { background:#1a2e1e; border:1px solid #2a3d2e; border-radius:12px; padding:20px; text-align:center; }
    .score-label { font-size:0.8rem; color:#7a9e7e; text-transform:uppercase; letter-spacing:1px; margin-bottom:8px; }
    .score-value { font-family:'DM Serif Display',serif; font-size:2.4rem; color:#a8d5a2; }
    .rec-box {
        background: #111f15;
        border: 1px solid #2a3d2e;
        border-radius: 10px;
        padding: 14px 18px;
        margin: 8px 0;
        font-size: 0.88rem;
        color: #c8e6c9;
        line-height: 1.6;
    }
    hr { border-color: #2a3d2e; }
</style>
""", unsafe_allow_html=True)

# ============================================================
# LOAD DATA
# ============================================================
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("final_greenwashing_report.csv")
        return df
    except FileNotFoundError:
        st.error("⚠️ Could not find final_greenwashing_report.csv — make sure it's in the same folder as this script.")
        st.stop()

df = load_data()

# --- Normalise risk level labels (handles emoji encoding issues) ---
def clean_risk(risk):
    r = str(risk)
    if "HIGH"   in r: return "🔴 HIGH RISK"
    if "MEDIUM" in r: return "🟡 MEDIUM RISK"
    if "LOW"    in r: return "🟢 LOW RISK"
    return "✅ CREDIBLE"

df["risk_level"] = df["risk_level"].apply(clean_risk)

# --- Ensure bord_bia_verified is boolean-friendly ---
df["bord_bia_verified"] = df["bord_bia_verified"].astype(str).str.lower().isin(["true", "1", "yes"])

# ============================================================
# SHARED CHART THEME
# ============================================================
CHART_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font_color="#e8f0e9",
    margin=dict(t=20, b=20)
)

COLOR_MAP = {
    "🔴 HIGH RISK"  : "#ff6b6b",
    "🟡 MEDIUM RISK": "#ffd166",
    "🟢 LOW RISK"   : "#06d6a0",
    "✅ CREDIBLE"   : "#74b9ff"
}

# ============================================================
# SIDEBAR
# ============================================================
st.sidebar.markdown("""
<div style='text-align:center; padding:20px 0'>
    <div style='font-family:DM Serif Display,serif; font-size:1.6rem; color:#a8d5a2'>☘️ GreenLens</div>
    <div style='font-size:0.75rem; color:#7a9e7e; margin-top:4px'>ESG Greenwashing Detector</div>
    <div style='font-size:0.7rem; color:#4a6e4e; margin-top:2px'>Irish Food SMEs Pilot • 2025</div>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigate",
    ["📊 Overview", "🔍 Company Deep Dive", "📈 Sector Analysis", "💡 Recommendations"],
    label_visibility="collapsed"
)

st.sidebar.markdown("---")
st.sidebar.markdown("""
<div style='font-size:0.75rem; color:#4a6e4e; padding:8px'>
    <b style='color:#7a9e7e'>How it works</b><br><br>
    🧠 <b>Talk Score</b> — Hybrid: LLM quality audit + FinBERT claim sentiment<br><br>
    📰 <b>Walk Score</b> — FinBERT on news headlines + article text<br><br>
    📏 <b>Gap Score</b> — Divergence between claims and public reality<br><br>
    ✅ <b>Bord Bia</b> — Independent Origin Green verification status
</div>
""", unsafe_allow_html=True)

# ============================================================
# HELPER FUNCTIONS
# ============================================================
def risk_badge(risk):
    if "HIGH"   in risk: return '<span class="badge-high">🔴 HIGH RISK</span>'
    if "MEDIUM" in risk: return '<span class="badge-medium">🟡 MEDIUM RISK</span>'
    if "LOW"    in risk: return '<span class="badge-low">🟢 LOW RISK</span>'
    return '<span class="badge-credible">✅ CREDIBLE</span>'


def make_gauge(value, title, color):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={"text": title, "font": {"color": "#e8f0e9", "size": 14}},
        number={"font": {"color": "#e8f0e9", "size": 28}},
        gauge={
            "axis": {"range": [0, 10], "tickcolor": "#4a6e4e"},
            "bar":  {"color": color},
            "bgcolor": "#1a2e1e",
            "bordercolor": "#2a3d2e",
            "steps": [
                {"range": [0,   3.3], "color": "#142019"},
                {"range": [3.3, 6.6], "color": "#1a2e1e"},
                {"range": [6.6, 10],  "color": "#1f3824"}
            ]
        }
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#e8f0e9",
        height=220,
        margin=dict(t=30, b=10, l=20, r=20)
    )
    return fig


def safe_get(row, col, default=0.0):
    """Safely get a column value — returns default if column missing or NaN."""
    try:
        val = row[col]
        return float(val) if pd.notna(val) else default
    except Exception:
        return default


# ============================================================
# PAGE 1: OVERVIEW
# ============================================================
if page == "📊 Overview":

    st.markdown("# ☘️ GreenLens — ESG Risk Overview")
    st.markdown("*Greenwashing detection across Irish Food SMEs using LLM + FinBERT hybrid pipeline*")
    st.markdown("---")

    # --- Top metrics ---
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Companies Analysed", len(df))
    c2.metric("🔴 High Risk",    (df["risk_level"] == "🔴 HIGH RISK").sum())
    c3.metric("🟡 Medium Risk",  (df["risk_level"] == "🟡 MEDIUM RISK").sum())
    c4.metric("✅ Bord Bia Verified", df["bord_bia_verified"].sum())
    c5.metric("Avg Gap Score",   f"{df['gap_score'].mean():.3f}")

    st.markdown("---")

    col_left, col_right = st.columns(2)

    # --- Risk distribution donut ---
    with col_left:
        st.markdown("### Risk Distribution")
        risk_counts = df["risk_level"].value_counts().reset_index()
        risk_counts.columns = ["Risk Level", "Count"]
        fig_pie = px.pie(
            risk_counts, values="Count", names="Risk Level",
            color="Risk Level", color_discrete_map=COLOR_MAP, hole=0.5
        )
        fig_pie.update_layout(**CHART_LAYOUT, legend=dict(font=dict(color="#e8f0e9")))
        fig_pie.update_traces(textfont_color="#0e1a12", textfont_size=13)
        st.plotly_chart(fig_pie, use_container_width=True)

    # --- Verified vs Not Verified bar chart ---
    with col_right:
        st.markdown("### Verified vs Not Verified")
        group_stats = df.groupby("group").agg(
            avg_talk=("talk_score",     "mean"),
            avg_walk=("walk_score_norm","mean"),
            avg_gap =("gap_score",      "mean")
        ).reset_index()

        fig_bar = go.Figure()
        for col_name, color, label in [
            ("avg_talk", "#a8d5a2", "Avg Talk Score"),
            ("avg_walk", "#74b9ff", "Avg Walk Score"),
            ("avg_gap",  "#ff6b6b", "Avg Gap Score"),
        ]:
            fig_bar.add_trace(go.Bar(
                name=label,
                x=group_stats["group"],
                y=group_stats[col_name].round(3),
                marker_color=color,
                text=group_stats[col_name].round(3),
                textposition="outside"
            ))
        fig_bar.update_layout(
            **CHART_LAYOUT,
            barmode="group",
            legend=dict(font=dict(color="#e8f0e9")),
            xaxis=dict(gridcolor="#2a3d2e"),
            yaxis=dict(gridcolor="#2a3d2e")
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    # --- Full risk table ---
    st.markdown("### All Companies — Risk Table")
    st.markdown("""
    <div class='info-box'>
    Sorted by Gap Score (highest risk first).
    <b>Talk Score</b> = Hybrid LLM + FinBERT claims score.
    <b>Walk Score</b> = FinBERT news sentiment.
    <b>Gap Score</b> = relative divergence — higher means more greenwashing risk.
    </div>
    """, unsafe_allow_html=True)

    display_cols = ["company_name", "sector", "group", "bord_bia_verified",
                    "talk_score", "walk_score_norm", "gap_score", "risk_level"]
    display_cols = [c for c in display_cols if c in df.columns]
    df_display   = df[display_cols].copy()
    df_display.columns = [c.replace("_", " ").title() for c in display_cols]

    def colour_risk(val):
        if "HIGH"   in str(val): return "background-color:#4a1515; color:#ff6b6b"
        if "MEDIUM" in str(val): return "background-color:#3d3015; color:#ffd166"
        if "LOW"    in str(val): return "background-color:#153d20; color:#06d6a0"
        return "background-color:#152a3d; color:#74b9ff"

    def colour_gap(val):
        try:
            v = float(val)
            if v >= 0.40: return "color:#ff6b6b; font-weight:600"
            if v >= 0.20: return "color:#ffd166; font-weight:600"
            if v >  0:    return "color:#06d6a0"
            return "color:#74b9ff"
        except Exception:
            return ""

    fmt = {}
    if "Talk Score"      in df_display.columns: fmt["Talk Score"]      = "{:.2f}"
    if "Walk Score Norm" in df_display.columns: fmt["Walk Score Norm"] = "{:.2f}"
    if "Gap Score"       in df_display.columns: fmt["Gap Score"]       = "{:.3f}"

    styled = df_display.style
    if "Risk Level" in df_display.columns:
        styled = styled.applymap(colour_risk, subset=["Risk Level"])
    if "Gap Score"  in df_display.columns:
        styled = styled.applymap(colour_gap,  subset=["Gap Score"])
    styled = styled.format(fmt)

    st.dataframe(styled, use_container_width=True, height=600)


# ============================================================
# PAGE 2: COMPANY DEEP DIVE
# ============================================================
elif page == "🔍 Company Deep Dive":

    st.markdown("# 🔍 Company Deep Dive")
    st.markdown("*Select any company to see its full ESG risk profile*")
    st.markdown("---")

    company_list = df.sort_values("gap_score", ascending=False)["company_name"].tolist()
    selected     = st.selectbox("Select a company", company_list)
    row          = df[df["company_name"] == selected].iloc[0]

    st.markdown("---")

    # --- Header row ---
    c1, c2, c3 = st.columns([2, 1, 1])
    with c1:
        st.markdown(f"## {selected}")
        sector = row.get("sector", "N/A") if "sector" in df.columns else "N/A"
        st.markdown(f"**Sector:** {sector} &nbsp;|&nbsp; **Group:** {row['group']}")
        bord_status = "✅ Bord Bia Origin Green Verified" if row["bord_bia_verified"] else "❌ Not Bord Bia Verified"
        st.markdown(f"**Verification:** {bord_status}")
    with c2:
        st.markdown(f"<div class='score-card'><div class='score-label'>Risk Level</div><div style='margin-top:8px'>{risk_badge(row['risk_level'])}</div></div>", unsafe_allow_html=True)
    with c3:
        st.markdown(f"<div class='score-card'><div class='score-label'>Gap Score</div><div class='score-value'>{row['gap_score']:.3f}</div></div>", unsafe_allow_html=True)

    st.markdown("---")

    # --- Three score gauges ---
    ct, cw, cg = st.columns(3)
    with ct:
        st.markdown("#### 🧠 Talk Score")
        st.markdown("<div class='info-box'>Hybrid: LLM claim quality (60%) + FinBERT claim sentiment (40%)</div>", unsafe_allow_html=True)
        st.plotly_chart(make_gauge(safe_get(row, "talk_score"), "Talk Score / 10", "#a8d5a2"), use_container_width=True)
    with cw:
        st.markdown("#### 📰 Walk Score")
        st.markdown("<div class='info-box'>FinBERT sentiment on Google News headlines + article text</div>", unsafe_allow_html=True)
        st.plotly_chart(make_gauge(safe_get(row, "walk_score_norm"), "Walk Score / 10", "#74b9ff"), use_container_width=True)
    with cg:
        st.markdown("#### 📏 Gap Score")
        st.markdown("<div class='info-box'>Divergence between claims and reality (0 = none, 1 = maximum)</div>", unsafe_allow_html=True)
        gap_color = "#ff6b6b" if row["gap_score"] >= 0.40 else "#ffd166" if row["gap_score"] >= 0.20 else "#06d6a0"
        st.plotly_chart(make_gauge(row["gap_score"] * 10, "Gap Score × 10", gap_color), use_container_width=True)

    st.markdown("---")

    # --- Talk Score breakdown ---
    st.markdown("### 🧠 Talk Score Breakdown")
    ta, tb = st.columns(2)

    with ta:
        st.markdown("**LLM Audit Sub-scores**")
        spec = safe_get(row, "claim_specificity", 0)
        cred = safe_get(row, "claim_credibility", 0)
        llm_q = safe_get(row, "llm_quality_score", 0)

        fig_llm = go.Figure(go.Bar(
            x=["Claim Specificity", "Claim Credibility", "LLM Quality"],
            y=[spec, cred, llm_q],
            marker_color=["#a8d5a2", "#74b9ff", "#ffd166"],
            text=[f"{spec:.1f}", f"{cred:.1f}", f"{llm_q:.1f}"],
            textposition="outside"
        ))
        fig_llm.update_layout(
            **CHART_LAYOUT,
            yaxis=dict(range=[0, 11], gridcolor="#2a3d2e"),
            xaxis=dict(gridcolor="#2a3d2e"),
            showlegend=False
        )
        st.plotly_chart(fig_llm, use_container_width=True)

    with tb:
        st.markdown("**LLM Auditor Notes**")
        if "auditor_note" in df.columns and pd.notna(row.get("auditor_note", "")):
            st.markdown(f"<div class='info-box'>📋 {row['auditor_note']}</div>", unsafe_allow_html=True)
        if "claims_found" in df.columns and pd.notna(row.get("claims_found", "")):
            claims = str(row["claims_found"]).split(" | ")
            claims_html = "".join([f"<li>{c}</li>" for c in claims if c.strip()])
            st.markdown(f"<div class='info-box'><b>Claims detected:</b><ul>{claims_html}</ul></div>", unsafe_allow_html=True)
        if "greenwash_signals" in df.columns and pd.notna(row.get("greenwash_signals", "")):
            sigs = str(row["greenwash_signals"]).split(" | ")
            sigs_html = "".join([f"<li style='color:#ff6b6b'>{s}</li>" for s in sigs if s.strip()])
            st.markdown(f"<div class='info-box' style='border-left-color:#ff6b6b'><b>⚠️ Greenwashing signals:</b><ul>{sigs_html}</ul></div>", unsafe_allow_html=True)

    st.markdown("---")

    # --- FinBERT Walk Score breakdown ---
    st.markdown("### 📰 Walk Score — FinBERT Sentiment Breakdown")
    cs1, cs2 = st.columns([1, 2])

    with cs1:
        # FIX: read all 3 scores directly from columns, clamp to valid range
        pos = safe_get(row, "walk_positive", 0.0)
        neg = safe_get(row, "walk_negative", 0.0)
        neu = safe_get(row, "walk_neutral",  0.0)

        # If neutral column missing, derive it from remainder
        if neu == 0.0:
            neu = max(0.0, round(1.0 - pos - neg, 4))

        # Use go.Bar with explicit marker_color — px.bar colour mapping
        # can render bars invisible against the dark background
        fig_sent = go.Figure()
        fig_sent.add_trace(go.Bar(
            x=["Positive", "Neutral", "Negative"],
            y=[pos, neu, neg],
            marker_color=["#06d6a0", "#74b9ff", "#ff6b6b"],
            text=[f"{pos:.3f}", f"{neu:.3f}", f"{neg:.3f}"],
            textposition="outside",
            textfont=dict(color="#e8f0e9", size=13),
            width=0.5
        ))
        y_max = max(0.15, pos, neu, neg) * 1.25   # headroom above tallest bar
        fig_sent.update_layout(
            **CHART_LAYOUT,
            showlegend=False,
            xaxis=dict(gridcolor="#2a3d2e", tickfont=dict(color="#e8f0e9")),
            yaxis=dict(gridcolor="#2a3d2e", range=[0, y_max],
                       tickfont=dict(color="#e8f0e9"), title="Score")
        )
        st.plotly_chart(fig_sent, use_container_width=True)

    with cs2:
        st.markdown("#### 📋 ESG Risk Interpretation")
        gap = row["gap_score"]

        if gap >= 0.40:
            st.markdown("""
            <div class='info-box' style='border-left-color:#ff6b6b'>
            🔴 <b>HIGH GREENWASHING RISK</b><br><br>
            This company's ESG claims significantly exceed what public news sentiment supports.
            Independent verification via Bord Bia Origin Green is absent.
            Warrants closer scrutiny before any ESG-related investment or partnership.
            </div>
            """, unsafe_allow_html=True)
        elif gap >= 0.20:
            st.markdown("""
            <div class='info-box' style='border-left-color:#ffd166'>
            🟡 <b>MEDIUM GREENWASHING RISK</b><br><br>
            A noticeable gap exists between ESG claims and public sentiment data.
            Caution advised — further due diligence recommended before ESG-linked commitments.
            </div>
            """, unsafe_allow_html=True)
        elif row["bord_bia_verified"]:
            st.markdown("""
            <div class='info-box' style='border-left-color:#06d6a0'>
            ✅ <b>CREDIBLE ESG PROFILE</b><br><br>
            Independently verified by Bord Bia Origin Green and public sentiment
            aligns with ESG claims. Low greenwashing risk.
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class='info-box' style='border-left-color:#74b9ff'>
            🟢 <b>LOW RISK — UNDER-CLAIMING</b><br><br>
            Public reputation exceeds stated ESG claims — a sign of genuine sustainability
            practice without marketing hype. Strong credibility signals despite no Bord Bia verification.
            </div>
            """, unsafe_allow_html=True)

        st.markdown("#### 🔢 Score Summary")
        articles = int(safe_get(row, "articles_used", 0))
        st.markdown(f"""
        | Metric | Value |
        |--------|-------|
        | LLM Quality Score | {safe_get(row, 'llm_quality_score'):.2f} / 10 |
        | FinBERT Claim Sentiment | {safe_get(row, 'claim_sent_score'):.2f} / 10 |
        | Talk Score (Hybrid, normalised) | {safe_get(row, 'talk_score'):.2f} / 10 |
        | Walk Score (normalised) | {safe_get(row, 'walk_score_norm'):.2f} / 10 |
        | Gap Score | {row['gap_score']:.3f} |
        | News Articles Analysed | {articles} |
        | Bord Bia Verified | {'Yes ✅' if row['bord_bia_verified'] else 'No ❌'} |
        """)


# ============================================================
# PAGE 3: SECTOR ANALYSIS
# ============================================================
elif page == "📈 Sector Analysis":

    st.markdown("# 📈 Sector Analysis")
    st.markdown("*How do greenwashing risk levels compare across Irish food sectors?*")
    st.markdown("---")

    # --- Guard: check sector column exists and is populated ---
    if "sector" not in df.columns or df["sector"].isna().all():
        st.warning("⚠️ No sector data found. Make sure your `combined_companies.csv` has a `sector` column and re-run the notebook.")
        st.stop()

    sector_stats = df.groupby("sector").agg(
        companies    =("company_name",    "count"),
        avg_talk     =("talk_score",      "mean"),
        avg_walk     =("walk_score_norm", "mean"),
        avg_gap      =("gap_score",       "mean"),
        high_risk    =("risk_level",      lambda x: (x == "🔴 HIGH RISK").sum()),
        verified_pct =("bord_bia_verified","mean")
    ).round(3).sort_values("avg_gap", ascending=False).reset_index()

    sector_stats["verified_pct"] = (sector_stats["verified_pct"] * 100).round(0).astype(int)

    # --- Average Gap Score by Sector bar chart ---
    st.markdown("### Average Gap Score by Sector")
    st.markdown("""
    <div class='info-box'>
    Each bar is the average Gap Score across all companies in that sector.
    Higher bars = more greenwashing risk on average in that sector.
    </div>
    """, unsafe_allow_html=True)

    fig_sector = px.bar(
        sector_stats, x="sector", y="avg_gap",
        color="avg_gap",
        color_continuous_scale=["#06d6a0", "#ffd166", "#ff6b6b"],
        text=sector_stats["avg_gap"].round(3),
        labels={"avg_gap": "Avg Gap Score", "sector": "Sector"},
        custom_data=["companies", "high_risk", "verified_pct"]
    )
    fig_sector.update_traces(
        textposition="outside",
        textfont_color="#e8f0e9",
        hovertemplate=(
            "<b>%{x}</b><br>"
            "Avg Gap Score: %{y:.3f}<br>"
            "Companies: %{customdata[0]}<br>"
            "High Risk: %{customdata[1]}<br>"
            "Bord Bia Verified: %{customdata[2]}%"
        )
    )
    fig_sector.update_layout(
        **CHART_LAYOUT,
        coloraxis_showscale=False,
        xaxis=dict(gridcolor="#2a3d2e"),
        yaxis=dict(gridcolor="#2a3d2e")
    )
    st.plotly_chart(fig_sector, use_container_width=True)

    st.markdown("---")

    # --- Talk vs Walk by Sector grouped bar ---
    st.markdown("### Talk vs Walk Score by Sector")
    fig_tw = go.Figure()
    fig_tw.add_trace(go.Bar(
        name="Avg Talk Score",
        x=sector_stats["sector"],
        y=sector_stats["avg_talk"].round(2),
        marker_color="#a8d5a2",
        text=sector_stats["avg_talk"].round(2),
        textposition="outside"
    ))
    fig_tw.add_trace(go.Bar(
        name="Avg Walk Score",
        x=sector_stats["sector"],
        y=sector_stats["avg_walk"].round(2),
        marker_color="#74b9ff",
        text=sector_stats["avg_walk"].round(2),
        textposition="outside"
    ))
    fig_tw.update_layout(
        **CHART_LAYOUT,
        barmode="group",
        legend=dict(font=dict(color="#e8f0e9")),
        xaxis=dict(gridcolor="#2a3d2e"),
        yaxis=dict(gridcolor="#2a3d2e", range=[0, 13])
    )
    st.plotly_chart(fig_tw, use_container_width=True)

    st.markdown("---")

    # --- Sector stats table ---
    st.markdown("### Sector Summary Table")
    st.dataframe(
        sector_stats.rename(columns={
            "sector":       "Sector",
            "companies":    "Companies",
            "avg_talk":     "Avg Talk",
            "avg_walk":     "Avg Walk",
            "avg_gap":      "Avg Gap",
            "high_risk":    "High Risk",
            "verified_pct": "Bord Bia Verified %"
        }),
        use_container_width=True
    )

    st.markdown("---")

    # --- Talk vs Walk scatter (greenwashing map) ---
    st.markdown("### Talk vs Walk — The Greenwashing Map")
    st.markdown("""
    <div class='info-box'>
    Each dot is a company. <b style='color:#ff6b6b'>Top-left</b> (high talk, low walk) = highest greenwashing risk.
    <b style='color:#06d6a0'>Bottom-right</b> (low talk, high walk) = most credible.
    Shape = sector. Colour = risk level.
    </div>
    """, unsafe_allow_html=True)

    # Use go.Scatter per risk group with hardcoded marker colours
    # px.scatter colour mapping is unreliable on dark backgrounds
    fig_scatter = go.Figure()

    for risk_label, hex_color in [
        ("🔴 HIGH RISK",   "#ff6b6b"),
        ("🟡 MEDIUM RISK", "#ffd166"),
        ("🟢 LOW RISK",    "#06d6a0"),
        ("✅ CREDIBLE",    "#74b9ff"),
    ]:
        subset = df[df["risk_level"] == risk_label]
        if subset.empty:
            continue
        fig_scatter.add_trace(go.Scatter(
            x=subset["walk_score_norm"],
            y=subset["talk_score"],
            mode="markers+text",
            name=risk_label,
            text=subset["company_name"],
            textposition="top center",
            textfont=dict(size=9, color="#c8e6c9"),
            marker=dict(
                color=hex_color,
                size=12,
                line=dict(color="#0e1a12", width=1)
            ),
            hovertemplate=(
                "<b>%{text}</b><br>"
                "Talk: %{y:.2f}<br>"
                "Walk: %{x:.2f}<extra></extra>"
            )
        ))

    fig_scatter.add_hline(y=5, line_dash="dot", line_color="#3a5a3e", line_width=1)
    fig_scatter.add_vline(x=5, line_dash="dot", line_color="#3a5a3e", line_width=1)
    fig_scatter.add_annotation(x=1,   y=9.5, text="⚠️ High Risk Zone",
        showarrow=False, font=dict(color="#ff6b6b", size=11))
    fig_scatter.add_annotation(x=8.5, y=0.5, text="✅ Credible Zone",
        showarrow=False, font=dict(color="#06d6a0", size=11))
    fig_scatter.update_layout(
        **CHART_LAYOUT,
        xaxis=dict(
            title="Walk Score (Public Sentiment) →",
            gridcolor="#1a2e1e", range=[-0.5, 11],
            tickfont=dict(color="#e8f0e9")
        ),
        yaxis=dict(
            title="Talk Score (ESG Claims) ↑",
            gridcolor="#1a2e1e", range=[-0.5, 11],
            tickfont=dict(color="#e8f0e9")
        ),
        legend=dict(font=dict(color="#e8f0e9"), bgcolor="#1a2e1e",
                    bordercolor="#2a3d2e", borderwidth=1),
        height=560
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

    st.markdown("---")

    # --- Key findings ---
    st.markdown("### 🔑 Key Pilot Findings")
    f1, f2, f3 = st.columns(3)

    with f1:
        st.markdown("""
        <div class='info-box'>
        <b>📡 Data Coverage</b><br><br>
        Google News RSS achieved <b>100% coverage</b> across all Irish Food SMEs —
        confirming it as a reliable, ToS-compliant data source for this sector.
        </div>
        """, unsafe_allow_html=True)

    with f2:
        ver_gap   = df[df["bord_bia_verified"] == True]["gap_score"].mean()
        unver_gap = df[df["bord_bia_verified"] == False]["gap_score"].mean()
        diff_pct  = round(((unver_gap - ver_gap) / ver_gap) * 100) if ver_gap > 0 else 0
        st.markdown(f"""
        <div class='info-box'>
        <b>📏 Verification Gap</b><br><br>
        Non-verified companies show a <b>{diff_pct}% higher average gap score</b> than
        Bord Bia verified companies — supporting the validity of the detection framework.
        </div>
        """, unsafe_allow_html=True)

    with f3:
        top_company = df.loc[df["gap_score"].idxmax(), "company_name"]
        top_sector  = df.loc[df["avg_gap"].idxmax(), "sector"] if "avg_gap" in df.columns else sector_stats.iloc[0]["sector"]
        st.markdown(f"""
        <div class='info-box'>
        <b>🚩 Strongest Signal</b><br><br>
        <b>{top_company}</b> recorded the highest individual gap score.
        Sector <b>{top_sector}</b> shows the highest average gap score across all companies.
        </div>
        """, unsafe_allow_html=True)


# ============================================================
# PAGE 4: RECOMMENDATIONS
# ============================================================
elif page == "💡 Recommendations":

    st.markdown("# 💡 Recommendations")
    st.markdown("*Targeted action points for each company and sector-level systemic insights*")
    st.markdown("---")

    tab1, tab2 = st.tabs(["🏢 Per-Company", "🏭 Sector-Level"])

    # -------------------------------------------------------
    # TAB 1: PER-COMPANY RECOMMENDATIONS
    # -------------------------------------------------------
    with tab1:
        st.markdown("### Per-Company Action Points")
        st.markdown("""
        <div class='info-box'>
        Select a risk level filter to focus on the companies that need the most attention.
        Recommendations are generated from each company's Gap Score, LLM audit findings,
        FinBERT sentiment, and Bord Bia verification status.
        </div>
        """, unsafe_allow_html=True)

        risk_filter = st.selectbox(
            "Filter by risk level",
            ["All", "🔴 HIGH RISK", "🟡 MEDIUM RISK", "🟢 LOW RISK", "✅ CREDIBLE"]
        )

        df_rec = df.copy() if risk_filter == "All" else df[df["risk_level"] == risk_filter].copy()
        df_rec = df_rec.sort_values("gap_score", ascending=False)

        for _, row in df_rec.iterrows():
            gap      = row["gap_score"]
            talk     = safe_get(row, "talk_score")
            walk     = safe_get(row, "walk_score_norm")
            verified = row["bord_bia_verified"]
            signals  = str(row.get("greenwash_signals", ""))
            sector   = row.get("sector", "N/A") if "sector" in df.columns else "N/A"

            # --- Build recommendation list ---
            recs = []

            if gap >= 0.40:
                recs.append("🔴 **HIGH PRIORITY:** ESG claims significantly exceed public evidence. Immediate independent verification strongly advised before any ESG-linked investment or partnership.")
            elif gap >= 0.20:
                recs.append("🟡 **MODERATE ACTION:** Noticeable gap between claims and public sentiment. Supplement website claims with measurable data or third-party certifications.")
            elif gap >= 0.05:
                recs.append("🟢 **LOW RISK:** Minor gap detected. Consider publishing a formal sustainability report with KPIs to fully close the credibility gap.")
            else:
                recs.append("✅ **CREDIBLE PROFILE:** Public reputation matches or exceeds ESG claims. Maintain approach and consider showcasing this credibility to investors.")

            if not verified and gap >= 0.20:
                recs.append("📋 Apply for **Bord Bia Origin Green** verification to provide independent backing for ESG claims.")
            elif verified and gap >= 0.30:
                recs.append("⚠️ Despite Bord Bia verification, gap score remains elevated. Review public communications — claims may be overstated relative to actions.")

            if talk >= 7.0 and walk <= 4.0:
                recs.append("📢 High claim volume but low public sentiment — ESG messaging may not be reaching external audiences. Consider PR strategy review.")
            elif talk <= 3.0:
                recs.append("📝 Low ESG claim density on website — opportunity to better communicate existing sustainability practices to customers and investors.")

            if walk >= 7.0 and talk <= 4.0:
                recs.append("🌟 Strong public reputation but under-claiming on website — update website to reflect strong ESG track record.")

            if "vague" in signals.lower():
                recs.append("🔍 LLM flagged **vague language** — replace generic phrases with specific, measurable commitments (e.g. 'reduced emissions 30% by 2027').")
            if "no measurable" in signals.lower() or "no target" in signals.lower():
                recs.append("🎯 **No measurable targets** detected — add quantified ESG KPIs to your sustainability page.")
            if "unverified" in signals.lower() or "certification" in signals.lower():
                recs.append("🏅 **Unverified certifications** flagged — ensure all mentioned certifications are currently valid and publicly verifiable.")

            # --- Display card ---
            badge = risk_badge(row["risk_level"])
            st.markdown(f"""
            <div class='rec-box'>
                <b style='font-size:1rem; color:#a8d5a2'>{row['company_name']}</b>
                &nbsp;&nbsp;{badge}&nbsp;&nbsp;
                <span style='color:#7a9e7e; font-size:0.8rem'>Sector: {sector} | Gap: {gap:.3f} | Talk: {talk:.2f} | Walk: {walk:.2f}</span>
            </div>
            """, unsafe_allow_html=True)

            for rec in recs:
                st.markdown(f"&nbsp;&nbsp;&nbsp;→ {rec}")
            st.markdown("")

    # -------------------------------------------------------
    # TAB 2: SECTOR-LEVEL RECOMMENDATIONS
    # -------------------------------------------------------
    with tab2:
        st.markdown("### Sector-Level Insights")
        st.markdown("""
        <div class='info-box'>
        Systemic patterns across sectors — useful for regulators, industry bodies,
        and your dissertation findings section.
        </div>
        """, unsafe_allow_html=True)

        if "sector" not in df.columns or df["sector"].isna().all():
            st.warning("No sector data available. Re-run the notebook with sector column populated.")
        else:
            sector_groups = df.groupby("sector")

            for sector_name, grp in sector_groups:
                avg_gap  = grp["gap_score"].mean()
                avg_talk = safe_get(grp.iloc[0], "talk_score") if len(grp) == 1 else grp["talk_score"].mean()
                avg_walk = safe_get(grp.iloc[0], "walk_score_norm") if len(grp) == 1 else grp["walk_score_norm"].mean()
                n        = len(grp)
                verified = grp["bord_bia_verified"].sum()
                high_n   = (grp["risk_level"] == "🔴 HIGH RISK").sum()

                avg_talk = grp["talk_score"].mean()
                avg_walk = grp["walk_score_norm"].mean()

                if avg_gap >= 0.40:
                    icon, border = "🔴", "#ff6b6b"
                    priority = "HIGH RISK SECTOR"
                    action   = (f"{high_n}/{n} companies flagged as high risk. "
                                "Sector-wide audit recommended. Regulators and buyers "
                                "should apply heightened scrutiny to ESG claims here.")
                elif avg_gap >= 0.20:
                    icon, border = "🟡", "#ffd166"
                    priority = "MODERATE RISK"
                    action   = ("Meaningful gap between claims and public reputation. "
                                "Industry body (e.g. Bord Bia) could issue sector-specific "
                                "ESG disclosure guidelines to improve consistency.")
                else:
                    icon, border = "🟢", "#06d6a0"
                    priority = "LOW RISK"
                    action   = ("Sector demonstrates a credible ESG profile overall. "
                                "Good candidate for ESG best-practice showcase in "
                                "Bord Bia Origin Green programme communications.")

                extra = ""
                if avg_talk > avg_walk + 2:
                    extra = "⚠️ Claims consistently outpace public sentiment across the sector — structural claim-reality gap detected."
                elif avg_walk > avg_talk + 2:
                    extra = "💡 Sector reputation exceeds stated claims — under-claiming pattern. Marketing opportunity for sector-wide ESG communication."

                st.markdown(f"""
                <div class='info-box' style='border-left-color:{border}'>
                    <b>{icon} {sector_name}</b> — {priority}
                    &nbsp;<span style='color:#7a9e7e; font-size:0.8rem'>
                    {n} companies | {verified} Bord Bia verified | 
                    Avg Gap: {avg_gap:.3f} | Avg Talk: {avg_talk:.2f} | Avg Walk: {avg_walk:.2f}
                    </span><br><br>
                    {action}<br>
                    {"<br>" + extra if extra else ""}
                </div>
                """, unsafe_allow_html=True)