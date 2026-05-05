import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="GreenLens — ESG Greenwashing Detector",
    page_icon="🌿",
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
    [data-testid="stSidebar"] { background-color: #142019; border-right: 1px solid #2a3d2e; }
    [data-testid="metric-container"] { background-color: #1a2e1e; border: 1px solid #2a3d2e; border-radius: 12px; padding: 16px; }
    h1, h2, h3 { font-family: 'DM Serif Display', serif; color: #a8d5a2; }
    [data-testid="stDataFrame"] { border-radius: 12px; overflow: hidden; }

    .badge-high     { background:#4a1515; color:#ff6b6b; padding:4px 12px; border-radius:20px; font-weight:600; font-size:0.85rem; border:1px solid #ff6b6b40; }
    .badge-medium   { background:#3d3015; color:#ffd166; padding:4px 12px; border-radius:20px; font-weight:600; font-size:0.85rem; border:1px solid #ffd16640; }
    .badge-low      { background:#153d20; color:#06d6a0; padding:4px 12px; border-radius:20px; font-weight:600; font-size:0.85rem; border:1px solid #06d6a040; }
    .badge-credible { background:#152a3d; color:#74b9ff; padding:4px 12px; border-radius:20px; font-weight:600; font-size:0.85rem; border:1px solid #74b9ff40; }
    .badge-v1       { background:#2a1f3d; color:#c8a8ff; padding:3px 10px; border-radius:20px; font-size:0.78rem; border:1px solid #c8a8ff40; }
    .badge-v2       { background:#1f3d2a; color:#a8ffd4; padding:3px 10px; border-radius:20px; font-size:0.78rem; border:1px solid #a8ffd440; }

    .info-box { background:#1a2e1e; border:1px solid #2a3d2e; border-left:3px solid #a8d5a2; border-radius:8px; padding:16px 20px; margin:12px 0; font-size:0.92rem; color:#c8e6c9; }
    .score-card  { background:#1a2e1e; border:1px solid #2a3d2e; border-radius:12px; padding:20px; text-align:center; }
    .score-label { font-size:0.8rem; color:#7a9e7e; text-transform:uppercase; letter-spacing:1px; margin-bottom:8px; }
    .score-value { font-family:'DM Serif Display',serif; font-size:2.4rem; color:#a8d5a2; }

    .version-toggle { background:#1a2e1e; border:1px solid #2a3d2e; border-radius:8px; padding:8px 16px; margin-bottom:12px; }
    hr { border-color: #2a3d2e; }
</style>
""", unsafe_allow_html=True)

# ============================================================
# LOAD DATA
# ============================================================
@st.cache_data
def load_data():
    # V1 results
    try:
        df_v1 = pd.read_csv("final_greenwashing_report.csv")
    except FileNotFoundError:
        st.error("⚠️ final_greenwashing_report.csv not found.")
        st.stop()

    # V2 results
    try:
        df_v2 = pd.read_csv("final_report_v2.csv")
    except FileNotFoundError:
        st.error("⚠️ final_report_v2.csv not found.")
        st.stop()

    # Comparison
    try:
        df_compare = pd.read_csv("v1_vs_v2_comparison.csv")
    except FileNotFoundError:
        st.error("⚠️ v1_vs_v2_comparison.csv not found.")
        st.stop()

    # LLM audit details
    try:
        df_llm = pd.read_csv("llm_talk_scores.csv")
    except FileNotFoundError:
        df_llm = pd.DataFrame()

    return df_v1, df_v2, df_compare, df_llm

df_v1, df_v2, df_compare, df_llm = load_data()

# Clean risk labels
def clean_risk(risk):
    r = str(risk)
    if "HIGH"   in r: return "🔴 HIGH RISK"
    if "MEDIUM" in r: return "🟡 MEDIUM RISK"
    if "LOW"    in r: return "🟢 LOW RISK"
    return "✅ CREDIBLE"

for col in ["risk_level"]:
    if col in df_v1.columns:
        df_v1[col] = df_v1[col].apply(clean_risk)
for col in ["risk_level_v2"]:
    if col in df_v2.columns:
        df_v2[col] = df_v2[col].apply(clean_risk)
for col in ["risk_v1","risk_v2"]:
    if col in df_compare.columns:
        df_compare[col] = df_compare[col].apply(clean_risk)

# ============================================================
# SIDEBAR
# ============================================================
st.sidebar.markdown("""
<div style='text-align:center; padding:20px 0'>
    <div style='font-family:DM Serif Display,serif; font-size:1.6rem; color:#a8d5a2'>🌿 GreenLens</div>
    <div style='font-size:0.75rem; color:#7a9e7e; margin-top:4px'>ESG Greenwashing Detector</div>
    <div style='font-size:0.7rem; color:#4a6e4e; margin-top:2px'>Irish Food SMEs Pilot • 2025</div>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigate",
    ["📊 Overview", "🔍 Company Deep Dive", "📈 Sector Analysis", "🔬 V1 vs V2 Comparison"],
    label_visibility="collapsed"
)

st.sidebar.markdown("---")
st.sidebar.markdown("""
<div style='font-size:0.75rem; color:#4a6e4e; padding:8px'>
    <b style='color:#7a9e7e'>Pipeline</b><br><br>
    <span class='badge-v1'>V1</span> <b>Talk</b> = ESG keyword count<br><br>
    <span class='badge-v2'>V2</span> <b>Talk</b> = LLM semantic audit<br><br>
    📰 <b>Walk</b> = FinBERT news sentiment<br><br>
    📏 <b>Gap</b> = Talk minus Walk divergence<br><br>
    ✅ <b>Bord Bia</b> = Independent verification
</div>
""", unsafe_allow_html=True)

# ============================================================
# HELPERS
# ============================================================
def risk_badge(risk):
    r = str(risk)
    if "HIGH"   in r: return '<span class="badge-high">🔴 HIGH RISK</span>'
    if "MEDIUM" in r: return '<span class="badge-medium">🟡 MEDIUM RISK</span>'
    if "LOW"    in r: return '<span class="badge-low">🟢 LOW RISK</span>'
    return '<span class="badge-credible">✅ CREDIBLE</span>'

def make_gauge(value, title, color, max_val=10):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={"text": title, "font": {"color": "#e8f0e9", "size": 13}},
        number={"font": {"color": "#e8f0e9", "size": 26}},
        gauge={
            "axis": {"range": [0, max_val], "tickcolor": "#4a6e4e"},
            "bar": {"color": color},
            "bgcolor": "#1a2e1e",
            "bordercolor": "#2a3d2e",
            "steps": [
                {"range": [0, max_val/3],     "color": "#142019"},
                {"range": [max_val/3, max_val*2/3], "color": "#1a2e1e"},
                {"range": [max_val*2/3, max_val],   "color": "#1f3824"}
            ]
        }
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#e8f0e9",
        height=210,
        margin=dict(t=30, b=10, l=20, r=20)
    )
    return fig

color_map = {
    "🔴 HIGH RISK"  : "#ff6b6b",
    "🟡 MEDIUM RISK": "#ffd166",
    "🟢 LOW RISK"   : "#06d6a0",
    "✅ CREDIBLE"   : "#74b9ff"
}

# ============================================================
# PAGE 1: OVERVIEW
# ============================================================
if page == "📊 Overview":

    st.markdown("# 🌿 GreenLens — ESG Risk Overview")
    st.markdown("*Greenwashing detection across 26 Irish Food SMEs — V1 keywords vs V2 LLM pipeline*")
    st.markdown("---")

    # Version toggle
    version = st.radio(
        "Pipeline Version",
        ["V1 — Keyword Counting", "V2 — LLM Semantic Audit"],
        horizontal=True
    )
    df_active      = df_v1    if "V1" in version else df_v2
    risk_col       = "risk_level" if "V1" in version else "risk_level_v2"
    gap_col        = "gap_score"  if "V1" in version else "gap_score_v2"
    talk_col       = "talk_score" if "V1" in version else "talk_norm"
    walk_col       = "walk_score_norm" if "V1" in version else "walk_norm"
    version_label  = "V1" if "V1" in version else "V2"

    st.markdown("---")

    # Top metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Companies", len(df_active))
    with col2:
        st.metric("🔴 High Risk", (df_active[risk_col] == "🔴 HIGH RISK").sum())
    with col3:
        st.metric("🟡 Medium Risk", (df_active[risk_col] == "🟡 MEDIUM RISK").sum())
    with col4:
        st.metric("✅ Bord Bia Verified", (df_active["bord_bia_verified"] == True).sum())
    with col5:
        st.metric("Avg Gap Score", f"{df_active[gap_col].mean():.3f}")

    st.markdown("---")
    col_left, col_right = st.columns(2)

    # Risk pie
    with col_left:
        st.markdown(f"### Risk Distribution — {version_label}")
        rc = df_active[risk_col].value_counts().reset_index()
        rc.columns = ["Risk Level", "Count"]
        fig_pie = px.pie(rc, values="Count", names="Risk Level",
                         color="Risk Level", color_discrete_map=color_map, hole=0.5)
        fig_pie.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                               font_color="#e8f0e9", legend=dict(font=dict(color="#e8f0e9")),
                               margin=dict(t=20,b=20))
        fig_pie.update_traces(textfont_color="#0e1a12", textfont_size=13)
        st.plotly_chart(fig_pie, use_container_width=True)

    # Group bar chart
    with col_right:
        st.markdown(f"### Verified vs Not Verified — {version_label}")
        grp = df_active.groupby("group").agg(
            avg_talk=(talk_col, "mean"),
            avg_walk=(walk_col, "mean"),
            avg_gap=(gap_col,  "mean")
        ).reset_index()
        fig_bar = go.Figure()
        fig_bar.add_trace(go.Bar(name="Avg Talk", x=grp["group"], y=grp["avg_talk"].round(2),
                                  marker_color="#a8d5a2", text=grp["avg_talk"].round(2), textposition="outside"))
        fig_bar.add_trace(go.Bar(name="Avg Walk", x=grp["group"], y=grp["avg_walk"].round(2),
                                  marker_color="#74b9ff", text=grp["avg_walk"].round(2), textposition="outside"))
        fig_bar.add_trace(go.Bar(name="Avg Gap",  x=grp["group"], y=grp["avg_gap"].round(3),
                                  marker_color="#ff6b6b", text=grp["avg_gap"].round(3), textposition="outside"))
        fig_bar.update_layout(barmode="group", paper_bgcolor="rgba(0,0,0,0)",
                               plot_bgcolor="rgba(0,0,0,0)", font_color="#e8f0e9",
                               xaxis=dict(gridcolor="#2a3d2e"), yaxis=dict(gridcolor="#2a3d2e"),
                               legend=dict(font=dict(color="#e8f0e9")), margin=dict(t=20,b=20))
        st.plotly_chart(fig_bar, use_container_width=True)

    # Full results table
    st.markdown(f"### All Companies — {version_label} Risk Table")
    st.markdown("<div class='info-box'>Sorted by Gap Score (highest risk first). Toggle version above to switch between V1 and V2 results.</div>",
                unsafe_allow_html=True)

    df_show = df_active[[
        "company_name", "group", "bord_bia_verified",
        talk_col, walk_col, gap_col, risk_col
    ]].copy().sort_values(gap_col, ascending=False)

    df_show.columns = ["Company", "Group", "Bord Bia ✅",
                        "Talk Score", "Walk Score", "Gap Score", "Risk Level"]

    def colour_risk(val):
        if "HIGH"   in str(val): return "background-color:#4a1515; color:#ff6b6b"
        if "MEDIUM" in str(val): return "background-color:#3d3015; color:#ffd166"
        if "LOW"    in str(val): return "background-color:#153d20; color:#06d6a0"
        return "background-color:#152a3d; color:#74b9ff"

    def colour_gap(val):
        try:
            v = float(val)
            if v >= 0.40:  return "color:#ff6b6b; font-weight:600"
            if v >= 0.20:  return "color:#ffd166; font-weight:600"
            if v > 0:      return "color:#06d6a0"
            return "color:#74b9ff"
        except: return ""

    styled = df_show.style\
        .applymap(colour_risk, subset=["Risk Level"])\
        .applymap(colour_gap,  subset=["Gap Score"])\
        .format({"Talk Score":"{:.2f}", "Walk Score":"{:.2f}", "Gap Score":"{:.3f}"})
    st.dataframe(styled, use_container_width=True, height=600)

# ============================================================
# PAGE 2: COMPANY DEEP DIVE
# ============================================================
elif page == "🔍 Company Deep Dive":

    st.markdown("# 🔍 Company Deep Dive")
    st.markdown("*Full ESG risk profile — compare V1 keyword vs V2 LLM scores side by side*")
    st.markdown("---")

    # Selector
    company_list = df_compare.sort_values("gap_v2", ascending=False)["company_name"].tolist()
    selected = st.selectbox("Select a company", company_list)

    row_v1 = df_v1[df_v1["company_name"] == selected].iloc[0]
    row_v2 = df_v2[df_v2["company_name"] == selected].iloc[0]
    row_cmp = df_compare[df_compare["company_name"] == selected].iloc[0]

    # LLM audit details if available
    llm_row = df_llm[df_llm["company_name"] == selected].iloc[0] if not df_llm.empty and selected in df_llm["company_name"].values else None

    st.markdown("---")

    # Header row
    col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
    with col1:
        st.markdown(f"## {selected}")
        st.markdown(f"**Group:** {row_cmp['group']}")
        bord = "✅ Bord Bia Origin Green Verified" if row_cmp["bord_bia_verified"] else "❌ Not Bord Bia Verified"
        st.markdown(f"**Verification:** {bord}")
    with col2:
        st.markdown(f"<div class='score-card'><div class='score-label'>V1 Risk</div><div style='margin-top:8px'>{risk_badge(row_cmp['risk_v1'])}</div></div>",
                    unsafe_allow_html=True)
    with col3:
        st.markdown(f"<div class='score-card'><div class='score-label'>V2 Risk</div><div style='margin-top:8px'>{risk_badge(row_cmp['risk_v2'])}</div></div>",
                    unsafe_allow_html=True)
    with col4:
        changed = "⬆️ Escalated" if row_cmp["gap_v2"] > row_cmp["gap_v1"] else "⬇️ De-escalated" if row_cmp["gap_v2"] < row_cmp["gap_v1"] else "➡️ Unchanged"
        colour  = "#ff6b6b" if "Esc" in changed else "#06d6a0" if "De-" in changed else "#74b9ff"
        st.markdown(f"<div class='score-card'><div class='score-label'>Classification</div><div style='color:{colour}; font-weight:600; margin-top:8px'>{changed}</div></div>",
                    unsafe_allow_html=True)

    st.markdown("---")

    # V1 vs V2 gauge comparison
    st.markdown("### 📊 Score Comparison — V1 vs V2")
    col_v1, col_v2 = st.columns(2)

    with col_v1:
        st.markdown("<div class='info-box'><span class='badge-v1'>V1</span> &nbsp;Keyword Counting Pipeline</div>",
                    unsafe_allow_html=True)
        g1, g2, g3 = st.columns(3)
        with g1: st.plotly_chart(make_gauge(row_cmp["talk_v1"], "Talk V1", "#a8d5a2"), use_container_width=True)
        with g2: st.plotly_chart(make_gauge(row_cmp["walk_v1"], "Walk V1", "#74b9ff"), use_container_width=True)
        with g3:
            gap_c = "#ff6b6b" if row_cmp["gap_v1"]>=0.40 else "#ffd166" if row_cmp["gap_v1"]>=0.20 else "#06d6a0"
            st.plotly_chart(make_gauge(row_cmp["gap_v1"]*10, "Gap V1 ×10", gap_c), use_container_width=True)

    with col_v2:
        st.markdown("<div class='info-box'><span class='badge-v2'>V2</span> &nbsp;LLM Semantic Audit Pipeline</div>",
                    unsafe_allow_html=True)
        g4, g5, g6 = st.columns(3)
        with g4: st.plotly_chart(make_gauge(row_cmp["talk_v2"], "Talk V2", "#a8ffd4"), use_container_width=True)
        with g5: st.plotly_chart(make_gauge(row_cmp["walk_v2"], "Walk V2", "#74b9ff"), use_container_width=True)
        with g6:
            gap_c2 = "#ff6b6b" if row_cmp["gap_v2"]>=0.40 else "#ffd166" if row_cmp["gap_v2"]>=0.20 else "#06d6a0"
            st.plotly_chart(make_gauge(row_cmp["gap_v2"]*10, "Gap V2 ×10", gap_c2), use_container_width=True)

    st.markdown("---")

    # LLM audit details + sentiment
    col_audit, col_sent = st.columns([1, 1])

    with col_audit:
        st.markdown("### 🤖 LLM Audit Details (V2)")
        if llm_row is not None:
            st.markdown(f"""
            <div class='info-box'>
            <b>Claim Specificity</b> : {llm_row['claim_specificity']}/10<br>
            <b>Claim Credibility</b> : {llm_row['claim_credibility']}/10<br><br>
            <b>Claims Found:</b><br>
            {llm_row['claims_found'] if llm_row['claims_found'] else 'None detected'}<br><br>
            <b>Greenwashing Signals:</b><br>
            <span style='color:#ff6b6b'>{llm_row['greenwash_signals'] if llm_row['greenwash_signals'] else 'None detected'}</span><br><br>
            <b>Strengths:</b><br>
            <span style='color:#06d6a0'>{llm_row['strengths'] if llm_row['strengths'] else 'None detected'}</span><br><br>
            <b>Auditor Note:</b><br>
            <i>{llm_row['auditor_note']}</i>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info("LLM audit data not available for this company.")

    with col_sent:
        st.markdown("### 📰 FinBERT Sentiment (Walk Score)")
        pos = float(row_v1.get("walk_positive", 0))
        neg = float(row_v1.get("walk_negative", 0))
        neu = max(0, 1 - pos - neg)
        sent_data = {
            "Sentiment": ["Positive", "Neutral", "Negative"],
            "Score"    : [round(pos,3), round(neu,3), round(neg,3)]
        }
        fig_sent = px.bar(sent_data, x="Sentiment", y="Score",
                          color="Sentiment",
                          color_discrete_map={"Positive":"#06d6a0","Neutral":"#74b9ff","Negative":"#ff6b6b"},
                          text_auto=".3f")
        fig_sent.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                font_color="#e8f0e9", showlegend=False,
                                xaxis=dict(gridcolor="#2a3d2e"),
                                yaxis=dict(gridcolor="#2a3d2e", range=[0,1]),
                                margin=dict(t=10,b=10))
        st.plotly_chart(fig_sent, use_container_width=True)

        # Score table
        st.markdown(f"""
        | Metric | V1 | V2 |
        |--------|----|----|
        | Talk Score | {row_cmp['talk_v1']:.2f} | {row_cmp['talk_v2']:.2f} |
        | Walk Score | {row_cmp['walk_v1']:.2f} | {row_cmp['walk_v2']:.2f} |
        | Gap Score  | {row_cmp['gap_v1']:.3f} | {row_cmp['gap_v2']:.3f} |
        | Risk Level | {row_cmp['risk_v1']} | {row_cmp['risk_v2']} |
        | Bord Bia   | {'✅' if row_cmp['bord_bia_verified'] else '❌'} | {'✅' if row_cmp['bord_bia_verified'] else '❌'} |
        """)

# ============================================================
# PAGE 3: SECTOR ANALYSIS
# ============================================================
elif page == "📈 Sector Analysis":

    st.markdown("# 📈 Sector Analysis")
    st.markdown("*Greenwashing risk patterns across food sectors*")
    st.markdown("---")

    version = st.radio("Pipeline Version", ["V1 — Keyword Counting", "V2 — LLM Semantic Audit"], horizontal=True)
    df_active = df_v1 if "V1" in version else df_v2
    gap_col   = "gap_score" if "V1" in version else "gap_score_v2"
    talk_col  = "talk_score" if "V1" in version else "talk_norm"
    walk_col  = "walk_score_norm" if "V1" in version else "walk_norm"
    risk_col  = "risk_level" if "V1" in version else "risk_level_v2"

    st.markdown("---")

    # Sector gap bar
    st.markdown("### Average Gap Score by Sector")
    if "sector" in df_active.columns:
        sec = df_active.groupby("sector").agg(
            avg_gap=(gap_col,"mean"), count=("company_name","count")
        ).reset_index().sort_values("avg_gap", ascending=False)
        fig_sec = px.bar(sec, x="sector", y="avg_gap", color="avg_gap",
                         color_continuous_scale=["#06d6a0","#ffd166","#ff6b6b"],
                         text=sec["avg_gap"].round(3))
        fig_sec.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                               font_color="#e8f0e9", coloraxis_showscale=False,
                               xaxis=dict(gridcolor="#2a3d2e"), yaxis=dict(gridcolor="#2a3d2e"),
                               margin=dict(t=20,b=20))
        fig_sec.update_traces(textposition="outside", textfont_color="#e8f0e9")
        st.plotly_chart(fig_sec, use_container_width=True)
    else:
        st.info("Sector column not available in this dataset.")

    st.markdown("---")

    # Scatter plot
    st.markdown("### 🗺️ The Greenwashing Map — Talk vs Walk")
    st.markdown("""
    <div class='info-box'>
    Top-left = <b style='color:#ff6b6b'>High Risk</b> (big claims, poor reputation).
    Bottom-right = <b style='color:#06d6a0'>Credible</b> (modest claims, strong reputation).
    </div>""", unsafe_allow_html=True)

    fig_scatter = px.scatter(df_active, x=walk_col, y=talk_col,
                              color=risk_col, color_discrete_map=color_map,
                              text="company_name", symbol="group",
                              labels={walk_col:"Walk Score →", talk_col:"Talk Score ↑"})
    fig_scatter.update_traces(textposition="top center",
                               textfont=dict(size=9, color="#c8e6c9"),
                               marker=dict(size=10))
    fig_scatter.add_hline(y=5, line_dash="dot", line_color="#2a3d2e")
    fig_scatter.add_vline(x=5, line_dash="dot", line_color="#2a3d2e")
    fig_scatter.add_annotation(x=1, y=9.5, text="⚠️ High Risk Zone",
                                 showarrow=False, font=dict(color="#ff6b6b", size=11))
    fig_scatter.add_annotation(x=8, y=0.5, text="✅ Credible Zone",
                                 showarrow=False, font=dict(color="#06d6a0", size=11))
    fig_scatter.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                               font_color="#e8f0e9",
                               xaxis=dict(gridcolor="#1a2e1e", range=[-0.5,11]),
                               yaxis=dict(gridcolor="#1a2e1e", range=[-0.5,11]),
                               legend=dict(font=dict(color="#e8f0e9"), bgcolor="#1a2e1e"),
                               height=550, margin=dict(t=20,b=20))
    st.plotly_chart(fig_scatter, use_container_width=True)

    st.markdown("---")
    st.markdown("### 🔑 Key Pilot Findings")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""<div class='info-box'><b>📡 News Coverage</b><br><br>
        Google News RSS achieved <b>100% coverage</b> across all 26 Irish Food SMEs.</div>""",
                    unsafe_allow_html=True)
    with c2:
        nv = df_active[df_active["bord_bia_verified"]==False][gap_col].mean()
        v  = df_active[df_active["bord_bia_verified"]==True][gap_col].mean()
        diff = round(((nv-v)/v)*100) if v > 0 else 0
        st.markdown(f"""<div class='info-box'><b>📏 Verification Gap</b><br><br>
        Non-verified companies show <b>{diff}% higher avg gap score</b> than Bord Bia verified.</div>""",
                    unsafe_allow_html=True)
    with c3:
        top = df_active.loc[df_active[gap_col].idxmax(), "company_name"]
        st.markdown(f"""<div class='info-box'><b>🚩 Strongest Signal</b><br><br>
        <b>{top}</b> recorded the highest gap score — highest claims vs lowest walk score.</div>""",
                    unsafe_allow_html=True)

# ============================================================
# PAGE 4: V1 vs V2 COMPARISON
# ============================================================
elif page == "🔬 V1 vs V2 Comparison":

    st.markdown("# 🔬 V1 vs V2 — Pipeline Comparison")
    st.markdown("*How much did LLM-based claim extraction improve greenwashing detection?*")
    st.markdown("---")

    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    changed = df_compare[df_compare["risk_v1"] != df_compare["risk_v2"]]
    with col1:
        st.metric("Companies Reclassified", f"{len(changed)}/26",
                  help="Companies whose risk level changed between V1 and V2")
    with col2:
        st.metric("Reclassification Rate", f"{round(len(changed)/26*100)}%")
    with col3:
        st.metric("Avg Gap V1", f"{df_compare['gap_v1'].mean():.3f}")
    with col4:
        st.metric("Avg Gap V2", f"{df_compare['gap_v2'].mean():.3f}",
                  delta=f"{df_compare['gap_v2'].mean()-df_compare['gap_v1'].mean():.3f}")

    st.markdown("---")

    # Reclassified companies highlight
    st.markdown("### 🔄 Companies That Changed Risk Classification")
    st.markdown("""<div class='info-box'>
    These are the 5 companies where the LLM audit produced a different risk verdict than keyword counting.
    Each change represents a correction — either catching a false positive or revealing a missed signal.
    </div>""", unsafe_allow_html=True)

    for _, row in changed.iterrows():
        direction = "⬆️ Escalated" if row["gap_v2"] > row["gap_v1"] else "⬇️ De-escalated"
        col_a, col_b, col_c, col_d = st.columns([2,2,2,2])
        with col_a:
            st.markdown(f"**{row['company_name']}**<br>{row['group']}",
                        unsafe_allow_html=True)
        with col_b:
            st.markdown(f"<span class='badge-v1'>V1</span> {risk_badge(row['risk_v1'])}",
                        unsafe_allow_html=True)
        with col_c:
            st.markdown(f"<span class='badge-v2'>V2</span> {risk_badge(row['risk_v2'])}",
                        unsafe_allow_html=True)
        with col_d:
            colour = "#ff6b6b" if "Esc" in direction else "#06d6a0"
            st.markdown(f"<span style='color:{colour}'>{direction}</span>",
                        unsafe_allow_html=True)
        st.markdown("---")

    # Full comparison table
    st.markdown("### 📋 Full V1 vs V2 Comparison Table")
    df_show = df_compare[[
        "company_name","group","bord_bia_verified",
        "talk_v1","talk_v2","gap_v1","gap_v2",
        "risk_v1","risk_v2"
    ]].copy().sort_values("gap_v2", ascending=False)

    df_show["talk_Δ"] = (df_show["talk_v2"] - df_show["talk_v1"]).round(2)
    df_show["gap_Δ"]  = (df_show["gap_v2"]  - df_show["gap_v1"]).round(3)

    def colour_delta(val):
        try:
            v = float(val)
            if v > 0:  return "color:#ff6b6b"
            if v < 0:  return "color:#06d6a0"
            return "color:#74b9ff"
        except: return ""

    styled = df_show.style\
        .applymap(lambda v: "color:#ff6b6b; font-weight:600" if "HIGH" in str(v)
                  else "color:#ffd166; font-weight:600" if "MEDIUM" in str(v)
                  else "color:#06d6a0" if "LOW" in str(v)
                  else "color:#74b9ff", subset=["risk_v1","risk_v2"])\
        .applymap(colour_delta, subset=["talk_Δ","gap_Δ"])\
        .format({"talk_v1":"{:.2f}","talk_v2":"{:.2f}",
                 "gap_v1":"{:.3f}","gap_v2":"{:.3f}",
                 "talk_Δ":"{:+.2f}","gap_Δ":"{:+.3f}"})
    st.dataframe(styled, use_container_width=True, height=600)

    st.markdown("---")

    # Talk score evolution scatter
    st.markdown("### 📈 Talk Score Evolution — V1 vs V2")
    st.markdown("""<div class='info-box'>
    Each point is a company. Points <b style='color:#ff6b6b'>above the diagonal</b> had their Talk Score 
    raised by the LLM (stronger claims found). Points <b style='color:#06d6a0'>below</b> were de-escalated.
    </div>""", unsafe_allow_html=True)

    fig_evo = px.scatter(df_compare, x="talk_v1", y="talk_v2",
                          color="group", text="company_name",
                          color_discrete_map={"Verified":"#a8d5a2","Not Verified":"#ff6b6b"},
                          labels={"talk_v1":"Talk Score V1 →","talk_v2":"Talk Score V2 ↑"})
    # Diagonal reference line
    fig_evo.add_shape(type="line", x0=0, y0=0, x1=10, y1=10,
                       line=dict(color="#2a3d2e", dash="dot"))
    fig_evo.add_annotation(x=8, y=9, text="No change line",
                             showarrow=False, font=dict(color="#4a6e4e", size=10))
    fig_evo.update_traces(textposition="top center",
                           textfont=dict(size=9, color="#c8e6c9"),
                           marker=dict(size=10))
    fig_evo.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                           font_color="#e8f0e9",
                           xaxis=dict(gridcolor="#1a2e1e", range=[-0.5,11]),
                           yaxis=dict(gridcolor="#1a2e1e", range=[-0.5,11]),
                           legend=dict(font=dict(color="#e8f0e9"), bgcolor="#1a2e1e"),
                           height=500, margin=dict(t=20,b=20))
    st.plotly_chart(fig_evo, use_container_width=True)