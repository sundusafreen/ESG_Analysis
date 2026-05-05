# ── greenlens_app.py — GreenLens Dynamic Dashboard ───────
# Upload any sustainability PDF → get ESG gap analysis
# Run with: streamlit run greenlens_app.py

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import PyPDF2
import requests
import xml.etree.ElementTree as ET
import urllib.parse
import re
import json
import time
from groq import Groq
from transformers import pipeline

# ════════════════════════════════════════════════════════════
# PAGE CONFIG
# ════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="GreenLens — ESG Greenwashing Detector",
    page_icon="🌿",
    layout="wide"
)

# ════════════════════════════════════════════════════════════
# API KEYS — paste yours here
# ════════════════════════════════════════════════════════════
GROQ_API_KEY = "REMOVED"

# ════════════════════════════════════════════════════════════
# LOAD FINBERT — cached so it only downloads once
# ════════════════════════════════════════════════════════════
@st.cache_resource
def load_finbert():
    return pipeline(
        "text-classification",
        model="ProsusAI/finbert",
        tokenizer="ProsusAI/finbert",
        top_k=None
    )

# ════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ════════════════════════════════════════════════════════════

def extract_pdf_text(uploaded_file):
    """
    Extract all text from an uploaded PDF file.
    PyPDF2 reads each page and joins the text together.
    Returns a single string of all text.
    """
    reader = PyPDF2.PdfReader(uploaded_file)
    text   = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + " "
    return text.strip()


def fetch_walk_data(company_name, max_articles=15):
    """
    Fetch recent news about the company from Google News RSS.
    Returns list of "title. description" strings.
    Only keeps articles where company name appears in text.
    """
    query    = f"{company_name} sustainability OR ESG OR environment OR greenwashing"
    encoded  = urllib.parse.quote(query)
    url      = f"https://news.google.com/rss/search?q={encoded}&hl=en-IE&gl=IE&ceid=IE:en"
    headers  = {"User-Agent": "Mozilla/5.0"}

    try:
        response = requests.get(url, headers=headers, timeout=10)
        root     = ET.fromstring(response.content)
        items    = root.findall(".//item")

        keywords = company_name.lower().split()
        texts    = []

        for item in items[:max_articles]:
            title = item.findtext("title") or ""
            desc  = item.findtext("description") or ""
            desc  = re.sub(r"<[^>]+>", "", desc).strip()
            combined       = f"{title}. {desc}".strip()
            combined_lower = combined.lower()

            # Only keep if company name keyword appears
            if any(kw in combined_lower for kw in keywords):
                if len(combined) > 20:
                    texts.append(combined)

        return texts

    except Exception as e:
        return []


def extract_esg_claims(company_name, text, groq_client):
    """
    Send PDF text to Groq/Llama and get back structured
    ESG claims and a Talk Score as JSON.
    """
    prompt = f"""
You are a senior ESG analyst reviewing a company sustainability report.
Extract every specific ESG claim and return them as JSON only.

Rules:
- Only extract claims that are specific and verifiable
- Ignore vague marketing language like "we care about the planet"
- Classify: E (Environmental), S (Social), G (Governance)
- Rate strength: "strong" (numbers/deadlines/partners), "moderate", "weak"
- Return ONLY valid JSON, nothing else

Company: {company_name}

Sustainability report text:
---
{text[:3000]}
---

Return exactly this JSON structure:
{{
  "company": "{company_name}",
  "claims": [
    {{"category": "E", "claim": "specific claim text", "strength": "strong"}}
  ],
  "talk_score": <0-10>,
  "summary": "<2 sentence summary of overall ESG positioning>"
}}
"""
    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": "You are an ESG analyst. Return valid JSON only."
            },
            {"role": "user", "content": prompt}
        ],
        temperature=0.1,
        max_tokens=1500
    )

    raw = response.choices[0].message.content.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]

    return json.loads(raw.strip())


def calculate_walk_score(texts, finbert_model):
    """
    Score Walk text using FinBERT.
    Returns walk_score (0-10) and sentence breakdown.
    """
    if not texts:
        return 5.0, 0, 0, 0

    # Split into sentences
    sentences = []
    for text in texts:
        parts = re.split(r'(?<=[.!?])\s+', text.strip())
        for p in parts:
            p = p.strip()
            if len(p) > 20:
                sentences.append(p[:512])

    if not sentences:
        return 5.0, 0, 0, 0

    results   = finbert_model(sentences, batch_size=8)
    scores    = []
    pos_count = neg_count = neu_count = 0

    NEGATIVE_AMPLIFIER = 1.5

    for result in results:
        label_scores = {r["label"]: r["score"] for r in result}
        pos = label_scores.get("positive", 0)
        neu = label_scores.get("neutral",  0)
        neg = label_scores.get("negative", 0)

        sentence_score = (pos * 1.0) + (neu * 0.2) + (neg * 0.0)
        top_label = max(label_scores, key=label_scores.get)

        if top_label == "positive":
            pos_count += 1
        elif top_label == "negative":
            neg_count += 1
            for _ in range(int(NEGATIVE_AMPLIFIER)):
                scores.append(0.0)
        else:
            neu_count += 1

        scores.append(sentence_score)

    avg       = sum(scores) / len(scores) if scores else 0.5
    pos_ratio = pos_count / len(sentences) if sentences else 0
    bonus     = 0.5 if pos_ratio >= 0.35 else 0
    walk_score = round(min(10, (avg * 10) + bonus), 2)

    return walk_score, pos_count, neu_count, neg_count


def calculate_gap(talk_score, walk_score):
    """
    Gap = (Talk - Walk) / Talk
    Bounded 0-1. Higher = more greenwashing risk.
    """
    if talk_score == 0:
        return 0
    return round(max(0, (talk_score - walk_score) / talk_score), 3)


def get_risk_level(gap):
    if gap >= 0.40:   return "🔴 HIGH"
    elif gap >= 0.20: return "🟡 MEDIUM"
    else:             return "🟢 LOW"


def generate_recommendations(company, claims, gap, walk_score,
                              pos_c, neg_c, groq_client):
    """
    Ask Groq to generate 4 specific recommendations based on
    the gap score, claims found, and sentiment breakdown.
    """
    claims_text = "\n".join([
        f"- [{c['category']}] {c['strength'].upper()}: {c['claim']}"
        for c in claims
    ])

    prompt = f"""
You are a senior ESG consultant writing a report for {company}.

Gap Score: {gap} ({"HIGH risk" if gap >= 0.4 else "MEDIUM risk" if gap >= 0.2 else "LOW risk"})
Walk Score: {walk_score}/10
Positive news sentences: {pos_c}
Negative news sentences: {neg_c}

ESG Claims found in their report:
{claims_text}

Write exactly 4 specific, actionable recommendations.
Each recommendation must:
- Reference a specific claim or gap found above
- Suggest a concrete next step
- Be 2-3 sentences long

Return as JSON only:
{{
  "recommendations": [
    {{"priority": "High/Medium/Low", "title": "short title", "detail": "2-3 sentence recommendation"}},
    ...
  ]
}}
"""
    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": "Return valid JSON only."},
            {"role": "user",   "content": prompt}
        ],
        temperature=0.2,
        max_tokens=1000
    )

    raw = response.choices[0].message.content.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]

    return json.loads(raw.strip())


# ════════════════════════════════════════════════════════════
# UI — HEADER
# ════════════════════════════════════════════════════════════
st.markdown("## 🌿 GreenLens")
st.markdown(
    "**ESG Greenwashing Detection Tool** · "
    "Upload a sustainability report → get instant gap analysis"
)
st.divider()

# ════════════════════════════════════════════════════════════
# UI — INPUT SECTION
# ════════════════════════════════════════════════════════════
col_input1, col_input2 = st.columns([1, 1])

with col_input1:
    company_name = st.text_input(
        "Company name",
        placeholder="e.g. Penneys, Chupi, Brown Thomas",
        help="Used to search for news articles (Walk data)"
    )

with col_input2:
    uploaded_pdf = st.file_uploader(
        "Upload sustainability report (PDF)",
        type=["pdf"],
        help="Annual report, CSR report, or sustainability page export"
    )

analyse_btn = st.button(
    "🔍 Analyse",
    type="primary",
    disabled=(not company_name or not uploaded_pdf)
)

# ════════════════════════════════════════════════════════════
# UI — ANALYSIS (runs when button clicked)
# ════════════════════════════════════════════════════════════
if analyse_btn:

    groq_client  = Groq(api_key=GROQ_API_KEY)
    finbert      = load_finbert()

    # ── Step 1: Extract PDF text ──────────────────────────
    with st.spinner("📄 Reading PDF..."):
        pdf_text   = extract_pdf_text(uploaded_pdf)
        word_count = len(pdf_text.split())

    if word_count < 50:
        st.error(
            "Could not extract enough text from this PDF. "
            "Try a different file or check the PDF is not scanned/image-only."
        )
        st.stop()

    st.success(f"✓ PDF read — {word_count:,} words extracted")

    # ── Step 2: Groq ESG claim extraction ─────────────────
    with st.spinner("🤖 Extracting ESG claims with Groq/Llama 3..."):
        try:
            groq_result = extract_esg_claims(
                company_name, pdf_text, groq_client
            )
            claims      = groq_result.get("claims", [])
            talk_score  = float(groq_result.get("talk_score", 5))
            esg_summary = groq_result.get("summary", "")
        except Exception as e:
            st.error(f"Groq extraction failed: {e}")
            st.stop()

    st.success(f"✓ {len(claims)} ESG claims extracted — Talk Score: {talk_score}/10")

    # ── Step 3: Fetch Walk data ────────────────────────────
    with st.spinner(f"📰 Fetching news about {company_name}..."):
        walk_articles = fetch_walk_data(company_name)

    if walk_articles:
        st.success(f"✓ {len(walk_articles)} relevant news articles found")
    else:
        st.warning(
            "No news articles found. Walk Score will use neutral baseline. "
            "Consider adding the company's country or sector to the name "
            "(e.g. 'Penneys Ireland')."
        )

    # ── Step 4: FinBERT Walk scoring ──────────────────────
    with st.spinner("📊 Scoring sentiment with FinBERT..."):
        walk_score, pos_c, neu_c, neg_c = calculate_walk_score(
            walk_articles, finbert
        )

    st.success(f"✓ Walk Score: {walk_score}/10 "
               f"(+{pos_c} positive / {neu_c} neutral / {neg_c} negative sentences)")

    # ── Step 5: Gap Score ─────────────────────────────────
    gap       = calculate_gap(talk_score, walk_score)
    risk      = get_risk_level(gap)

    # ── Step 6: Recommendations ───────────────────────────
    with st.spinner("📝 Generating recommendations..."):
        try:
            rec_result = generate_recommendations(
                company_name, claims, gap,
                walk_score, pos_c, neg_c, groq_client
            )
            recommendations = rec_result.get("recommendations", [])
        except Exception as e:
            recommendations = []

    # ════════════════════════════════════════════════════════
    # RESULTS DISPLAY
    # ════════════════════════════════════════════════════════
    st.divider()
    st.markdown(f"## Results — {company_name}")

    # ── Risk alert banner ─────────────────────────────────
    if gap >= 0.40:
        st.error(
            f"⚠️ **High greenwashing risk detected.** "
            f"Gap score of {gap:.3f} exceeds the 0.40 alert threshold. "
            f"Independent verification recommended before ESG-linked decisions."
        )
    elif gap >= 0.20:
        st.warning(
            f"⚡ **Medium risk — inconsistencies identified.** "
            f"Gap score of {gap:.3f}. Some claims lack external backing."
        )
    else:
        st.success(
            f"✅ **Low greenwashing risk.** "
            f"Gap score of {gap:.3f}. Claims broadly supported by evidence."
        )

    # ── Score cards ───────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Talk Score",  f"{talk_score:.1f} / 10",
              help="Intensity of ESG claims from PDF report")
    c2.metric("Walk Score",  f"{walk_score:.2f} / 10",
              help="Public sentiment from news coverage")
    c3.metric("Gap Score",   f"{gap:.3f}",
              help="(Talk - Walk) / Talk — higher = more risk")
    c4.metric("Risk Level",  risk)

    st.divider()

    # ── Two column charts ─────────────────────────────────
    left, right = st.columns(2)

    with left:
        st.markdown("#### Talk vs Walk")

        fig_gauge = go.Figure()
        fig_gauge.add_trace(go.Bar(
            x=["Talk Score", "Walk Score"],
            y=[talk_score, walk_score],
            marker_color=["#3ecf72", "#4a9eff"],
            text=[f"{talk_score:.1f}", f"{walk_score:.2f}"],
            textposition="outside"
        ))
        fig_gauge.add_hline(
            y=talk_score,
            line_dash="dash",
            line_color="#3ecf72",
            opacity=0.4
        )
        fig_gauge.update_layout(
            yaxis=dict(range=[0, 11]),
            height=300,
            margin=dict(l=10, r=10, t=10, b=10),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

    with right:
        st.markdown("#### Sentiment Breakdown (FinBERT)")

        fig_pie = go.Figure(go.Pie(
            labels=["Positive", "Neutral", "Negative"],
            values=[max(pos_c, 0.01), max(neu_c, 0.01), max(neg_c, 0.01)],
            marker_colors=["#3ecf72", "#8a9b8e", "#e85454"],
            hole=0.4,
            textinfo="label+percent"
        ))
        fig_pie.update_layout(
            height=300,
            margin=dict(l=10, r=10, t=10, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            showlegend=False
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    st.divider()

    # ── ESG Summary ───────────────────────────────────────
    if esg_summary:
        st.markdown("#### ESG Positioning Summary")
        st.info(esg_summary)

    # ── Extracted Claims ──────────────────────────────────
    st.markdown("#### Extracted ESG Claims")
    st.caption(
        "Extracted from your PDF by Groq/Llama 3. "
        "Strength rated by specificity, numbers, deadlines, and named partners."
    )

    cat_icons   = {"E": "🟢", "S": "🔵", "G": "🟣"}
    str_colours = {
        "strong":   "green",
        "moderate": "orange",
        "weak":     "red"
    }

    if claims:
        # Group by category
        for cat_label, cat_code in [
            ("Environmental", "E"),
            ("Social", "S"),
            ("Governance", "G")
        ]:
            cat_claims = [c for c in claims if c.get("category") == cat_code]
            if cat_claims:
                st.markdown(
                    f"**{cat_icons.get(cat_code,'')} {cat_label}**"
                )
                for claim in cat_claims:
                    strength = claim.get("strength", "moderate").lower()
                    colour   = str_colours.get(strength, "gray")
                    st.markdown(
                        f"  :{colour}[{strength.upper()}] — "
                        f"{claim.get('claim','')}"
                    )
    else:
        st.info("No specific claims extracted.")

    st.divider()

    # ── Recommendations ───────────────────────────────────
    st.markdown("#### Recommendations")
    st.caption(
        "Generated by Groq based on gap score, claims found, "
        "and sentiment analysis."
    )

    priority_colours = {
        "High":   "red",
        "Medium": "orange",
        "Low":    "green"
    }

    if recommendations:
        for i, rec in enumerate(recommendations):
            priority = rec.get("priority", "Medium")
            title    = rec.get("title",    "Recommendation")
            detail   = rec.get("detail",   "")
            colour   = priority_colours.get(priority, "gray")

            with st.expander(
                f"{i+1}. {title} "
                f"[:{colour}[{priority} Priority]]"
            ):
                st.write(detail)
    else:
        st.info("Recommendations unavailable.")

    st.divider()

    # ── Methodology note ──────────────────────────────────
    with st.expander("📐 Methodology & Data Sources"):
        st.markdown(f"""
**Talk Score** — Extracted from uploaded PDF using Groq/Llama 3.3-70B.
Claims classified as Environmental (E), Social (S), or Governance (G).
Scored 0–10 based on quantity and strength of specific, verifiable claims.

**Walk Score** — {len(walk_articles)} news articles fetched from Google News RSS
for query: *"{company_name} sustainability OR ESG OR environment"*.
Sentiment scored per sentence using FinBERT (ProsusAI/finbert).
Weighting: Positive=1.0 · Neutral=0.2 · Negative=0.0 + amplification penalty.

**Gap Score** — Formula: (Talk − Walk) / Talk.
Thresholds: <0.20 Low · 0.20–0.39 Medium · ≥0.40 High.

**Limitations** — PDF text extraction may miss tables and images.
Google News RSS covers last 30 days only. FinBERT trained on financial
text — may underweight implicit ESG negativity in general journalism.
Manual review recommended for High risk findings.
        """)

# ════════════════════════════════════════════════════════════
# FOOTER — shown always
# ════════════════════════════════════════════════════════════
st.markdown(
    """
    <div style='text-align:center;color:gray;font-size:11px;margin-top:3rem'>
    GreenLens · MSc Business Analytics Capstone Project ·
    Models: Groq/Llama 3.3-70B + FinBERT · 
    Not a substitute for professional ESG audit
    </div>
    """,
    unsafe_allow_html=True
)