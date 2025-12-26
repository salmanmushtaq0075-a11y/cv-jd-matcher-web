
# app.py
import os
import re
import json
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
import streamlit as st

from matcher import (
    find_skills_regex,
    find_skills_fuzzy,
)
from llm import build_skills_map_from_jd, local_fallback_skills, embed_texts, generate_cv_tips
from analytics import build_ranking_table, score_distribution, top_missing_skills
from settings import WEIGHTS, MIN_SCORE_ALERT, GEMINI_TEXT_MODEL, GEMINI_EMBED_MODEL

# ---------------------- Page & Theme ----------------------

st.set_page_config(page_title="CV–JD Match Assistant", page_icon="📄", layout="wide")

def inject_css():
    st.markdown("""
    <style>
      html, body, [class*="css"] {
        font-family: Inter, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, "Noto Sans";
      }
      .main { padding-top: 14px; }
      .hero {
        padding: 22px;
        border-radius: 18px;
        background: linear-gradient(120deg, #4f46e5 0%, #06b6d4 50%, #22c55e 100%);
        color: white;
        box-shadow: 0 8px 24px rgba(0,0,0,.12);
        margin-bottom: 12px;
      }
      .hero h1 { margin: 0; font-size: 26px; font-weight: 800; }
      .hero p { margin: 6px 0 0; opacity: .95; font-size: 14px; }
      .glass {
        background: rgba(255,255,255,.60);
        backdrop-filter: blur(8px);
        border-radius: 16px;
        border: 1px solid rgba(255,255,255,.35);
        padding: 14px;
        margin-bottom: 8px;
      }
      .section-title { font-weight: 700; font-size: 18px; margin-bottom: 6px; }
      .pill {
        display: inline-block; padding: 4px 10px; border-radius: 999px;
        font-size: 12px; font-weight: 600; margin-right: 6px;
        background: #eef2ff; color: #3730a3; border: 1px solid #c7d2fe;
      }
      .pill.red   { background: #fee2e2; color: #991b1b; border-color: #fecaca; }
      .pill.green { background: #dcfce7; color: #166534; border-color: #bbf7d0; }
      .pill.blue  { background: #e0f2fe; color: #075985; border-color: #bae6fd; }
      .scorebar { width: 100%; height: 8px; background: #eef2ff; border-radius: 999px; overflow: hidden; }
      .scorebar > div { height: 100%; background: linear-gradient(90deg, #22c55e 0%, #06b6d4 50%, #4f46e5 100%); }
      .hint { font-size: 12px; opacity: .8; }
    </style>
    """, unsafe_allow_html=True)

inject_css()
st.markdown("""
<div class="hero">
  <h1>📄 CV–JD Match Assistant</h1>
  <p>LLM‑extracted skills from JD (no JSON), multi‑CV ranking, evidence, ATS health, analytics, and CV improvement tips.</p>
</div>
""", unsafe_allow_html=True)

# ---------------------- Secrets bootstrap (safe) ----------------------
try:
    KEY_FROM_SECRETS = st.secrets["GEMINI_API_KEY"]
except Exception:
    KEY_FROM_SECRETS = None
if KEY_FROM_SECRETS:
    os.environ["GEMINI_API_KEY"] = KEY_FROM_SECRETS

st.sidebar.caption("API key in secrets: " + ("✅ found" if KEY_FROM_SECRETS else "❌ missing"))
st.sidebar.caption("Env GEMINI_API_KEY set: " + ("✅ yes" if bool(os.getenv("GEMINI_API_KEY")) else "❌ no"))

# ---------------------- Role templates (optional pre-weight) ----------------------
ROLE_TEMPLATES: Dict[str, List[str]] = {
    "Data Engineer": ["python", "sql", "pyspark", "spark", "airflow", "aws", "docker"],
    "ML Engineer":   ["python", "tensorflow", "pytorch", "mlops", "docker", "cloud"],
    "QA Engineer":   ["selenium", "cypress", "jira", "api testing", "sql"],
    "Frontend Dev":  ["javascript", "react", "typescript", "css", "html", "vite"],
    "Backend Dev":   ["python", "django", "flask", "fastapi", "postgresql", "docker"],
}

# ---------------------- Sidebar settings ----------------------
st.sidebar.header("⚙️ Settings")
use_fuzzy = st.sidebar.checkbox("Use fuzzy matching", value=True)
fuzzy_threshold = st.sidebar.slider("Fuzzy threshold", 70, 95, 85)

tips_on = st.sidebar.checkbox("Generate CV improvement tips (LLM)", value=True)
tips_scope = st.sidebar.selectbox("Tips scope", ["Top-3 CVs", "All CVs"], index=0)

selected_role = st.sidebar.selectbox(
    "Role template (optional)",
    ["(none)"] + list(ROLE_TEMPLATES.keys()),
    index=0,
)

st.sidebar.caption(f"Text model: `{GEMINI_TEXT_MODEL}` • Embeddings: `{GEMINI_EMBED_MODEL}`")

# ---------------------- Tabs ----------------------
tabs = st.tabs(["📤 Upload", "🧮 Match", "📊 Analytics"
                , "⬇ Export"])

# Upload tab: JD box + optional upload
with tabs[0]:
    st.markdown('<div class="glass">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Job Description</div>', unsafe_allow_html=True)
    jd_text_input = st.text_area(
        "Paste or write the JD here",
        height=220,
        placeholder="Paste the full JD (responsibilities, required skills, nice-to-haves, tools, platforms)…",
    )
    jd_file = st.file_uploader("Or upload JD (.txt)", type=["txt"], help="If provided and the box is empty, it will be used.")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="glass">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Candidate CVs</div>', unsafe_allow_html=True)
    cv_files = st.file_uploader("Upload CVs (.txt)", type=["txt"], accept_multiple_files=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ---------------------- Helpers ----------------------
def cosine_sim(a: List[float], b: List[float]) -> float:
    a, b = np.array(a, dtype=float), np.array(b, dtype=float)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom else 0.0

def extract_evidence(text: str, skill: str, variants: List[str], max_sentences: int = 2) -> List[str]:
    normalized = skill.lower()
    candidates = [normalized] + [v.lower() for v in (variants or [])]
    sentences = re.split(r'(?<=[.!?])\s+', text)
    hits = []
    for s in sentences:
        s_low = s.lower()
        if any(re.search(r"\b" + re.escape(p) + r"\b", s_low) for p in candidates):
            hits.append(s.strip())
            if len(hits) >= max_sentences:
                break
    return hits

def ats_health(text: str) -> Tuple[float, Dict[str, bool]]:
    checks = {
        "has_email": bool(re.search(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", text)),
        "has_phone": bool(re.search(r"\+?\d[\d\-\s]{7,}\d", text)),
        "has_sections": any(k in text.lower() for k in ["experience", "education", "skills", "projects"]),
        "word_count_ok": 250 <= len(text.split()) <= 2000,
        "no_weird_chars": not bool(re.search(r"[\uFFFD]", text)),
    }
    score = (checks["has_email"] + checks["has_phone"] + checks["has_sections"]) * 25 \
            + (checks["word_count_ok"]) * 15 + (checks["no_weird_chars"]) * 10
    return float(score), checks

@st.cache_data(show_spinner=False)
def cached_skills_map(jd_text: str) -> Dict[str, List[str]]:
    return build_skills_map_from_jd(jd_text)

@st.cache_data(show_spinner=False)
def cached_cv_tips(cv_text: str, jd_text: str, matched: List[str], missing: List[str], checks: Dict[str, bool]) -> List[str]:
    return generate_cv_tips(cv_text, jd_text, matched, missing, checks)

# ---------------------- Match tab ----------------------
with tabs[1]:
    st.markdown('<div class="glass">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Run matching</div>', unsafe_allow_html=True)
    run_btn = st.button("🚀 Run Matching")
    st.markdown('</div>', unsafe_allow_html=True)

    if run_btn:
        # Determine JD text: prefer box; otherwise file
        if jd_text_input and jd_text_input.strip():
            jd_text = jd_text_input.strip()
        elif jd_file:
            try:
                jd_text = jd_file.read().decode("utf-8", errors="ignore")
            except Exception:
                jd_text = jd_file.read().decode("latin-1", errors="ignore")
        else:
            st.error("Please paste the JD or upload a JD file.")
            st.stop()

        if not cv_files:
            st.error("Please upload at least one CV (.txt).")
            st.stop()

        # Read CVs
        cvs: List[Dict] = []
        for cv in cv_files:
            try:
                text = cv.read().decode("utf-8", errors="ignore")
            except Exception:
                text = cv.read().decode("latin-1", errors="ignore")
            cvs.append({"name": cv.name, "text": text})

        # --- 1) Build skills map directly from JD via LLM (cached) or fallback ---
        if not os.getenv("GEMINI_API_KEY"):
            st.error("Gemini API key missing. Add to `.streamlit/secrets.toml` as GEMINI_API_KEY or set the env var.")
            st.stop()

        with st.spinner("Extracting skills from JD (Gemini)…"):
            skills_map = cached_skills_map(jd_text)

        if not skills_map:
            # Never block the demo: use local fallback extractor
            skills_map = local_fallback_skills(jd_text)
            st.warning("Using local fallback skills extractor due to rate limit or parsing. Matching continues.")

        # JD canonical list from skills_map
        canonical_list = sorted(list(skills_map.keys()))

        # --- 2) Detect JD skills (rule + fuzzy) for coverage baseline ---
        jd_regex, _ = find_skills_regex(jd_text, skills_map)
        jd_fuzzy, _ = find_skills_fuzzy(jd_text, skills_map, threshold=fuzzy_threshold) if use_fuzzy else ([], [])
        jd_skills = sorted(set(jd_regex + jd_fuzzy)) or canonical_list

        # Role template pre-weights
        role_weights: Dict[str, float] = {s: 1.0 for s in jd_skills}
        selected_base = ROLE_TEMPLATES.get(selected_role, []) if selected_role in ROLE_TEMPLATES else []
        for s in jd_skills:
            if s in selected_base:
                role_weights[s] = 2.0

        # --- 3) First pass over CVs ---
        interim: List[Dict] = []
        for item in cvs:
            name, text = item["name"], item["text"]
            cv_regex, _ = find_skills_regex(text, skills_map)
            cv_fuzzy, _ = find_skills_fuzzy(text, skills_map, threshold=fuzzy_threshold) if use_fuzzy else ([], [])
            cv_skills = sorted(set(cv_regex + cv_fuzzy))

            matched = sorted(set(jd_skills) & set(cv_skills))
            missing = sorted(set(jd_skills) - set(cv_skills))
            hits = sum(role_weights.get(s, 1.0) for s in matched)
            total = sum(role_weights.values()) or 1.0
            weighted_score = round((hits / total) * 100, 2)

            quality_score, quality_checks = ats_health(text)

            interim.append({
                "candidate": name,
                "text": text,
                "skills": cv_skills,
                "matched": matched,
                "missing": missing,
                "weighted_score": weighted_score,
                "quality_score": quality_score,
                "quality_checks": quality_checks,
            })

        # --- 4) Batch embeddings for JD + CVs ---
        try:
            texts = [jd_text] + [r["text"] for r in interim]
            all_embs = embed_texts(texts)
            jd_embed = all_embs[0]
            cv_embs = all_embs[1:]
        except Exception:
            jd_embed, cv_embs = None, [None] * len(interim)
            st.warning("Embeddings failed (quota/network). Falling back to rule/ATS only.")

        # --- 5) Final scoring & evidence + CV tips (Top‑3 or All) ---
        results: List[Dict] = []
        for i, r in enumerate(interim):
            sim_raw = 0.0
            if jd_embed is not None and cv_embs[i] is not None:
                sim_raw = cosine_sim(jd_embed, cv_embs[i])      # [-1, 1]
            sim_pct = (sim_raw + 1.0) * 50.0                    # [0, 100]

            final_score = round(
                WEIGHTS.get("rule_match", 0.65) * r["weighted_score"]
                + WEIGHTS.get("semantic_sim", 0.25) * sim_pct
                + WEIGHTS.get("cv_quality", 0.10) * r["quality_score"],
                2
            )

            evidence = {}
            for s in r["matched"][:3]:
                evidence[s] = [e for e in extract_evidence(r["text"], s, skills_map.get(s, []), max_sentences=2) if e]

            # IMPORTANT: initialize tips to [] so downstream access never KeyErrors
            results.append({
                "candidate": r["candidate"],
                "final_score": final_score,
                "weighted_rule": r["weighted_score"],
                "semantic_sim": round(sim_raw, 4),
                "cv_quality": r["quality_score"],
                "matched": r["matched"],
                "missing": r["missing"],
                "evidence": evidence,
                "quality_checks": r["quality_checks"],
                "text": r["text"],
                "tips": [],  # <-- initialized here
            })

        # Build ranking
        df = build_ranking_table(results)
        df["final_score"] = pd.to_numeric(df["final_score"], errors="coerce")

        # Generate CV tips for scope
        if tips_on and not df.empty:
            if tips_scope == "Top-3 CVs":
                target_names = df.head(3)["candidate"].tolist()
            else:
                target_names = df["candidate"].tolist()

            # index for quick lookup
            name_to_row = {r["candidate"]: r for r in results}
            for nm in target_names:
                row = name_to_row.get(nm)
                if not row:
                    continue
                if row.get("tips"):  # already present
                    continue
                # LLM tips (robust JSON mode; rate-limit-aware)
                row["tips"] = cached_cv_tips(
                    row["text"],
                    jd_text,
                    row["matched"],
                    row["missing"],
                    row["quality_checks"],
                ) or []  # ensure list even if empty

        # ---------- Ranking UI ----------
        st.markdown('<div class="glass">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">🏆 Ranking</div>', unsafe_allow_html=True)

        if df.empty:
            st.info("No results. Check uploads and try again.")
            st.markdown('</div>', unsafe_allow_html=True)
            st.stop()

        for row in df.to_dict(orient="records"):
            st.markdown(f"""
            <div style="display:flex; align-items:center; gap:12px; margin:8px 0;">
              <div style="flex:1;">
                <div style="font-weight:700;">{row['candidate']}</div>
                <div class="scorebar"><div style="width:{row['final_score']}%;"></div></div>
                <div class="hint" style="margin-top:4px;">
                  <span class="pill blue">Rule {row['weighted_rule']:.1f}%</span>
                  <span class="pill green">Sim {(row['semantic_sim']+1)*50:.1f}%</span>
                  <span class="pill">ATS {row['cv_quality']:.0f}</span>
                  <span class="pill">Final {row['final_score']:.1f}%</span>
                </div>
              </div>
              <div>
                <span class="pill red">Missing {len(row['missing'])}</span>
                <span class="pill green">Matched {len(row['matched'])}</span>
              </div>
            </div>
            """, unsafe_allow_html=True)

            # SAFE tips access to avoid KeyError
            with st.expander(f"🔎 Evidence & Tips — {row['candidate']}"):
                cols = st.columns(2)
                with cols[0]:
                    st.write("**Matched skills**", row["matched"])
                    st.write("**Missing skills**", row["missing"])
                    st.write("**Evidence (sentences)**")
                    # find current candidate in results and read evidence
                    evidence = next((r.get("evidence", {}) for r in results if r["candidate"] == row["candidate"]), {})
                    if not evidence:
                        st.write("—")
                    else:
                        for sk, sents in evidence.items():
                            st.markdown(f"• **{sk}**")
                            for s in sents:
                                st.markdown(f"  - {s}")
                with cols[1]:
                    st.write("**Auto CV improvement tips**")
                    tips = next((r.get("tips", []) for r in results if r["candidate"] == row["candidate"]), [])
                    if tips:
                        for t in tips:
                            st.markdown(f"- {t}")
                    else:
                        st.caption("Tips not generated for this candidate (scope or quota).")

        low = df[df["final_score"] < MIN_SCORE_ALERT]
        if not low.empty:
            st.warning(f"{len(low)} candidate(s) below threshold ({MIN_SCORE_ALERT}).")

        st.markdown('</div>', unsafe_allow_html=True)

        # ---------- Analytics tab ----------
        with tabs[2]:
            st.markdown('<div class="glass">', unsafe_allow_html=True)
            st.markdown('<div class="section-title">📊 Analytics</div>', unsafe_allow_html=True)

            if df.empty:
                st.info("No candidates to analyze.")
            else:
                top_row = df.head(1).iloc[0].to_dict()
                k1, k2, k3 = st.columns(3)
                k1.metric("Top Candidate", top_row["candidate"])
                k2.metric("Top Score", f"{top_row['final_score']:.1f}%")
                k3.metric("Candidates", f"{len(df)}")

                dist = score_distribution(df)
                if dist.empty:
                    st.write("No scores yet.")
                else:
                    st.bar_chart(dist.set_index("bin")["count"])

                st.caption("Top missing skills across candidates")
                tm = top_missing_skills(results, top_k=10)
                st.table(tm)

            st.markdown('</div>', unsafe_allow_html=True)



        # ---------- Export tab ----------
        with tabs[4]:
            st.markdown('<div class="glass">', unsafe_allow_html=True)
            st.markdown('<div class="section-title">⬇ Export</div>', unsafe_allow_html=True)

            st.download_button(
                "⬇ Download Ranking (CSV)",
                data=df.to_csv(index=False),
                file_name="ranking.csv",
                mime="text/csv",
                use_container_width=True,
            )
            st.download_button(
                "⬇ Download Results (JSON)",
                data=json.dumps(df.to_dict(orient="records"), indent=2),
                file_name="results.json",
                mime="application/json",
                use_container_width=True,
            )
            st.markdown('</div>', unsafe_allow_html=True)
