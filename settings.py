
# settings.py

# --- Gemini model IDs (current names) ---
GEMINI_TEXT_MODEL = "gemini-2.5-flash"       # or "gemini-2.5-flash-lite" / "gemini-2.5-pro"
GEMINI_EMBED_MODEL = "gemini-embedding-001"  # embeddings model

# --- Scoring weights ---
# final_score = 0.65 * weighted_rule + 0.25 * semantic_sim + 0.10 * cv_quality
WEIGHTS = {
    "rule_match": 0.65,
    "semantic_sim": 0.25,
    "cv_quality": 0.10,
}

# --- Analytics & thresholds ---
SCORE_BINS = [0, 25, 50, 75, 90, 100]
MIN_SCORE_ALERT = 60.0
