
# matcher.py
import json
import re
from typing import Dict, List, Tuple, Set
from rapidfuzz import process, fuzz

def load_skills_from_file(skills_path: str) -> Dict[str, List[str]]:
    """(Optional legacy path) Load skills.json and normalize to lowercase."""
    with open(skills_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {k.lower(): [v.lower() for v in (vals or [])] for k, vals in data.items()}

def find_skills_regex(text: str, skills_map: Dict[str, List[str]]) -> Tuple[List[str], List[str]]:
    """Strict word-boundary matching (exact tokens)."""
    text = text.lower()
    found: Set[str] = set()
    reasons: List[str] = []
    for canonical, variants in skills_map.items():
        phrases = [canonical] + (variants or [])
        for p in phrases:
            pattern = r"\b" + re.escape(p) + r"\b"
            if re.search(pattern, text):
                found.add(canonical)
                reasons.append(f"Found '{p}' → '{canonical}'")
                break
    return sorted(found), reasons

def find_skills_fuzzy(text: str, skills_map: Dict[str, List[str]], threshold: int = 85) -> Tuple[List[str], List[str]]:
    """Fuzzy matching to catch typos and close variants."""
    text_tokens = set(re.findall(r"[a-zA-Z0-9+.#]+", text.lower()))
    reasons: List[str] = []
    found: Set[str] = set()
    for canonical, variants in skills_map.items():
        candidates = [canonical] + (variants or [])
        for token in text_tokens:
            match, score, _ = process.extractOne(token, candidates, scorer=fuzz.ratio)
            if score >= threshold:
                found.add(canonical)
                reasons.append(f"Fuzzy '{token}'≈'{match}' ({score}) → '{canonical}'")
                break
    return sorted(found), reasons

def compute_match(jd_skills: List[str], cv_skills: List[str]) -> Tuple[float, List[str], List[str]]:
    """Simple coverage %: JD→CV."""
    if not jd_skills:
        return 0.0, [], []
    matched = sorted(set(jd_skills) & set(cv_skills))
    missing = sorted(set(jd_skills) - set(cv_skills))
    score = round((len(matched) / len(jd_skills)) * 100, 1)
    return score, matched, missing
