
# llm.py
import os
import time
import re
from typing import List, Dict
from google import genai
from google.genai import types
from google.genai.errors import ClientError
from settings import GEMINI_TEXT_MODEL, GEMINI_EMBED_MODEL

# --- Client ---
def _client() -> genai.Client:
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Gemini API key not found. Add GEMINI_API_KEY in .streamlit/secrets.toml "
            "or export it as an environment variable."
        )
    return genai.Client(api_key=api_key)

# --- Defensive JSON parsing ---
def _safe_json(text: str) -> Dict:
    try:
        import json
        return json.loads(text)
    except Exception:
        # try to salvage a {...} block
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                import json
                return json.loads(text[start:end+1])
            except Exception:
                return {}
        return {}

# --- Local fallback extractor: curated tech lexicon ---
_FALLBACK_LEXICON = {
    # data/engineering
    "python": ["py", "python3", "pandas", "numpy"],
    "sql": ["postgresql", "mysql", "mssql", "sqlite", "oracle"],
    "pyspark": ["spark", "databricks"],
    "airflow": ["apache airflow"],
    "aws": ["amazon web services", "s3", "lambda", "redshift", "glue", "emr"],
    "gcp": ["google cloud", "bigquery", "gcs"],
    "azure": ["azure data factory", "synapse"],
    "docker": ["containers"],
    "kubernetes": ["k8s"],
    "terraform": [],
    "git": ["github", "gitlab", "bitbucket"],
    "linux": ["ubuntu", "debian"],
    "snowflake": [],
    "kafka": [],
    "hadoop": [],
    "tableau": [],
    "power bi": ["pbi"],
    # ml/ai
    "pytorch": ["torch"],
    "tensorflow": ["tf", "keras"],
    "scikit-learn": ["sklearn"],
    "nlp": ["spaCy", "transformers"],
    "opencv": [],
    # web/backend
    "javascript": ["js", "node", "node.js"],
    "typescript": ["ts"],
    "react": ["react.js", "reactjs"],
    "html": [],
    "css": [],
    "django": [],
    "flask": [],
    "fastapi": [],
    "spring": ["spring boot"],
    "java": [],
    "c#": [".net", "dotnet"],
    "go": ["golang"],
    "php": ["laravel"],
    "graphql": [],
    "rest": ["api"],
    # qa/testing/devops
    "selenium": ["webdriver"],
    "cypress": ["e2e testing"],
    "pytest": [],
    "junit": [],
    "jenkins": [],
    "github actions": ["ci/cd"],
    "jira": [],
}

def local_fallback_skills(jd_text: str) -> Dict[str, List[str]]:
    text = jd_text.lower()
    found = {}
    for canonical, variants in _FALLBACK_LEXICON.items():
        hit = False
        # canonical
        if re.search(r"\b" + re.escape(canonical) + r"\b", text):
            hit = True
        # variants
        if not hit:
            for v in variants:
                if re.search(r"\b" + re.escape(v.lower()) + r"\b", text):
                    hit = True
                    break
        if hit:
            found[canonical] = [v.lower() for v in variants]
    return found

# --- LLM: build skills map from JD (JSON mode + retry + fallback) ---
def build_skills_map_from_jd(jd_text: str, max_skills: int = 40, variants_per_skill: int = 5) -> Dict[str, List[str]]:
    client = _client()
    prompt = (
        "From the provided Job Description, extract a canonical skills map.\n"
        f"- Return strict JSON with shape: {{\"skills\": {{\"skill\": [\"variant1\",\"variant2\", ...]}}}}\n"
        f"- Include at most {max_skills} skills, and up to {variants_per_skill} variants per skill.\n"
        "- Lowercase everything. Only include concrete, technical skills (omit soft skills).\n\n"
        "Job Description:\n" + jd_text
    )
    config = types.GenerateContentConfig(response_mime_type="application/json")

    # Try up to 3 times (handle 429 retry hints)
    for attempt in range(3):
        try:
            resp = client.models.generate_content(
                model=GEMINI_TEXT_MODEL,
                contents=types.Part.from_text(prompt),
                config=config,
            )
            data = _safe_json(resp.text)
            raw = (data or {}).get("skills", {})
            if isinstance(raw, dict) and raw:
                skills_map = {k.lower(): [v.lower() for v in (vals or [])] for k, vals in raw.items()}
                for k in list(skills_map.keys()):
                    if not isinstance(skills_map[k], list):
                        skills_map[k] = []
                return skills_map
        except ClientError as e:
            # Respect server retry hint for 429
            if getattr(e, "status_code", None) == 429:
                time.sleep(3)
            else:
                time.sleep(0.8)
        except Exception:
            time.sleep(0.5)

    # Fallback: local extractor so the demo never stops
    return local_fallback_skills(jd_text)

# --- Embeddings ---
def embed_texts(texts: List[str]) -> List[List[float]]:
    client = _client()
    result = client.models.embed_content(model=GEMINI_EMBED_MODEL, contents=texts)
    return [e.values for e in result.embeddings]

# --- CV improvement tips (JSON mode + retry) ---
def generate_cv_tips(cv_text: str, jd_text: str, matched: List[str], missing: List[str], quality_checks: Dict[str, bool]) -> List[str]:
    client = _client()
    gaps = [k for (k, ok) in quality_checks.items() if not ok]
    prompt = (
        "You are a senior technical recruiter and CV writing coach.\n"
        "Write 5–8 specific, actionable improvement tips for the candidate's CV, tailored to the Job Description.\n"
        "Each tip must be concise, start with a verb, and avoid generic phrasing.\n"
        "Return a strict JSON array of strings only.\n\n"
        f"Missing skills (prioritize): {missing}\n"
        f"Matched skills (keep & quantify): {matched}\n"
        f"ATS issues (False flags to fix): {gaps}\n\n"
        "Job Description:\n" + jd_text + "\n\n"
        "CV:\n" + cv_text + "\n"
    )
    config = types.GenerateContentConfig(response_mime_type="application/json")

    for attempt in range(2):
        try:
            resp = client.models.generate_content(
                model=GEMINI_TEXT_MODEL,
                contents=types.Part.from_text(prompt),
                config=config,
            )
            data = _safe_json(resp.text)
            if isinstance(data, list) and data:
                return [str(t).strip() for t in data][:8]
        except ClientError as e:
            if getattr(e, "status_code", None) == 429:
                time.sleep(3)
            else:
                time.sleep(0.8)
        except Exception:
            time.sleep(0.5)
    return []
