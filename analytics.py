
# analytics.py
import pandas as pd
from typing import List, Dict
from settings import SCORE_BINS

def build_ranking_table(results: List[Dict]) -> pd.DataFrame:
    df = pd.DataFrame(results)
    if "final_score" in df.columns:
        df = df.sort_values("final_score", ascending=False)
    return df

def score_distribution(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "final_score" not in df.columns:
        return pd.DataFrame({"bin": [], "count": []})
    s = pd.to_numeric(df["final_score"], errors="coerce")
    binned = pd.cut(s, bins=SCORE_BINS, include_lowest=True, right=True)
    counts = binned.value_counts(sort=False)
    out = counts.reset_index(name="count")
    out.rename(columns={"final_score": "bin"}, inplace=True)
    out["bin"] = out["bin"].astype(str)
    return out

def top_missing_skills(results: List[Dict], top_k: int = 10) -> pd.DataFrame:
    missing = []
    for r in results:
        for s in r.get("missing", []):
            missing.append(s)
    return (pd.Series(missing)
            .value_counts()
            .head(top_k)
            .rename_axis("skill")
            .reset_index(name="missing_count"))
