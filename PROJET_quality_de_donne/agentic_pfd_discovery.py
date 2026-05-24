import os
import re
import json
import pandas as pd
from dotenv import load_dotenv
from mistralai.client import Mistral

from classical_pfd_algorithm import (
    CandidateGenerator,
    PFDValidator,
    _format_lhs
)

load_dotenv()


# =========================
# MISTRAL CALL
# =========================
def call_mistral(prompt: str) -> str:
    client = Mistral(api_key=os.environ.get("MISTRAL_API_KEY"))
    response = client.chat.complete(
        model="mistral-small-latest",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content


# =========================
# NOTATION CONVERTER
# =========================
def notation_to_derived(pattern_str: str):
    s = pattern_str.strip()

    m = re.match(r'^(prefix|suffix)\(([\w\s]+),\s*(\d+)\)$', s)
    if m:
        return f"{m.group(1)}_{m.group(3)}__{m.group(2).strip()}"

    m = re.match(r'^(first_token|last_token)\(([\w\s]+)\)$', s)
    if m:
        return f"{m.group(1)}__{m.group(2).strip()}"

    m = re.match(r'^domain\(([\w\s]+)\)$', s)
    if m:
        return f"domain__{m.group(1).strip()}"

    m = re.match(r'^area_code\(([\w\s]+)\)$', s)
    if m:
        return f"area_code__{m.group(1).strip()}"

    return None


# =========================
# PROMPT
# =========================
def build_initial_prompt(df):
    cols_info = "\n".join(
        f"- {col}: {df[col].dropna().astype(str).head(5).tolist()}"
        for col in df.columns
    )

    return f"""
Tu es un expert en data quality.

On cherche des PFDs :
pattern(col) → col

Transformations :
- prefix(col,k)
- suffix(col,k)
- first_token(col)
- last_token(col)
- domain(col)
- area_code(col)

Colonnes :
{cols_info}

Retourne EXACTEMENT 5 règles au format JSON :
[
  {{"lhs": "prefix(col,k)", "rhs": "col"}}
]
"""


# =========================
# PARSER SAFE
# =========================
def parse_llm_response(response: str):
    try:
        match = re.search(r'\[.*\]', response, re.DOTALL)
        if not match:
            return []

        data = json.loads(match.group())

        if isinstance(data, dict):
            return [data]
        if isinstance(data, list):
            return data

        return []
    except:
        return []


# =========================
# AGENT
# =========================
class AgentInLoopPFDDiscovery:

    def __init__(self, df, min_support=0.5, min_confidence=0.9):

        self.df = df.copy()

        generator = CandidateGenerator(df)
        generator._build_enriched_table()

        self.enriched_df = generator.enriched_df
        self.derived_set = {d for d, _ in generator.derived_cols}

        self.min_support = min_support
        self.min_confidence = min_confidence

        self.validated = []

    # -------------------------
    def _convert(self, raw):
        candidates = []

        for r in raw:
            if "lhs" not in r or "rhs" not in r:
                continue

            lhs = notation_to_derived(r["lhs"])
            rhs = r["rhs"]

            if lhs is None:
                continue
            if lhs not in self.derived_set:
                continue
            if rhs not in self.df.columns:
                continue

            candidates.append(((lhs,), rhs))

        return candidates

    # -------------------------
    def _validate(self, candidates):

        if not candidates:
            return []

        validator = PFDValidator(self.df)

        df_valid = validator.validate_candidates(
            candidates,
            enriched_df=self.enriched_df,
            min_support=self.min_support,
            min_confidence=self.min_confidence
        )

        results = []

        if df_valid is None or len(df_valid) == 0:
            return results

        for _, r in df_valid.iterrows():
            results.append({
                "LHS": _format_lhs(r["lhs_cols"]),
                "RHS": r["rhs_col"],
                "Support": r["support"],
                "Confidence": r["confidence"],
                "Score": r["support"] * r["confidence"]
            })

        return results

    # -------------------------
    def discover(self, n_iterations=3):

        print("\nSTART AGENT (MISTRAL)")

        for i in range(n_iterations):

            prompt = build_initial_prompt(self.df)
            response = call_mistral(prompt)

            raw = parse_llm_response(response)
            candidates = self._convert(raw)

            valid = self._validate(candidates)

            self.validated.extend(valid)

            print(f"Iteration {i+1} → {len(valid)} valid PFDs")

        # 🔥 IMPORTANT : TOUJOURS DATAFRAME
        if len(self.validated) == 0:
            return pd.DataFrame(columns=[
                "LHS", "RHS", "Support", "Confidence", "Score"
            ])

        return pd.DataFrame(self.validated).sort_values(
            "Score",
            ascending=False
        ).reset_index(drop=True)