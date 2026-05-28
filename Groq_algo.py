import os
import re
import json
import time
import pandas as pd
from dotenv import load_dotenv
from groq import Groq

from classical_pfd_algorithm import CandidateGenerator, PFDValidator

load_dotenv()


# =========================
# GROQ CALL
# =========================
def call_groq(prompt: str, max_retries=5, base_delay=5) -> str:
    client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",  # modèle gratuit Groq
                messages=[{"role": "user", "content": prompt}]
            )
            return response.choices[0].message.content

        except Exception as e:
            error_str = str(e).lower()

            # Erreur de rate limit ou serveur indisponible → réessai
            if "rate_limit" in error_str or "503" in error_str or "unavailable" in error_str:
                if attempt < max_retries - 1:
                    wait = base_delay * (2 ** attempt)  # 5s, 10s, 20s, 40s...
                    print(f"  ⚠️  Groq busy, retrying in {wait}s... (attempt {attempt+1}/{max_retries})")
                    time.sleep(wait)
                else:
                    print(f"  ❌ Groq failed after {max_retries} attempts: {e}")
                    raise
            else:
                raise


# =========================
# NOTATION CONVERTER
# =========================
def notation_to_derived(pattern_str: str):
    """Convertit la notation texte en nom de colonne dérivée"""
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
def build_initial_prompt(df, history_rules=None):
    """Construit le prompt pour Groq"""
    cols_info = "\n".join(
        f"- {col}: {df[col].dropna().astype(str).head(3).tolist()}"
        for col in df.columns
    )

    prompt = f"""Tu es un expert en data quality et en Pattern Functional Dependencies (PFDs).

Colonnes disponibles:
{cols_info}

Patterns disponibles:
- prefix(col_name, k): les k premiers caractères
- suffix(col_name, k): les k derniers caractères
- first_token(col_name): premier mot/token
- last_token(col_name): dernier mot/token
- domain(col_name): domaine email (texte après @)
- area_code(col_name): codes d'accès (3 premiers chiffres)

Une PFD est: pattern(col) → target_col

Tâche: Retourne entre 10 et 15 PFDs valides au format JSON.

Règles:
1. Chaque pattern doit utiliser une colonne qui existe
2. Chaque target_col doit exister
3. Les PFDs doivent être réalistes
4. Évite les PFDs triviales

OBLIGATOIRE: Retourne UNIQUEMENT du JSON, pas d'explications.

Format exact:
[
  {{"lhs": "prefix(colname, 2)", "rhs": "targetcol"}},
  {{"lhs": "first_token(colname)", "rhs": "targetcol"}}
]
"""

    if history_rules:
        prompt += "\n\nPFDs DÉJÀ TROUVÉES (à éviter):\n"
        for rule in history_rules[:5]:
            prompt += f"- {rule['lhs']} → {rule['rhs']}\n"
        prompt += "\nPROPOSE DES PFDs DIFFÉRENTES!\n"

    return prompt


# =========================
# PARSER SAFE
# =========================
def parse_llm_response(response: str):
    """Extrait le JSON de la réponse Groq"""
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
    except Exception as e:
        print(f"[ERROR] Parsing JSON failed: {e}")
        return []


# =========================
# AGENT
# =========================
class AgentInLoopPFDDiscovery:

    def __init__(self, df, min_support=0.1, min_confidence=0.75):
        """Initialise l'agent avec un dataframe

        Args:
            df: DataFrame à analyser
            min_support: Support minimum (0.0 à 1.0) - défaut 0.1 (10%)
            min_confidence: Confidence minimum (0.0 à 1.0) - défaut 0.75 (75%)
        """
        self.df = df.copy()

        # Construire la table enrichie avec les patterns
        generator = CandidateGenerator(df)
        generator._build_enriched_table()

        self.enriched_df = generator.enriched_df
        self.derived_set = {d for d, _ in generator.derived_cols}

        self.min_support = min_support
        self.min_confidence = min_confidence

        # Historique des PFDs trouvées
        self.validated = []
        self.history = []
        self.seen_pairs = set()

    # -------------------------
    def _convert(self, raw):
        """Convertit les règles brutes en candidats"""
        candidates = []

        for r in raw:
            if "lhs" not in r or "rhs" not in r:
                continue

            lhs = notation_to_derived(r["lhs"])
            rhs = r["rhs"].strip()

            if lhs is None:
                continue
            if lhs not in self.derived_set:
                continue
            if rhs not in self.df.columns:
                continue

            candidates.append(((lhs,), rhs))

        return candidates

    # -------------------------
    def is_trivial(self, lhs, rhs):
        """Détecte les PFDs triviales"""
        lhs_str = str(lhs)

        if "RowID" in lhs_str or "rowid" in lhs_str.lower():
            return True

        if lhs_str == rhs:
            return True

        return False

    # -------------------------
    def _validate(self, candidates):
        """Valide les candidats PFDs"""
        if not candidates:
            return []

        validator = PFDValidator(self.df)

        valid_list = validator.validate_candidates(
            candidates,
            enriched_df=self.enriched_df,
            min_support=self.min_support,
            min_confidence=self.min_confidence
        )

        results = []

        for item in valid_list:
            if self.is_trivial(item["LHS"], item["RHS"]):
                continue

            results.append({
                "LHS": item["LHS"],
                "RHS": item["RHS"],
                "Support": item["Support"],
                "Confidence": item["Confidence"],
                "Score": item["Score"]
            })

        return results

    # -------------------------
    def discover(self, n_iterations=3):
        """Lance la découverte de PFDs

        Args:
            n_iterations: Nombre d'itérations Groq

        Returns:
            DataFrame avec les PFDs trouvées (colonnes: LHS, RHS, Support, Confidence, Score)
        """

        print("\n" + "=" * 70)
        print("GROQ PFD DISCOVERY")
        print(f"min_support={self.min_support}, min_confidence={self.min_confidence}")
        print("=" * 70)

        for iteration in range(n_iterations):

            print(f"\n[Iteration {iteration + 1}/{n_iterations}]")

            # Construire le prompt avec historique
            prompt = build_initial_prompt(
                self.df,
                self.history if self.history else None
            )

            # Appel Groq
            print("  Calling Groq...")
            response = call_groq(prompt)

            # Délai entre itérations pour respecter les limites de débit
            if iteration < n_iterations - 1:
                time.sleep(4)  # ~15 req/min max

            # Parser la réponse
            raw = parse_llm_response(response)
            print(f"  → Extracted {len(raw)} rules from JSON")

            # Convertir en candidats
            candidates = self._convert(raw)
            print(f"  → {len(candidates)} valid candidates")

            # Valider les candidats
            valid = self._validate(candidates)
            print(f"  → {len(valid)} passed validation")

            # Ajouter au résultat (éviter les doublons)
            added = 0
            for v in valid:
                pair = (v["LHS"], v["RHS"])

                if pair not in self.seen_pairs:
                    self.seen_pairs.add(pair)
                    self.validated.append(v)

                    self.history.append({
                        "lhs": v["LHS"],
                        "rhs": v["RHS"],
                        "score": v["Score"]
                    })

                    added += 1
                    print(f"    ✓ {v['LHS']} → {v['RHS']} (Score: {v['Score']:.4f})")

            print(f"  Result: {added} new PFDs found (Total: {len(self.validated)})")

        # ============================================================
        # RETOUR FINAL
        # ============================================================
        print(f"\n{'=' * 70}")
        print(f"FINAL RESULT: {len(self.validated)} unique PFDs discovered")
        print(f"{'=' * 70}")

        if not self.validated:
            return pd.DataFrame(columns=[
                "LHS", "RHS", "Support", "Confidence", "Score"
            ])

        df_result = pd.DataFrame(self.validated).sort_values(
            "Score",
            ascending=False
        ).reset_index(drop=True)

        return df_result