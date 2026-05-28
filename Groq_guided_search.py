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
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}]
            )
            return response.choices[0].message.content

        except Exception as e:
            error_str = str(e).lower()
            if "rate_limit" in error_str or "503" in error_str or "unavailable" in error_str:
                if attempt < max_retries - 1:
                    wait = base_delay * (2 ** attempt)
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


def derived_to_notation(derived_col: str) -> str:
    """Convertit un nom de colonne dérivée en notation lisible pour le LLM"""
    # prefix_3__colname → prefix(colname, 3)
    m = re.match(r'^(prefix|suffix)_(\d+)__(.+)$', derived_col)
    if m:
        return f"{m.group(1)}({m.group(3)}, {m.group(2)})"

    # first_token__colname → first_token(colname)
    m = re.match(r'^(first_token|last_token)__(.+)$', derived_col)
    if m:
        return f"{m.group(1)}({m.group(2)})"

    # domain__colname → domain(colname)
    m = re.match(r'^domain__(.+)$', derived_col)
    if m:
        return f"domain({m.group(1)})"

    # area_code__colname → area_code(colname)
    m = re.match(r'^area_code__(.+)$', derived_col)
    if m:
        return f"area_code({m.group(1)})"

    return derived_col


# =========================
# PROMPTS GUIDED SEARCH
# =========================
def build_ranking_prompt(df, candidates: list, top_k: int = 30) -> str:
    """
    Guided Search : on donne TOUS les candidats générés par l'algo
    au LLM et on lui demande de sélectionner les top_k les plus prometteurs.
    """
    cols_info = "\n".join(
        f"- {col}: {df[col].dropna().astype(str).head(3).tolist()}"
        for col in df.columns
    )

    # Convertir les candidats en notation lisible
    candidates_str = "\n".join(
        f"{i+1}. {derived_to_notation(lhs[0])} → {rhs}"
        for i, (lhs, rhs) in enumerate(candidates)
    )

    prompt = f"""Tu es un expert en data quality et en Pattern Functional Dependencies (PFDs).

CONTEXTE DES DONNÉES:
{cols_info}

LISTE DES CANDIDATS PFDs GÉNÉRÉS PAR L'ALGORITHME ({len(candidates)} candidats):
{candidates_str}

TÂCHE: Sélectionne les {top_k} PFDs les plus prometteuses parmi cette liste.

CRITÈRES DE SÉLECTION (par ordre de priorité):
1. Pertinence sémantique : le pattern LHS a un sens logique par rapport à la colonne RHS
2. Réalisme métier : la dépendance est plausible dans un contexte réel
3. Non-trivialité : évite les relations évidentes ou tautologiques
4. Diversité : couvre des colonnes et patterns variés

OBLIGATOIRE: Retourne UNIQUEMENT un tableau JSON avec les numéros des candidats sélectionnés.
Format exact (liste d'indices, base 1):
[3, 7, 12, 15, 21, ...]

Retourne exactement {top_k} indices ou moins si la liste est plus courte.
"""
    return prompt


def build_refinement_prompt(df, validated: list, remaining_candidates: list, top_k: int = 20) -> str:
    """
    Phase 2 : le LLM connaît déjà les PFDs validées,
    il guide la recherche vers de nouvelles zones non explorées.
    """
    cols_info = "\n".join(
        f"- {col}: {df[col].dropna().astype(str).head(3).tolist()}"
        for col in df.columns
    )

    validated_str = "\n".join(
        f"- {derived_to_notation(v['LHS'][0] if isinstance(v['LHS'], tuple) else v['LHS'])} → {v['RHS']} (score={v['Score']:.3f})"
        for v in validated[:10]
    )

    remaining_str = "\n".join(
        f"{i+1}. {derived_to_notation(lhs[0])} → {rhs}"
        for i, (lhs, rhs) in enumerate(remaining_candidates)
    )

    prompt = f"""Tu es un expert en data quality et en Pattern Functional Dependencies (PFDs).

CONTEXTE DES DONNÉES:
{cols_info}

PFDs DÉJÀ VALIDÉES (à ne pas répéter):
{validated_str}

CANDIDATS RESTANTS À EXPLORER ({len(remaining_candidates)} candidats):
{remaining_str}

TÂCHE: Sélectionne les {top_k} candidats restants les plus susceptibles d'être de nouvelles PFDs valides.

CRITÈRES:
1. Complémentarité : couvre des colonnes/patterns pas encore explorés
2. Pertinence sémantique avec les données
3. Non-redondance avec les PFDs déjà trouvées

OBLIGATOIRE: Retourne UNIQUEMENT un tableau JSON d'indices (base 1).
Format: [2, 5, 11, ...]
"""
    return prompt


# =========================
# PARSER
# =========================
def parse_indices_response(response: str, max_index: int) -> list:
    """Extrait la liste d'indices retournée par Groq"""
    try:
        match = re.search(r'\[[\d,\s]+\]', response)
        if not match:
            return []

        indices = json.loads(match.group())
        # Filtrer les indices valides (base 1 → base 0)
        valid = [i - 1 for i in indices if isinstance(i, int) and 1 <= i <= max_index]
        return valid

    except Exception as e:
        print(f"[ERROR] Parsing indices failed: {e}")
        return []


def parse_llm_response(response: str):
    """Extrait le JSON de la réponse Groq (fallback générique)"""
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
# GUIDED SEARCH
# =========================
class GuidedSearchPFDDiscovery:
    """
    Guided Search PFD Discovery
    ───────────────────────────
    Stratégie en 2 phases :

    Phase 1 — Génération & Priorisation
      L'algorithme classique génère TOUS les candidats possibles.
      Le LLM (Groq) analyse les candidats et sélectionne les top_k
      les plus prometteurs selon leur pertinence sémantique.

    Phase 2 — Raffinement
      Après validation des candidats prioritaires, le LLM guide
      une seconde exploration des candidats restants en tenant
      compte des PFDs déjà validées pour maximiser la diversité.

    Avantages vs Agent in Loop :
      - Pas d'hallucination : le LLM ne "invente" pas de PFDs,
        il choisit parmi des candidats réels générés par l'algo
      - Plus efficace : on valide uniquement les candidats prometteurs
      - Meilleure couverture : la phase 2 évite les zones déjà explorées
    """

    def __init__(self, df, min_support=0.1, min_confidence=0.75,
                 top_k_phase1=30, top_k_phase2=20):
        """
        Args:
            df: DataFrame à analyser
            min_support: Support minimum (défaut 0.1)
            min_confidence: Confidence minimum (défaut 0.75)
            top_k_phase1: Nombre de candidats sélectionnés en phase 1
            top_k_phase2: Nombre de candidats sélectionnés en phase 2
        """
        self.df = df.copy()

        # Générer TOUS les candidats via l'algo classique
        print("  [Init] Generating all candidates...")
        generator = CandidateGenerator(df)
        generator._build_enriched_table()

        self.enriched_df = generator.enriched_df
        self.derived_set = {d for d, _ in generator.derived_cols}

        # Construire la liste complète des candidats (lhs_tuple, rhs)
        self.all_candidates = self._generate_all_candidates(generator)

        self.min_support = min_support
        self.min_confidence = min_confidence
        self.top_k_phase1 = top_k_phase1
        self.top_k_phase2 = top_k_phase2

        self.validated = []
        self.seen_pairs = set()

        print(f"  [Init] {len(self.all_candidates)} total candidates generated")

    # -------------------------
    def _generate_all_candidates(self, generator) -> list:
        """Génère tous les candidats possibles (derived_col → original_col)"""
        candidates = []
        original_cols = list(self.df.columns)
        derived_cols = [d for d, _ in generator.derived_cols]

        for lhs in derived_cols:
            for rhs in original_cols:
                # Éviter trivialités évidentes
                if rhs.lower() in lhs.lower():
                    continue
                if "rowid" in lhs.lower() or "rowid" in rhs.lower():
                    continue
                candidates.append(((lhs,), rhs))

        return candidates

    # -------------------------
    def _validate(self, candidates: list) -> list:
        """Valide une liste de candidats PFDs"""
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
            lhs_str = str(item["LHS"])
            if "RowID" in lhs_str or "rowid" in lhs_str.lower():
                continue
            if lhs_str == item["RHS"]:
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
    def _add_results(self, valid: list) -> int:
        """Ajoute les PFDs validées en évitant les doublons. Retourne le nombre ajouté."""
        added = 0
        for v in valid:
            pair = (str(v["LHS"]), v["RHS"])
            if pair not in self.seen_pairs:
                self.seen_pairs.add(pair)
                self.validated.append(v)
                added += 1
                print(f"    ✓ {v['LHS']} → {v['RHS']} (Score: {v['Score']:.4f})")
        return added

    # -------------------------
    def discover(self) -> pd.DataFrame:
        """
        Lance la découverte guidée en 2 phases.

        Returns:
            DataFrame avec les PFDs trouvées (LHS, RHS, Support, Confidence, Score)
        """
        print("\n" + "=" * 70)
        print("GROQ GUIDED SEARCH — PFD DISCOVERY")
        print(f"min_support={self.min_support}, min_confidence={self.min_confidence}")
        print(f"Candidates pool: {len(self.all_candidates)}")
        print("=" * 70)

        remaining = list(self.all_candidates)

        # ══════════════════════════════════════════════
        # PHASE 1 — Priorisation initiale par le LLM
        # ══════════════════════════════════════════════
        print(f"\n[Phase 1] LLM prioritizes top {self.top_k_phase1} candidates...")

        prompt1 = build_ranking_prompt(self.df, remaining, top_k=self.top_k_phase1)
        print("  Calling Groq...")
        response1 = call_groq(prompt1)
        time.sleep(4)  # respect rate limit

        indices1 = parse_indices_response(response1, len(remaining))
        print(f"  → LLM selected {len(indices1)} candidates")

        # Récupérer les candidats sélectionnés
        if indices1:
            selected1 = [remaining[i] for i in indices1 if i < len(remaining)]
        else:
            # Fallback : prendre les top_k premiers si le LLM échoue
            print("  ⚠️  Fallback: taking first candidates")
            selected1 = remaining[:self.top_k_phase1]

        # Retirer les candidats sélectionnés du pool restant
        selected_set1 = set(indices1)
        remaining = [c for i, c in enumerate(remaining) if i not in selected_set1]

        # Valider les candidats de phase 1
        print(f"  Validating {len(selected1)} candidates...")
        valid1 = self._validate(selected1)
        print(f"  → {len(valid1)} passed validation")
        added1 = self._add_results(valid1)
        print(f"  [Phase 1] Result: {added1} PFDs found")

        # ══════════════════════════════════════════════
        # PHASE 2 — Raffinement guidé par les résultats
        # ══════════════════════════════════════════════
        if remaining:
            print(f"\n[Phase 2] LLM refines search from {len(remaining)} remaining candidates...")

            prompt2 = build_refinement_prompt(
                self.df,
                self.validated,
                remaining,
                top_k=self.top_k_phase2
            )
            print("  Calling Groq...")
            response2 = call_groq(prompt2)
            time.sleep(4)

            indices2 = parse_indices_response(response2, len(remaining))
            print(f"  → LLM selected {len(indices2)} candidates")

            if indices2:
                selected2 = [remaining[i] for i in indices2 if i < len(remaining)]
            else:
                print("  ⚠️  Fallback: taking next candidates")
                selected2 = remaining[:self.top_k_phase2]

            print(f"  Validating {len(selected2)} candidates...")
            valid2 = self._validate(selected2)
            print(f"  → {len(valid2)} passed validation")
            added2 = self._add_results(valid2)
            print(f"  [Phase 2] Result: {added2} new PFDs found")

        # ══════════════════════════════════════════════
        # RETOUR FINAL
        # ══════════════════════════════════════════════
        print(f"\n{'=' * 70}")
        print(f"FINAL RESULT: {len(self.validated)} unique PFDs discovered")
        print(f"{'=' * 70}")

        if not self.validated:
            return pd.DataFrame(columns=["LHS", "RHS", "Support", "Confidence", "Score"])

        df_result = pd.DataFrame(self.validated).sort_values(
            "Score", ascending=False
        ).reset_index(drop=True)

        return df_result