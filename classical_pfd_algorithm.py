import pandas as pd
from itertools import combinations
from collections import defaultdict
import re
import time


# ============================================================================
# PATTERN EXTRACTOR
# ============================================================================

class PatternExtractor:

    PREFIX_LENGTHS = [2, 3, 4, 5]
    SUFFIX_LENGTHS = [2, 3, 4]

    def prefix(self, series, k):
        return series.astype(str).str[:k].where(
            series.notna() & (series.astype(str).str.len() >= k)
        )

    def suffix(self, series, k):
        return series.astype(str).str[-k:].where(
            series.notna() & (series.astype(str).str.len() >= k)
        )

    def first_token(self, series):
        def _ft(v):
            if pd.isna(v): return None
            parts = str(v).split()
            return parts[0] if parts else None
        return series.map(_ft)

    def last_token(self, series):
        def _lt(v):
            if pd.isna(v): return None
            parts = str(v).split()
            return parts[-1] if parts else None
        return series.map(_lt)

    def domain(self, series):
        def _dom(v):
            if pd.isna(v): return None
            s = str(v)
            return s.split("@", 1)[1].lower() if "@" in s else None
        return series.map(_dom)

    def area_code(self, series):
        def _ac(v):
            if pd.isna(v): return None
            digits = re.sub(r"\D", "", str(v))
            return digits[:3] if len(digits) >= 3 else None
        return series.map(_ac)

    def extract_patterns(self, series, col_name):
        results = []

        def _add(name, derived):
            non_null = derived.dropna()
            if non_null.nunique() >= 2:
                results.append((name, derived))

        for k in self.PREFIX_LENGTHS:
            _add(f"prefix_{k}__{col_name}", self.prefix(series, k))

        for k in self.SUFFIX_LENGTHS:
            _add(f"suffix_{k}__{col_name}", self.suffix(series, k))

        _add(f"first_token__{col_name}", self.first_token(series))
        _add(f"last_token__{col_name}",  self.last_token(series))
        _add(f"domain__{col_name}",      self.domain(series))
        _add(f"area_code__{col_name}",   self.area_code(series))

        return results


# ============================================================================
# TABLE ENRICHER  (utilisé par mistral_pfd_algo.py)
# ============================================================================

class TableEnricher:

    def __init__(self):
        self.extractor = PatternExtractor()
        self.derived_cols = []   # liste de (derived_name, source_col)

    def enrich(self, df):
        extras = {}
        self.derived_cols = []

        for col in df.columns:
            series = df[col]
            if series.nunique() <= 1:
                continue

            pairs = self.extractor.extract_patterns(series, col)
            for derived_name, derived_series in pairs:
                extras[derived_name] = derived_series.values
                self.derived_cols.append((derived_name, col))

        enriched = df.copy()
        for name, vals in extras.items():
            enriched[name] = vals

        return enriched


# ============================================================================
# CANDIDATE GENERATOR
# ============================================================================

class CandidateGenerator:

    def __init__(self, df, min_pattern_support=0.0):
        self.df = df
        self.min_pattern_support = min_pattern_support
        self.extractor = PatternExtractor()
        self.enriched_df = None
        self.derived_cols = []

    def _build_enriched_table(self):
        print("\n[EXTRACTION DES PATTERNS]")
        extras = {}

        for col in self.df.columns:
            series = self.df[col]
            if series.nunique() <= 1:
                continue

            pairs = self.extractor.extract_patterns(series, col)
            for derived_name, derived_series in pairs:
                extras[derived_name] = derived_series.values
                self.derived_cols.append((derived_name, col))

            print(f"  ✓ {col}: {len(pairs)} patterns dérivés")

        self.enriched_df = self.df.copy()
        for name, vals in extras.items():
            self.enriched_df[name] = vals

        print(f"  → Table enrichie : {len(self.derived_cols)} colonnes dérivées")

    def generate_candidates(self, max_lhs_size=1):
        if self.enriched_df is None:
            self._build_enriched_table()

        print("\n[GÉNÉRATION DES CANDIDATS]")

        original_cols = list(self.df.columns)
        derived_names = [d for d, _ in self.derived_cols]
        source_of = {d: s for d, s in self.derived_cols}

        candidates = []

        for size in range(1, max_lhs_size + 1):
            for lhs_combo in combinations(derived_names, size):
                lhs_sources = {source_of[d] for d in lhs_combo}
                for rhs_col in original_cols:
                    if rhs_col not in lhs_sources:
                        candidates.append((lhs_combo, rhs_col))

        print(f"  ✓ {len(candidates)} candidats générés")
        return candidates


# ============================================================================
# PFD VALIDATOR
# ============================================================================

class PFDValidator:

    def __init__(self, df):
        self.df = df

    def validate_candidates(
        self,
        candidates,
        enriched_df,
        min_support=0.50,
        min_confidence=0.90
    ):
        print("\n[VALIDATION DES CANDIDATS]")
        print(f"  Seuils: support >= {min_support}, confidence >= {min_confidence}")

        total = len(self.df)
        validated = []

        for idx, (lhs_tuple, rhs_col) in enumerate(candidates, 1):

            try:
                cols_needed = list(lhs_tuple) + [rhs_col]
                missing = [c for c in cols_needed if c not in enriched_df.columns]
                if missing:
                    continue

                # Ignorer les colonnes RHS trivialement constantes (faux positifs)
                if enriched_df[rhs_col].nunique(dropna=True) < 2:
                    continue

                temp = enriched_df[cols_needed].dropna(subset=list(lhs_tuple))

                if temp.empty:
                    continue

                covered = len(temp)
                support = covered / total

                if support < min_support:
                    continue

                if len(lhs_tuple) == 1:
                    grouped = temp.groupby(lhs_tuple[0], sort=False)
                else:
                    temp = temp.copy()
                    temp["__lhs_key__"] = list(zip(*(temp[c] for c in lhs_tuple)))
                    grouped = temp.groupby("__lhs_key__", sort=False)

                consistent_rows = 0
                for _, group in grouped:
                    if group[rhs_col].nunique(dropna=False) == 1:
                        consistent_rows += len(group)

                confidence = consistent_rows / covered

                if confidence < min_confidence:
                    continue

                validated.append({
                    'LHS':        _format_lhs(lhs_tuple),
                    'RHS':        rhs_col,
                    'Support':    round(support, 4),
                    'Confidence': round(confidence, 4),
                    'Score':      round(support * confidence, 4),
                    'Noise':      round(1.0 - confidence, 4),
                    'Type':       'PFD'
                })

            except Exception as e:
                print(f"  Erreur candidat {idx}: {e}")

        print(f"\n  ✓ {len(validated)} PFDs validées trouvées")
        return validated   # ← retourne une LISTE (pas un DataFrame)


# ============================================================================
# HELPER
# ============================================================================

def _format_lhs(lhs_tuple):
    return ", ".join(_format_single(d) for d in lhs_tuple)

def _format_single(derived_name):
    m = re.match(r"^(prefix|suffix)_(\d+)__(.+)$", derived_name)
    if m:
        return f"{m.group(1)}({m.group(3)}, {m.group(2)})"

    m = re.match(r"^(first_token|last_token)__(.+)$", derived_name)
    if m:
        return f"{m.group(1)}({m.group(2)})"

    m = re.match(r"^domain__(.+)$", derived_name)
    if m:
        return f"domain({m.group(1)})"

    m = re.match(r"^area_code__(.+)$", derived_name)
    if m:
        return f"area_code({m.group(1)})"

    return derived_name


# ============================================================================
# MAIN CLASS
# ============================================================================

class ClassicalPFDDiscovery:

    def __init__(self, df):
        self.df = df
        self.candidates = []
        self.validated_pfds = []

    def discover(
        self,
        min_support=0.5,
        min_confidence=0.9,
        max_lhs_size=1
    ):
        print("=" * 70)
        print("DÉCOUVERTE CLASSIQUE DE PFD")
        print("=" * 70)

        start_time = time.time()

        generator = CandidateGenerator(self.df)
        self.candidates = generator.generate_candidates(
            max_lhs_size=max_lhs_size
        )

        validator = PFDValidator(self.df)
        self.validated_pfds = validator.validate_candidates(
            self.candidates,
            enriched_df=generator.enriched_df,
            min_support=min_support,
            min_confidence=min_confidence
        )

        elapsed = time.time() - start_time

        print("\n[RÉSULTATS]")
        print(f"  Temps:      {elapsed:.2f}s")
        print(f"  Candidats:  {len(self.candidates)}")
        print(f"  PFDs:       {len(self.validated_pfds)}")

        if self.validated_pfds:   # fonctionne correctement sur une liste
            return pd.DataFrame(self.validated_pfds).sort_values(
                by="Score", ascending=False
            )

        return pd.DataFrame()
