import pandas as pd
from itertools import combinations
from collections import defaultdict
import re
import time


# ============================================================================
# PATTERN EXTRACTOR
# ============================================================================

class PatternExtractor:

    PATTERNS = {

        # =========================================================
        # EMAIL / WEB
        # =========================================================
        'EMAIL': r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$',
        'URL': r'^(http|https)://[^\s]+$',

        # =========================================================
        # PHONE / CODES
        # =========================================================
        'PHONE_US': r'^\d{3}-\d{3}-\d{4}$',
        'PHONE_INTL': r'^\+?\d[\d\s\-]{7,15}$',
        'ZIP_CODE': r'^\d{5}(-\d{4})?$',
        'STATE_CODE': r'^[A-Z]{2}$',

        # =========================================================
        # DATES
        # =========================================================
        'DATE_YYYYMMDD': r'^\d{4}-\d{2}-\d{2}$',
        'DATE_DDMMYYYY': r'^\d{2}/\d{2}/\d{4}$',
        'DATE_MMDDYYYY': r'^\d{2}-\d{2}-\d{4}$',

        # =========================================================
        # TEXT
        # =========================================================
        'UPPERCASE': r'^[A-Z]+$',
        'LOWERCASE': r'^[a-z]+$',
        'CAPITALIZED': r'^[A-Z][a-z]+$',
        'WORDS_WITH_SPACE': r'^[A-Za-z ]+$',
        'JOB_TITLE': r'^[A-Za-z\s\-/&,\.]+$',
        'MULTI_WORD_TEXT': r'^[A-Za-z]+(?:\s[A-Za-z]+)+$',

        # =========================================================
        # NUMERIC
        # =========================================================
        'NUMERIC': r'^\d+$',
        'DECIMAL': r'^\d+\.\d+$',
        'ALPHANUMERIC': r'^[A-Za-z0-9]+$',
        'ALPHANUMERIC_SPACE': r'^[A-Za-z0-9 ]+$',

        # =========================================================
        # ADDRESSES
        # =========================================================
        'ADDRESS': r'^\d+\s+[A-Za-z0-9\s,.-]+$',

        # =========================================================
        # GENERAL
        # =========================================================
        'MIXED_TEXT': r'^[A-Za-z0-9\s,.\-_/&()]+$',
        'NON_EMPTY': r'^.+$',
    }

    def extract_patterns(self, values):

        clean_values = [
            str(v).strip()
            for v in values
            if pd.notna(v)
        ]

        if not clean_values:
            return {}

        pattern_matches = defaultdict(int)

        for value in clean_values:

            for pattern_name, pattern_regex in self.PATTERNS.items():

                try:

                    if re.match(pattern_regex, value):
                        pattern_matches[pattern_name] += 1

                except Exception:
                    pass

        total = len(clean_values)

        return {
            pattern: round(count / total, 4)
            for pattern, count in pattern_matches.items()
        }


# ============================================================================
# CANDIDATE GENERATOR
# ============================================================================

class CandidateGenerator:

    def __init__(self, df, min_pattern_support=0.3):

        self.df = df
        self.min_pattern_support = min_pattern_support
        self.extractor = PatternExtractor()
        self.all_patterns = {}

    def extract_all_patterns(self):

        print("\n[EXTRACTION DES PATTERNS]")

        for col in self.df.columns:

            try:

                patterns = self.extractor.extract_patterns(
                    self.df[col].values
                )

                self.all_patterns[col] = {
                    p: supp
                    for p, supp in patterns.items()
                    if supp >= self.min_pattern_support
                }

                print(
                    f"  ✓ {col}: "
                    f"{len(self.all_patterns[col])} patterns détectés"
                )

            except Exception as e:

                print(f"  ✗ Erreur colonne {col}: {e}")

    def generate_candidates(self, max_lhs_size=1):

        if not self.all_patterns:
            self.extract_all_patterns()

        print("\n[GÉNÉRATION DES CANDIDATS]")

        candidates = []

        columns = list(self.df.columns)

        for size in range(1, max_lhs_size + 1):

            for lhs_cols in combinations(columns, size):

                for rhs_col in columns:

                    if rhs_col not in lhs_cols:

                        candidates.append((lhs_cols, rhs_col))

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
        min_support=0.90,
        min_confidence=0.95
    ):

        print("\n[VALIDATION DES CANDIDATS]")
        print(
            f"  Seuils: support >= {min_support}, "
            f"confidence >= {min_confidence}"
        )

        validated = []

        total = len(candidates)

        for idx, (lhs_cols, rhs_col) in enumerate(candidates, 1):

            try:

                lhs_cols = list(lhs_cols)

                print(
                    f"  [{idx}/{total}] "
                    f"Test: {lhs_cols} -> {rhs_col}",
                    flush=True
                )

                # garder uniquement les colonnes utiles
                temp = self.df[lhs_cols + [rhs_col]].dropna()

                if temp.empty:
                    continue

                # nombre de valeurs RHS distinctes par groupe
                grouped = temp.groupby(lhs_cols)[rhs_col].nunique()

                if grouped.empty:
                    continue

                # groupes valides
                valid_groups = grouped[grouped == 1]

                # tailles des groupes
                group_sizes = temp.groupby(lhs_cols).size()

                # lignes valides
                valid_rows = sum(
                    group_sizes[group_name]
                    for group_name in valid_groups.index
                )

                # métriques
                confidence = valid_rows / len(temp)
                support = len(temp) / len(self.df)

                # validation
                if (
                    support >= min_support
                    and confidence >= min_confidence
                ):

                    validated.append({

                        'LHS': ", ".join(lhs_cols),

                        'RHS': rhs_col,

                        'Support': round(support, 4),

                        'Confidence': round(confidence, 4),

                        'Score': round(
                            (support + confidence) / 2,
                            4
                        ),

                        'Type': 'Classical'
                    })

            except Exception as e:

                print(f"Erreur candidat {idx}: {e}")

        print(f"\n  ✓ {len(validated)} PFDs validées trouvées")

        return validated


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
        min_support=0.90,
        min_confidence=0.95,
        max_lhs_size=1
    ):

        print("=" * 70)
        print("DÉCOUVERTE CLASSIQUE DE PFD")
        print("=" * 70)

        start_time = time.time()

        # génération candidats
        generator = CandidateGenerator(self.df)

        self.candidates = generator.generate_candidates(
            max_lhs_size=max_lhs_size
        )

        # validation
        validator = PFDValidator(self.df)

        self.validated_pfds = validator.validate_candidates(
            self.candidates,
            min_support=min_support,
            min_confidence=min_confidence
        )

        elapsed = time.time() - start_time

        print("\n[RÉSULTATS]")
        print(f"  Temps: {elapsed:.2f}s")
        print(f"  Candidats: {len(self.candidates)}")
        print(f"  PFDs: {len(self.validated_pfds)}")

        if self.validated_pfds:

            df_result = pd.DataFrame(self.validated_pfds)

            df_result = df_result.sort_values(
                by="Score",
                ascending=False
            )

            return df_result

        return pd.DataFrame()
