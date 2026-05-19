import sys
import os
import pandas as pd
import time

project_path = r"C:\Users\manso\Downloads\PROJEKT_KHALID"
sys.path.insert(0, project_path)

from classical_pfd_algorithm import ClassicalPFDDiscovery

DATA_BASE_PATH = r"C:\Users\manso\Downloads\PROJET_KHALID\data"

print("\n" + "=" * 80)
print("SOURCE 2: CHE - APPROCHE CLASSIQUE (TOUS LES FICHIERS)")
print("=" * 80)

# ✅ AVEC EXTENSION .csv
che_files = [
    os.path.join(DATA_BASE_PATH, "CHE", "mechanism_refs.csv"),
    os.path.join(DATA_BASE_PATH, "CHE", "metabolism_refs.csv"),
    os.path.join(DATA_BASE_PATH, "CHE", "protein_classification.csv"),
    os.path.join(DATA_BASE_PATH, "CHE", "research_companies.csv"),
    os.path.join(DATA_BASE_PATH, "CHE", "variant_sequences.csv"),
]

results = []

for i, file_path in enumerate(che_files, 1):
    file_name = os.path.basename(file_path)
    print(f"\n[{i}/5] {file_name}")

    try:
        print("  Chargement du fichier...", flush=True)

        df = pd.read_csv(
            file_path,
            low_memory=False,
            on_bad_lines='skip'
        )

        df = df.select_dtypes(include=['object', 'string'])
        df = df.dropna(axis=1, how='all')
        df = df.astype(str)

        MAX_ROWS = 2000

        if len(df) > MAX_ROWS:
            print(f"  Dataset trop grand ({len(df)} lignes)")
            print(f"  Réduction à {MAX_ROWS} lignes pour accélérer")
            df = df.sample(n=MAX_ROWS, random_state=42)

        print(f"  ✓ {len(df)} lignes")
        print(f"  ✓ {len(df.columns)} colonnes")

        print("  Initialisation algorithme...", flush=True)
        discovery = ClassicalPFDDiscovery(df)

        print("  Lancement discover()...", flush=True)
        start = time.time()

        pfds = discovery.discover(
            min_support=0.95,
            min_confidence=0.95,
            max_lhs_size=1
        )

        elapsed = time.time() - start

        num_pfds = len(pfds)

        print(f"  ✓ {num_pfds} PFDs trouvées")
        print(f"  ✓ Temps: {elapsed:.2f}s")

        if not pfds.empty and num_pfds > 0:
            print("\n  📋 Détail des PFDs:")
            for idx, row in pfds.iterrows():
                lhs = row['LHS']
                rhs = row['RHS']
                support = row['Support']
                confidence = row['Confidence']
                print(f"    [{idx+1}] {lhs} → {rhs}")
                print(f"         Support: {support:.3f}, Confidence: {confidence:.3f}")
        else:
            print("  (Aucune PFD découverte)")

        results.append({
            'Fichier': file_name,
            'Lignes': len(df),
            'Colonnes': len(df.columns),
            'PFDs': num_pfds,
            'Temps': round(elapsed, 2)
        })

    except Exception as e:
        print("\n" + "!" * 60)
        print(f"ERREUR SUR {file_name}")
        print("!" * 60)
        print(type(e).__name__)
        print(e)
        continue

print("\n" + "=" * 80)
print("RÉSUMÉ FINAL")
print("=" * 80)

if results:
    df_res = pd.DataFrame(results)
    print("\n")
    print(df_res.to_string(index=False))

    print("\n" + "-" * 80)
    print("STATISTIQUES")
    print("-" * 80)

    print(f"Total PFDs découvertes: {df_res['PFDs'].sum()}")
    print(f"Temps moyen par fichier: {df_res['Temps'].mean():.2f}s")
    print(f"Temps total: {df_res['Temps'].sum():.2f}s")

else:
    print("Aucun résultat obtenu.")

print("\n" + "=" * 80)
print("EXÉCUTION TERMINÉE")
print("=" * 80)