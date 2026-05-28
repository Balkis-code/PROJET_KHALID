import sys
import os
import pandas as pd
import time

# =========================
# CONFIG
# =========================
project_path = r"C:\Users\manso\Downloads\PROJET_KHALID\data"
sys.path.insert(0, project_path)

from classical_pfd_algorithm import ClassicalPFDDiscovery

DATA_BASE_PATH = r"C:\Users\manso\Downloads\PROJET_KHALID\data"

# =========================
# AFFICHAGE
# =========================
print("\n" + "=" * 80)
print("SOURCE 1: DGOV - APPROCHE CLASSIQUE (TOUS LES FICHIERS)")
print("=" * 80)

# =========================
# FICHIERS
# =========================
dgov_files = [
    os.path.join(DATA_BASE_PATH, "DGOV", "570-1.csv"),
    os.path.join(DATA_BASE_PATH, "DGOV", "6339-1.csv"),
    os.path.join(DATA_BASE_PATH, "DGOV", "6397-1.csv"),
    os.path.join(DATA_BASE_PATH, "DGOV", "10492-1.csv"),
    os.path.join(DATA_BASE_PATH, "DGOV", "10642-1.csv"),
]

results = []

# =========================
# BOUCLE PRINCIPALE
# =========================
for i, file_path in enumerate(dgov_files, 1):

    file_name = os.path.basename(file_path)

    print(f"\n[{i}/5] {file_name}")

    try:
        # =========================
        # CHARGEMENT
        # =========================
        print("  Chargement du fichier...", flush=True)

        df = pd.read_csv(
            file_path,
            low_memory=False,
            on_bad_lines='skip'
        )

        # garder seulement les colonnes texte
        df = df.select_dtypes(include=['object', 'string'])

        # supprimer colonnes totalement vides
        df = df.dropna(axis=1, how='all')

        # convertir en string
        df = df.astype(str)

        # =========================
        # SOLUTION ANTI-LENTEUR
        # =========================
      

        # =========================
        # ANALYSE
        # =========================
        print("  Initialisation algorithme...", flush=True)

        discovery = ClassicalPFDDiscovery(df)

        print("  Lancement discover()...", flush=True)

        start = time.time()

        pfds = discovery.discover(
            min_support=0.50,
            min_confidence=0.90,
            max_lhs_size=1
        )

        elapsed = time.time() - start

        print("  discover() terminé", flush=True)

        # =========================
        # RÉSULTATS
        # =========================
        num_pfds = len(pfds)

        print(f"  ✓ {num_pfds} PFDs trouvées")
        print(f"  ✓ Temps: {elapsed:.2f}s")

        # ✅ AFFICHER LES PFDs DÉCOUVERTES
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

# =========================
# RÉSUMÉ FINAL
# =========================
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