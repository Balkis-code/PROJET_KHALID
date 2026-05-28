import os
import pandas as pd
import time

from Groq_algo import AgentInLoopPFDDiscovery

# =========================
# PATH
# =========================
DATA_BASE_PATH = r"C:\Users\manso\Downloads\PROJET_KHALID\data"

dgov_files = [
    os.path.join(DATA_BASE_PATH, "DGOV", "570-1.csv"),
    os.path.join(DATA_BASE_PATH, "DGOV", "6339-1.csv"),
    os.path.join(DATA_BASE_PATH, "DGOV", "6397-1.csv"),
    os.path.join(DATA_BASE_PATH, "DGOV", "10492-1.csv"),
    os.path.join(DATA_BASE_PATH, "DGOV", "10642-1.csv"),
]

# =========================
all_results = []

print("\n" + "=" * 80)
print("DGOV BATCH - Groq_discovery")
print("=" * 80)

start_total = time.time()

for i, file_path in enumerate(dgov_files):

    print("\n" + "-" * 80)
    print(f"FILE: {os.path.basename(file_path)}")
    print("-" * 80)

    try:
        df = pd.read_csv(file_path, nrows=300, on_bad_lines="skip", dtype=str)
        df = df.dropna(axis=1, how="all")

        start = time.time()

        agent = AgentInLoopPFDDiscovery(
            df,
            min_support=0.5,
            min_confidence=0.9
        )

        result = agent.discover(n_iterations=3)

        elapsed = round(time.time() - start, 2)

        if not result.empty:
            result = result.copy()
            result["file"] = os.path.basename(file_path)
            all_results.append(result)

        print(f"✔ Done in {elapsed}s")
        print(f"PFD found: {len(result)}")

    except Exception as e:
        print(f"❌ Error in {file_path}: {e}")

    # Délai entre chaque fichier pour éviter le rate limit
    if i < len(dgov_files) - 1:
        print("\n⏳ Waiting 20s before next file...")
        time.sleep(20)

# =========================
# FINAL RESULTS
# =========================
if all_results:
    final_df = pd.concat(all_results, ignore_index=True)

    print("\n" + "=" * 80)
    print("FINAL Groq DGOV RESULTS")
    print("=" * 80)

    final_df = final_df.sort_values("Confidence", ascending=False)

    print(final_df)

    final_df.to_csv("dgov_Groq_results.csv", index=False)

else:
    print("❌ No results found")

print("\nTOTAL TIME:", round(time.time() - start_total, 2))