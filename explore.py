import pandas as pd
import numpy as np

pd.set_option("display.max_rows", 200)
pd.set_option("display.width", 160)

df = pd.read_csv("data/dataset_final.csv")
print("shape:", df.shape)
print()
print("columns:", list(df.columns))
print()

FEATURES = [
    "genre", "mood", "tempo", "language", "popularity_level",
    "duration_length", "danceability_level", "energy_level", "valence_level",
    "acoustic_level", "instrumental_level", "liveness_level", "speechiness_level",
    "loudness_level", "key_category", "mode_category", "time_signature_category",
    "content_rating", "artist_type", "release_type", "track_version"
]

print("=== 21 features: nunique + value_counts ===")
for f in FEATURES:
    vc = df[f].value_counts(dropna=False)
    print(f"\n--- {f} ({df[f].nunique()} unique) ---")
    print(vc)

print()
print("=== missing values across all columns ===")
print(df.isna().sum()[df.isna().sum() > 0])

print()
print("=== duplicate full-attribute-vector check (on 21 features) ===")
dupe_mask = df.duplicated(subset=FEATURES, keep=False)
print("rows sharing attr vector with >=1 other row:", dupe_mask.sum())
vc2 = df.groupby(FEATURES).size().sort_values(ascending=False)
print("distinct attribute vectors:", len(vc2))
print("largest equivalence class:", vc2.max())
print("top 5 classes:\n", vc2.head())

print()
print("=== raw continuous columns present? ===")
for c in ["danceability","energy","valence","acousticness","instrumentalness",
          "liveness","speechiness","loudness","popularity","duration_ms","tempo"]:
    print(c, c in df.columns, df[c].dtype if c in df.columns else None)
