# ================================
# FULL NOTEBOOK-STYLE CODE (NO CLI)
# ================================
# What it does:
# 1) Loads your .h5ad (AnnData)
# 2) Chooses expression source (counts layer OR raw OR X)
# 3) Normalizes + log1p if needed
# 4) Ranks marker genes per group (cluster/domain/etc.)
# 5) Saves CSVs + optional plots
#
# ✅ You only need to edit TWO lines:
#   H5AD_PATH = "..."
#   GROUPBY   = "..."

import os
import numpy as np
import pandas as pd
import scanpy as sc

# ----------------
# EDIT THESE
# ----------------
H5AD_PATH = "E:\Project_Large_Datasets\ST\Stereoseq\\FB2_D1_stereo-seq_processed_updated.h5ad"   # <-- put your file path here
GROUPBY   = "annotation"           # <-- e.g. "domain", "cluster", "leiden", "annotation", "layer"
OUTDIR    = "markers_out"      # output folder name

# ----------------
# OPTIONAL SETTINGS
# ----------------
COUNTS_LAYER = None            # e.g. "counts" if adata.layers["counts"] exists; else keep None
METHOD       = "wilcoxon"      # "wilcoxon" (best default), "t-test", "logreg"
N_GENES_SAVE = 200             # how many genes to compute/store per group
TOP_N_EXPORT = 20              # top N markers per group in separate CSV
TARGET_SUM   = 1e4             # normalize_total target
MAKE_PLOTS   = True            # saves rank plot + dotplot PNGs

# ----------------
# Helper functions
# ----------------
def _is_sparse(x):
    try:
        import scipy.sparse as sp
        return sp.issparse(x)
    except Exception:
        return False

def _sample_values(X, n=20000, seed=0):
    rng = np.random.default_rng(seed)
    if _is_sparse(X):
        data = X.data
    else:
        data = np.asarray(X).ravel()
    if data.size == 0:
        return np.array([], dtype=float)
    if data.size <= n:
        return data.astype(float, copy=False)
    idx = rng.choice(data.size, size=n, replace=False)
    return data[idx].astype(float, copy=False)

def looks_like_counts(X):
    """Heuristic: counts are non-negative, often integer-like, sometimes with large max."""
    vals = _sample_values(X, n=20000, seed=1)
    if vals.size == 0:
        return True
    vmin = float(np.min(vals))
    vmax = float(np.max(vals))
    if vmin < -1e-6:
        return False
    if vmax >= 50:
        return True
    int_like = np.mean(np.isclose(vals, np.round(vals)))
    if int_like > 0.98 and vmax >= 20:
        return True
    return False

def ensure_log_normalized(adata, target_sum=1e4):
    """If X looks like counts, normalize_total + log1p. Then store in adata.raw."""
    if looks_like_counts(adata.X):
        print("[INFO] adata.X looks like counts -> normalize_total + log1p")
        sc.pp.normalize_total(adata, target_sum=target_sum)
        sc.pp.log1p(adata)
    else:
        print("[INFO] adata.X does NOT look like raw counts -> using as-is")
    adata.raw = adata.copy()
    return adata

# ----------------
# Load
# ----------------
print("[INFO] Reading:", H5AD_PATH)
adata = sc.read_h5ad(H5AD_PATH)
adata.var_names_make_unique()

print("[INFO] AnnData:", adata)
print("[INFO] obs columns:", list(adata.obs.columns))
print("[INFO] layers:", list(adata.layers.keys()))
print("[INFO] raw exists?", adata.raw is not None)

# ----------------
# Pick expression matrix source
# ----------------
use_raw_for_ranking = False

if COUNTS_LAYER is not None:
    if COUNTS_LAYER not in adata.layers:
        raise ValueError(f"COUNTS_LAYER='{COUNTS_LAYER}' not found in adata.layers: {list(adata.layers.keys())}")
    print(f"[INFO] Using adata.layers['{COUNTS_LAYER}'] as counts -> adata.X")
    adata.X = adata.layers[COUNTS_LAYER].copy()
    adata = ensure_log_normalized(adata, target_sum=TARGET_SUM)
    use_raw_for_ranking = True

else:
    # If adata.raw exists, prefer it (common in pipelines)
    if adata.raw is not None:
        print("[INFO] Using adata.raw for ranking (assuming it is log-normalized).")
        use_raw_for_ranking = True
    else:
        # Otherwise use X; normalize if it looks like counts
        adata = ensure_log_normalized(adata, target_sum=TARGET_SUM)
        use_raw_for_ranking = True

# ----------------
# Validate groupby
# ----------------
if GROUPBY not in adata.obs.columns:
    raise ValueError(f"GROUPBY='{GROUPBY}' not found in adata.obs columns: {list(adata.obs.columns)}")

adata.obs[GROUPBY] = adata.obs[GROUPBY].astype("category")
print(f"[INFO] Groups in '{GROUPBY}':", list(adata.obs[GROUPBY].cat.categories))

# ----------------
# Rank marker genes
# ----------------
print(f"[INFO] Running rank_genes_groups (method={METHOD}, n_genes={N_GENES_SAVE}, use_raw={use_raw_for_ranking})")

try:
    sc.tl.rank_genes_groups(
        adata,
        groupby=GROUPBY,
        method=METHOD,
        n_genes=N_GENES_SAVE,
        use_raw=use_raw_for_ranking,
        pts=True
    )
except TypeError:
    # Older scanpy versions may not support pts=
    sc.tl.rank_genes_groups(
        adata,
        groupby=GROUPBY,
        method=METHOD,
        n_genes=N_GENES_SAVE,
        use_raw=use_raw_for_ranking
    )

# Convert results to dataframe
df_all = sc.get.rank_genes_groups_df(adata, group=None)

# ----------------
# Save outputs
# ----------------
os.makedirs(OUTDIR, exist_ok=True)

all_csv = os.path.join(OUTDIR, "markers_all.csv")
df_all.to_csv(all_csv, index=False)
print("[INFO] Saved:", all_csv)

# Top N per group
if "scores" in df_all.columns:
    df_top = (
        df_all.sort_values(["group", "scores"], ascending=[True, False])
              .groupby("group", as_index=False)
              .head(TOP_N_EXPORT)
    )
else:
    df_top = df_all.groupby("group", as_index=False).head(TOP_N_EXPORT)

top_csv = os.path.join(OUTDIR, f"markers_top{TOP_N_EXPORT}.csv")
df_top.to_csv(top_csv, index=False)
print("[INFO] Saved:", top_csv)

# Print a preview
print("\n===== TOP MARKERS (preview) =====")
print(df_top.head(50))

# ----------------
# Optional plots
# ----------------
if MAKE_PLOTS:
    import matplotlib.pyplot as plt

    # Rank genes plot
    try:
        sc.pl.rank_genes_groups(adata, n_genes=TOP_N_EXPORT, sharey=False, show=False)
        plt.tight_layout()
        p1 = os.path.join(OUTDIR, f"rank_genes_groups_top{TOP_N_EXPORT}.png")
        plt.savefig(p1, dpi=200)
        plt.close()
        print("[INFO] Saved:", p1)
    except Exception as e:
        print("[WARN] rank_genes_groups plot failed:", e)

    # Dotplot of union of top genes (cap at 200 genes to keep it readable)
    try:
        genes_union = list(pd.unique(df_top["names"]))
        genes_union = genes_union[: min(200, len(genes_union))]
        dp = sc.pl.dotplot(
            adata,
            var_names=genes_union,
            groupby=GROUPBY,
            use_raw=use_raw_for_ranking,
            show=False
        )
        p2 = os.path.join(OUTDIR, f"dotplot_union_top{TOP_N_EXPORT}.png")
        dp.savefig(p2, dpi=200)
        print("[INFO] Saved:", p2)
    except Exception as e:
        print("[WARN] dotplot failed:", e)

print("[DONE]")
