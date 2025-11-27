# %% 
import numpy as np
import scanpy as sc
import matplotlib.pyplot as plt
import os

# Set working directory to the script's location
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Ensure output directories exist
# We are in grn/data/ now
os.makedirs("../results", exist_ok=True)

results_file = "./paul15.h5ad"

# Load dataset
adata = sc.datasets.paul15()
adata.X = adata.X.astype("float64")
print(adata)


# %% 
# Preprocessing
sc.pp.recipe_zheng17(adata)
sc.tl.pca(adata, svd_solver="arpack")
sc.pp.neighbors(adata, n_neighbors=4, n_pcs=20)
sc.tl.draw_graph(adata)

# Optional: Denoising (skipping for now based on tutorial note saying it's optional, but keeping diffusion map as it's used for DPT later)
sc.tl.diffmap(adata)
sc.pp.neighbors(adata, n_neighbors=10, use_rep="X_diffmap")
sc.tl.draw_graph(adata)

# Clustering and PAGA
sc.tl.louvain(adata, resolution=1.0)
adata.obs["louvain_anno"] = adata.obs["louvain"].cat.rename_categories(
    {
        "16": "16/Stem",
        "10": "10/Ery",
        "19": "19/Neu",
        "20": "20/Mk",
        "22": "22/Baso",
        "24": "24/Mo",
    }
)

sc.tl.paga(adata, groups="louvain_anno")
sc.pl.paga(adata, plot=False)  # Compute PAGA positions

# Recomputing embedding using PAGA-initialization
sc.tl.draw_graph(adata, init_pos="paga")

# Diffusion Pseudotime
if "16/Stem" in adata.obs["louvain_anno"].cat.categories:
    adata.uns["iroot"] = np.flatnonzero(adata.obs["louvain_anno"] == "16/Stem")[0]
    sc.tl.dpt(adata)
    print("Computed DPT with root 16/Stem")
else:
    print("Warning: 16/Stem cluster not found. Skipping DPT or using default root.")

# Save processed data
print(f"Saving processed data to {results_file}")
adata.write(results_file)

print("Done.")
print("Observations columns:", adata.obs.columns)


# %%
# Visualize PAGA results and embedding in one figure
fig, axs = plt.subplots(1, 2, figsize=(20, 8))

sc.pl.paga(adata, color=['louvain_anno'], show=False, ax=axs[0])
axs[0].set_title("PAGA")

sc.pl.draw_graph(adata, color=['louvain_anno'], legend_loc='on data', show=False, ax=axs[1])
axs[1].set_title("FA (ForceAtlas2) Embedding")

plt.tight_layout()
plt.savefig("../results/paul15_data_results.png", bbox_inches="tight")
plt.close()

print("Saved combined visualization to ../results/paul15_data_results.png")

# %%
