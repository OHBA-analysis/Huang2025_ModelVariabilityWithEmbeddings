import os
from collections import defaultdict

import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import (
    davies_bouldin_score,
    silhouette_score,
    calinski_harabasz_score,
)
from scipy.spatial.distance import pdist, squareform

from osl_dynamics.models.sehmm import Model
from osl_dynamics.utils import plotting

base_dir = "results"
plot_dir = "figures"

clustering_scores = {
    "hmm": defaultdict(list),
    "sehmm": defaultdict(list),
}
subject_labels = np.repeat(np.arange(19), 6)
fe = {
    "hmm": [],
    "sehmm": [],
}

for model in ["hmm", "sehmm"]:
    model_dir = f"{base_dir}/{model}/"
    n_runs = len(os.listdir(model_dir))
    for run in range(1, n_runs + 1):
        fe[model].append(
            pickle.load(open(f"{model_dir}/run{run}/model/history.pkl", "rb"))[
                "free_energy"
            ]
        )
        if model == "sehmm":
            sehmm_model = Model.load(f"{model_dir}/run{run}/model")
            covs = sehmm_model.get_subject_means_covariances()[1]
        else:
            covs = pickle.load(
                open(f"{model_dir}/run{run}/model/dual_estimation.pkl", "rb")
            )[1]

        covs_flatten = np.array([cov.flatten() for cov in covs])
        clustering_scores[model]["silhouette"].append(
            silhouette_score(covs_flatten, subject_labels)
        )
        clustering_scores[model]["davies_bouldin"].append(
            davies_bouldin_score(covs_flatten, subject_labels)
        )
        clustering_scores[model]["calinski_harabasz"].append(
            calinski_harabasz_score(covs_flatten, subject_labels)
        )

# plot results
plotting.plot_violin(
    np.array(
        [
            clustering_scores["sehmm"]["silhouette"],
            clustering_scores["hmm"]["silhouette"],
        ]
    ),
    ["SE-HMM", "HMM"],
    sns_kwargs={"cut": 0, "scale": "width"},
    filename=f"{base_dir}/silhouette_scores.png",
)

plotting.plot_violin(
    np.array(
        [
            clustering_scores["sehmm"]["davies_bouldin"],
            clustering_scores["hmm"]["davies_bouldin"],
        ]
    ),
    ["SE-HMM", "HMM"],
    sns_kwargs={"cut": 0, "scale": "width"},
    filename=f"{base_dir}/davies_bouldin_scores.png",
)

plotting.plot_violin(
    np.array(
        [
            clustering_scores["sehmm"]["calinski_harabasz"],
            clustering_scores["hmm"]["calinski_harabasz"],
        ]
    ),
    ["SE-HMM", "HMM"],
    sns_kwargs={"cut": 0, "scale": "width"},
    filename=f"{base_dir}/calinski_harabasz_scores.png",
)

# Get the best runs
best_hmm_run = np.argmin(fe["hmm"])
best_sehmm_run = np.argmin(fe["sehmm"])

best_hmm_covs = pickle.load(
    open(f"{base_dir}/hmm/run{best_hmm_run + 1}/model/dual_estimation.pkl", "rb")
)[1]
best_sehmm_model = Model.load(f"{base_dir}/sehmm/run{best_sehmm_run + 1}/model")
best_sehmm_covs = best_sehmm_model.get_subject_means_covariances()[1]

best_hmm_covs_flatten = np.array([cov.flatten() for cov in best_hmm_covs])
best_sehmm_covs_flatten = np.array([cov.flatten() for cov in best_sehmm_covs])

# Get the pairwise distances
hmm_pdist = squareform(pdist(best_hmm_covs_flatten, metric="euclidean"))
sehmm_pdist = squareform(pdist(best_sehmm_covs_flatten, metric="euclidean"))

fig, ax = plotting.plot_matrices(
    [
        sehmm_pdist,
        hmm_pdist,
    ],
    titles=["SE-HMM pairwise distances", "HMM pairwise distances"],
)
ax[0][0].set_xticks(
    ticks=np.arange(0, 114, 6) + 3,
    labels=[f"sub_{i + 1}" for i in range(19)],
    fontsize=7,
)
ax[0][0].set_yticks(
    ticks=np.arange(0, 114, 6) + 3,
    labels=[f"sub_{i + 1}" for i in range(19)],
    fontsize=7,
)
plt.setp(ax[0][0].get_xticklabels(), rotation=90)
ax[0][1].set_xticks(
    ticks=np.arange(0, 114, 6) + 3,
    labels=[f"sub_{i + 1}" for i in range(19)],
    fontsize=7,
)

plt.setp(ax[0][1].get_xticklabels(), rotation=90)
fig.savefig(f"{plot_dir}/pairwise_distances.png", dpi=300)
plt.close(fig)
