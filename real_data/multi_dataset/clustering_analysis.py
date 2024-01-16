import os
from collections import defaultdict

import numpy as np
from sklearn.metrics import (
    davies_bouldin_score,
    silhouette_score,
    calinski_harabasz_score,
)

from osl_dynamics.utils import plotting
from osl_dynamics.utils.misc import load

base_dir = "results"
plot_dir = "figures"
os.makedirs(plot_dir, exist_ok=True)

clustering_scores = {
    "hmm": defaultdict(list),
    "hive": defaultdict(list),
}
subject_labels = [0] * 64 + [1] * 64

for model in ["hmm", "hive"]:
    model_dir = f"{base_dir}/{model}"
    n_runs = len(os.listdir(model_dir))
    for run in range(1, n_runs + 1):
        if model == "hive":
            covs = load(f"{model_dir}/run{run}/inf_params/covs.npy")
        else:
            covs = load(f"{model_dir}/run{run}/dual_estimates/covs.npy")

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
            clustering_scores["hive"]["silhouette"],
            clustering_scores["hmm"]["silhouette"],
        ]
    ),
    ["HIVE", "HMM-DE"],
    title="Silhouette score",
    y_label="Score",
    sns_kwargs={"cut": 0, "scale": "width"},
    filename=f"{plot_dir}/silhouette_scores.png",
)

plotting.plot_violin(
    1
    - np.array(
        [
            clustering_scores["hive"]["davies_bouldin"],
            clustering_scores["hmm"]["davies_bouldin"],
        ]
    ),
    ["HIVE", "HMM-DE"],
    title="Negative Davies-Bouldin score",
    y_label="Score",
    sns_kwargs={"cut": 0, "scale": "width"},
    filename=f"{plot_dir}/davies_bouldin_scores.png",
)

plotting.plot_violin(
    np.array(
        [
            clustering_scores["hive"]["calinski_harabasz"],
            clustering_scores["hmm"]["calinski_harabasz"],
        ]
    ),
    ["HIVE", "HMM-DE"],
    title="Calinski-Harabasz score",
    y_label="Score",
    sns_kwargs={"cut": 0, "scale": "width"},
    filename=f"{plot_dir}/calinski_harabasz_scores.png",
)
