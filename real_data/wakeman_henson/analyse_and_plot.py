import os
from glob import glob
from collections import defaultdict

import numpy as np
import pickle
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import (
    davies_bouldin_score,
    silhouette_score,
    calinski_harabasz_score,
)

from osl_dynamics.utils import plotting, set_random_seed
from osl_dynamics.inference import tf_ops
from osl_dynamics.models import hive
from osl_dynamics.data import Data
from osl_dynamics.analysis import spectral, power, connectivity

tf_ops.gpu_growth()

set_random_seed(0)


def get_best_dimension(dimensions, significance_level=0.01):
    best_dim = dimensions[0]
    fe_1 = [
        pickle.load(open(f"results/hive_{best_dim}/run{i}/model/history.pkl", "rb"))[
            "free_energy"
        ]
        for i in range(1, 11)
    ]

    for i in range(len(dimensions) - 1):
        dim = dimensions[i + 1]
        fe_2 = [
            pickle.load(open(f"results/hive_{dim}/run{i}/model/history.pkl", "rb"))[
                "free_energy"
            ]
            for i in range(1, 11)
        ]
        pvalue = ttest_ind(fe_1, fe_2, alternative="greater", permutations=5000).pvalue
        print(f"p-value for {best_dim} vs {dim}: {pvalue}")
        if pvalue < significance_level:
            best_dim = dim
            fe_1 = fe_2

    print(f"Best dimension: {best_dim}")
    return best_dim


def get_best_hive_run(dim):
    best_fe = np.Inf
    for run in range(1, 11):
        history = pickle.load(
            open(f"results/hive_{dim}/run{run}/model/history.pkl", "rb")
        )
        if history["loss"][-1] < best_fe:
            best_fe = history["loss"][-1]
            best_run = run

    print(f"Best HIVE run: {best_run}")
    return best_run


def get_best_hmm_run():
    best_fe = np.Inf
    for run in range(1, 11):
        history = pickle.load(open(f"results/hmm/run{run}/model/history.pkl", "rb"))
        if history["loss"][-1] < best_fe:
            best_fe = history["loss"][-1]
            best_run = run

    print(f"Best HMM run: {best_run}")
    return best_run


def load_data(use_tfrecord=True, buffer_size=2000, n_jobs=16):
    """Load the data."""

    data_paths = sorted(
        glob(
            "/well/woolrich/projects/wakeman_henson/spring23/src/sub*_run*/sflip_parc-raw.fif"
        )
    )
    training_data = Data(
        data_paths,
        sampling_frequency=250,
        mask_file="MNI152_T1_8mm_brain.nii.gz",
        parcellation_file="fmri_d100_parcellation_with_PCC_reduced_2mm_ss5mm_ds8mm.nii.gz",
        picks="misc",
        reject_by_annotation="omit",
        use_tfrecord=use_tfrecord,
        buffer_size=buffer_size,
        n_jobs=n_jobs,
    )

    return training_data


best_dim = get_best_dimension([3, 5, 10, 20, 30])
best_hive_run = get_best_hive_run(best_dim)
hive_loss = pickle.load(
    open(f"results/hive_{best_dim}/run{best_hive_run}/model/history.pkl", "rb")
)["loss"]

best_hmm_run = get_best_hmm_run()
hmm_loss = pickle.load(open(f"results/hmm/run{best_hmm_run}/model/history.pkl", "rb"))[
    "loss"
]
model = hive.Model.load(f"results/hive_{best_dim}/run{best_hive_run}/model")
inf_params_dir = f"results/hive_{best_dim}/run{best_hive_run}/inf_params"

best_run_dir = f"results/best_run"
os.makedirs(best_run_dir, exist_ok=True)

plot_dir = f"{best_run_dir}/plots"
os.makedirs(plot_dir, exist_ok=True)

plotting.plot_line(
    [range(len(hive_loss)), range(len(hmm_loss))],
    [hive_loss, hmm_loss],
    labels=["HIVE", "HMM-DE"],
    filename=f"{plot_dir}/loss.png",
)

alpha = pickle.load(open(f"{inf_params_dir}/alp.pkl", "rb"))
embeddings = np.load(f"{inf_params_dir}/summed_embeddings.npy")

plotting.plot_alpha(
    alpha[0],
    n_samples=2000,
    cmap="tab10",
    filename=f"{plot_dir}/alpha.png",
)

# Pairwise correlation between subject embeddings
se_cosine = squareform(pdist(embeddings, metric="cosine"))
fig, ax = plotting.plot_matrices(
    se_cosine,
    cmap="coolwarm",
)
ax[0][0].set_xticks(
    ticks=np.arange(0, 114, 6) + 3, labels=[f"{i + 1}" for i in range(19)]
)
ax[0][0].set_yticks(
    ticks=np.arange(0, 114, 6) + 3, labels=[f"{i + 1}" for i in range(19)]
)
fig.savefig(f"{plot_dir}/embeddings_cosine.png")


training_data = load_data(n_jobs=16)
trimmed_data = training_data.trim_time_series(
    sequence_length=200,
    n_embeddings=15,
    prepared=False,
)
spectra = spectral.multitaper_spectra(
    data=trimmed_data,
    alpha=alpha,
    sampling_frequency=250,
    n_jobs=16,
)
pickle.dump(spectra, open(f"{inf_params_dir}/spectra.pkl", "wb"))

f, psd, coh = pickle.load(open(f"{inf_params_dir}/spectra.pkl", "rb"))
nnmf = spectral.decompose_spectra(coh, n_components=2)
np.save(f"{inf_params_dir}/nnmf_2.npy", nnmf)

f, psd, coh = pickle.load(open(f"{inf_params_dir}/spectra.pkl", "rb"))
frequency_range = [1, 45]
n_components = nnmf.shape[0]
plotting.plot_line(
    [f] * n_components,
    nnmf,
    labels=[f"Component {i}" for i in range(n_components)],
    x_label="Frequency (Hz)",
    y_label="Weighting",
)

# Calculate group average
gpsd = np.average(psd, axis=0)
gcoh = np.average(coh, axis=0)

# Calculate average PSD across channels and the standard error
p = np.mean(gpsd, axis=-2)
e = np.std(gpsd, axis=-2) / np.sqrt(gpsd.shape[-2])

# Plot PSDs
n_states = gpsd.shape[0]
fig, axes = plt.subplots(1, 6, figsize=(36, 6))
for i in range(model.config.n_states):
    plotting.plot_line(
        [f],
        [p[i]],
        errors=[[p[i] - e[i]], [p[i] + e[i]]],
        labels=[f"State {i + 1}"],
        x_range=[f[0], f[-1]],
        y_range=[p.min() - 0.1 * p.max(), 1.2 * p.max()],
        x_label="Frequency (Hz)",
        y_label="PSD (a.u.)",
        ax=axes[i],
    )
    axes[i].axvspan(
        frequency_range[0],
        frequency_range[1],
        alpha=0.25,
        color="gray",
    )
fig.savefig(f"{plot_dir}/spectra_psd.png")

gp = power.variance_from_spectra(f, gpsd, nnmf)
power.save(
    gp,
    mask_file=training_data.mask_file,
    parcellation_file=training_data.parcellation_file,
    subtract_mean=True,
    show_plots=False,
    filename=f"{plot_dir}/psd.png",
    combined=True,
    titles=[f"Mode {i+1}" for i in range(model.config.n_states)],
    plot_kwargs={"views": ["lateral"]},
)

gc = connectivity.mean_coherence_from_spectra(f, gcoh, nnmf)
gc = connectivity.threshold(gc, percentile=97, subtract_mean=True)
connectivity.save(
    gc,
    parcellation_file=training_data.parcellation_file,
    combined=True,
    titles=[f"Mode {i+1}" for i in range(model.config.n_states)],
    filename=f"{plot_dir}/coh.png",
)

clustering_scores = {
    "hmm": defaultdict(list),
    "hive": defaultdict(list),
}
subject_labels = np.repeat(np.arange(19), 6)

for model in ["hmm", "hive"]:
    model_dir = f"results/{model}"
    for run in range(1, 11):
        if model == "hive":
            covs = np.load(
                f"{model_dir}_{best_dim}/run{run}/inf_params/session_covs.npy"
            )
        else:
            covs = np.load(f"{model_dir}/run{run}/dual_estimates/covs.npy")

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

best_hmm_covs = np.load(f"results//hmm/run{best_hmm_run}/dual_estimates/covs.npy")
best_hive_covs = np.load(
    f"results/hive_{best_dim}/run{best_hive_run}/inf_params/session_covs.npy"
)

best_hmm_covs_flatten = np.array([cov.flatten() for cov in best_hmm_covs])
best_hive_covs_flatten = np.array([cov.flatten() for cov in best_hive_covs])

# Get the pairwise distances
hmm_pdist = squareform(pdist(best_hmm_covs_flatten, metric="euclidean"))
hive_pdist = squareform(pdist(best_hive_covs_flatten, metric="euclidean"))

fig, ax = plotting.plot_matrices(
    [
        hive_pdist,
        hmm_pdist,
    ],
    titles=["HIVE", "HMM-DE"],
    cmap="coolwarm",
)
ax[0][0].set_xticks(
    ticks=np.arange(0, 114, 6) + 3,
    labels=[f"{i + 1}" for i in range(19)],
    fontsize=6,
)
ax[0][0].set_yticks(
    ticks=np.arange(0, 114, 6) + 3,
    labels=[f"{i + 1}" for i in range(19)],
    fontsize=6,
)
plt.setp(ax[0][0].get_xticklabels(), rotation=45)
ax[0][1].set_xticks(
    ticks=np.arange(0, 114, 6) + 3,
    labels=[f"{i + 1}" for i in range(19)],
    fontsize=6,
)
ax[0][1].set_yticks(
    ticks=np.arange(0, 114, 6) + 3,
    labels=[f"{i + 1}" for i in range(19)],
    fontsize=6,
)
plt.setp(ax[0][1].get_xticklabels(), rotation=45)
fig.savefig(f"{plot_dir}/covs_pairwise_distances.png")
