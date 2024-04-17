import os
from glob import glob
from collections import defaultdict

import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import (
    davies_bouldin_score,
    silhouette_score,
    calinski_harabasz_score,
)
from sklearn.decomposition import PCA

from osl_dynamics.utils import plotting, set_random_seed
from osl_dynamics.inference import tf_ops
from osl_dynamics.models import hive
from osl_dynamics.data import Data
from osl_dynamics.analysis import spectral, power, connectivity

tf_ops.gpu_growth()

set_random_seed(0)


def match_subjects():
    """Match the demographics distribution of each dataset."""

    # Get subject ids from each dataset
    notts_subjects = [
        os.path.basename(f)[-6:]
        for f in sorted(glob("/well/woolrich/projects/mrc_meguk/notts/ec/src/sub-*"))
        if os.path.exists(f + "/sflip_parc-raw.fif")
    ]
    camcan_subjects = [
        os.path.basename(f)
        for f in sorted(glob("/well/woolrich/projects/camcan/winter23/src/sub*"))
        if os.path.exists(f + "/sflip_parc.npy")
    ]

    # Load the demographic data
    notts_demo = pd.read_csv(
        "/well/woolrich/projects/mrc_meguk/raw/Demographics/nottingham_demographics_anonymised.csv"
    )
    notts_demo.rename({"gender": "sex"}, inplace=True, axis=1)
    camcan_demo = pd.read_csv(
        "/well/woolrich/projects/camcan/participants.tsv", sep="\t"
    )

    # Match the demographics
    notts_demo = notts_demo[notts_demo["id"].isin(notts_subjects)]
    notts_groups = notts_demo.groupby(["age_range", "sex"]).size()

    # Propcess the camcan demographics
    # turn sex to lower case and bin the ages
    camcan_demo["sex"] = camcan_demo["sex"].map(str.lower)
    camcan_demo = camcan_demo[camcan_demo["participant_id"].isin(camcan_subjects)]
    bins = [18, 24, 30, 36, 42, 48, 54, 60, 100]
    labels = ["{0}-{1}".format(bins[i], bins[i + 1]) for i in range(len(bins) - 2)] + [
        "60+"
    ]
    camcan_demo["age_range"] = pd.cut(
        camcan_demo["age"], bins, right=False, labels=labels
    )

    # Match the demographics (Cam-CAN has more subjects than notts in all groups)
    camcan_matched_subjects = []
    for age in labels[:-1]:
        for sex in ["male", "female"]:
            n_subjects = notts_groups[age][sex]
            camcan_matched_subjects.extend(
                camcan_demo[
                    np.all(
                        [camcan_demo["age_range"] == age, camcan_demo["sex"] == sex],
                        axis=0,
                    )
                ]
                .participant_id.iloc[:n_subjects]
                .to_list()
            )

    # There's no male subjects in the 60+ group in notts
    n_subjects = notts_groups["60+"]["female"]
    camcan_matched_subjects.extend(
        camcan_demo[
            np.all(
                [camcan_demo["age_range"] == "60+", camcan_demo["sex"] == "female"],
                axis=0,
            )
        ]
        .participant_id.iloc[:n_subjects]
        .to_list()
    )

    # Now slice the camcan demo to only include the matched subjects
    camcan_demo = camcan_demo[
        camcan_demo["participant_id"].isin(camcan_matched_subjects)
    ]

    df = pd.DataFrame(columns=["id", "dataset", "age_range", "sex", "data_path"])
    df["id"] = notts_demo["id"].to_list() + camcan_demo["participant_id"].to_list()
    df["dataset"] = ["notts"] * len(notts_demo) + ["camcan"] * len(camcan_demo)
    df["age_range"] = (
        notts_demo["age_range"].to_list() + camcan_demo["age_range"].to_list()
    )
    df.age_range = pd.Categorical(df.age_range, categories=labels, ordered=True)
    df["sex"] = notts_demo["sex"].to_list() + camcan_demo["sex"].to_list()

    notts_data_dir = "/well/woolrich/projects/mrc_meguk/notts/ec/src"
    camcan_data_dir = "/well/woolrich/projects/camcan/winter23/src"
    df["data_path"] = [
        f"{notts_data_dir}/sub-{s}/sflip_parc-raw.fif"
        for s in notts_demo["id"].to_list()
    ] + [
        f"{camcan_data_dir}/{s}/sflip_parc.npy"
        for s in camcan_demo["participant_id"].to_list()
    ]
    return df


def load_data(store_dir, use_tfrecord=True, buffer_size=2000, n_jobs=16):
    """Load data."""
    df = match_subjects()
    training_data = Data(
        df.data_path.to_list(),
        picks="misc",
        sampling_frequency=250,
        reject_by_annotation="omit",
        use_tfrecord=use_tfrecord,
        buffer_size=buffer_size,
        n_jobs=n_jobs,
        store_dir=store_dir,
    )
    methods = {
        "tde_pca": {"n_embeddings": 15, "n_pca_components": 80},
        "standardize": {},
    }
    training_data.prepare(methods)
    return training_data


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
plotting.plot_alpha(
    alpha[0],
    n_samples=2000,
    cmap="tab10",
    filename=f"{plot_dir}/alpha.png",
)

embeddings = np.load(f"{inf_params_dir}/summed_embeddings.npy")


def plot_embeddings(embeddings):
    se_cosine = squareform(pdist(embeddings, metric="cosine"))
    fig, ax = plotting.plot_matrices(
        se_cosine,
        cmap="coolwarm",
    )
    ax[0][0].set_xticks(
        ticks=[64 // 2, 64, 64 + 64 // 2],
        labels=["Nottingham", "", "Cam-CAN"],
        fontsize=15,
    )
    ax[0][0].set_yticks(
        ticks=[64 // 2, 64, 64 + 64 // 2],
        labels=["Nottingham", "", "Cam-CAN"],
        fontsize=15,
    )
    plt.setp(ax[0][0].get_yticklabels(), rotation=90, va="center")
    fig.savefig(f"{plot_dir}/se_cos.png")

    df = match_subjects()
    # Normalise and PCA
    pca = PCA(n_components=2)
    embeddings -= embeddings.mean(axis=0)
    embeddings /= embeddings.std(axis=0)
    pca_embeddings = pca.fit_transform(embeddings)

    # Get the centroids of the clusters
    centroids = np.array(
        [
            np.mean(embeddings[df["dataset"] == "notts"], axis=0),
            np.mean(embeddings[df["dataset"] == "camcan"], axis=0),
        ]
    )
    pca_centroids = pca.transform(centroids)

    # 2D plot
    sns.set_palette("colorblind", n_colors=2)
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.scatter(
        pca_embeddings[df["dataset"] == "notts", 0],
        pca_embeddings[df["dataset"] == "notts", 1],
        label="Nottingham",
    )
    ax.scatter(
        pca_embeddings[df["dataset"] == "camcan", 0],
        pca_embeddings[df["dataset"] == "camcan", 1],
        label="Cam-CAN",
    )
    ax.scatter(
        pca_centroids[:, 0],
        pca_centroids[:, 1],
        color="black",
        marker="*",
        s=100,
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{plot_dir}/se_2d.png", dpi=300)
    plt.close()

    sns.set_palette("viridis", n_colors=8)
    fig, ax = plt.subplots(figsize=(5, 3))
    for age_range in [
        "18-24",
        "24-30",
        "30-36",
        "36-42",
        "42-48",
        "48-54",
        "54-60",
        "60+",
    ]:
        ax.scatter(
            pca_embeddings[df.age_range == age_range, 0],
            pca_embeddings[df.age_range == age_range, 1],
            label=age_range,
        )
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{plot_dir}/se_2d_age.png", dpi=300)
    plt.close()
    sns.reset_defaults()


plot_embeddings(embeddings)

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


def plot_dataset_networks_diff(embeddings, f, psd, nnmf):
    """Plot the difference in networks between datasets."""

    def _get_nearest_neighbours(candidates, target, n_neighbours):
        """Get the nearest neighbours."""
        distances = np.linalg.norm(target[None, ...] - candidates, axis=1)
        return np.argsort(distances)[:n_neighbours]

    df = match_subjects()

    # Normalise and PCA
    embeddings -= embeddings.mean(axis=0)
    embeddings /= embeddings.std(axis=0)

    # Get the centroids of the clusters
    centroids = np.array(
        [
            np.mean(embeddings[df["dataset"] == "notts"], axis=0),
            np.mean(embeddings[df["dataset"] == "camcan"], axis=0),
        ]
    )

    # Get the nearest neighbours of the centroids
    notts_neighbours = _get_nearest_neighbours(
        candidates=embeddings,
        target=centroids[0],
        n_neighbours=10,
    )
    camcan_neighbours = _get_nearest_neighbours(
        candidates=embeddings,
        target=centroids[1],
        n_neighbours=10,
    )
    cluster_gpsd = [
        np.average(psd[notts_neighbours], axis=0),
        np.average(psd[camcan_neighbours], axis=0),
    ]

    notts_power_map = power.variance_from_spectra(f, cluster_gpsd[0], nnmf)
    camcan_power_map = power.variance_from_spectra(f, cluster_gpsd[1], nnmf)
    notts_minus_camcan_power_map = notts_power_map - camcan_power_map

    power.save(
        notts_minus_camcan_power_map,
        mask_file=training_data.mask_file,
        parcellation_file=training_data.parcellation_file,
        subtract_mean=False,
        filename=f"{plot_dir}/pow_diff.png",
        combined=True,
        titles=[f"Mode {i+1}" for i in range(model.config.n_states)],
        plot_kwargs={"views": ["lateral"]},
    )

    # Difference in PSDs
    P_notts = cluster_gpsd[0]
    P_camcan = cluster_gpsd[1]

    P_notts_minus_camcan = P_notts - P_camcan
    P_mean = np.mean(P_notts_minus_camcan, axis=1)
    P_std = np.std(P_notts_minus_camcan, axis=1)
    fig, axes = plt.subplots(1, P_notts_minus_camcan.shape[0], figsize=(20, 5))
    for j in range(P_notts_minus_camcan.shape[0]):
        plotting.plot_line(
            [f],
            [P_mean[j]],
            errors=[[P_mean[j] + P_std[j]], [P_mean[j] - P_std[j]]],
            x_range=[0, 45],
            y_range=[-0.03, 0.02],
            ax=axes[j],
        )
        axes[j].axhline(0, color="k", linestyle="--")
        axes[j].set_xticklabels(axes[j].get_xticks(), fontsize=15)
        axes[j].set_yticklabels(axes[j].get_yticks().round(2), fontsize=15)

    fig.tight_layout()
    fig.savefig(f"{plot_dir}/old_vs_young_psd.png", dpi=300)


plot_dataset_networks_diff(embeddings, f, psd, nnmf)

clustering_scores = {
    "hmm": defaultdict(list),
    "hive": defaultdict(list),
}
subject_labels = [0] * 64 + [1] * 64

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
