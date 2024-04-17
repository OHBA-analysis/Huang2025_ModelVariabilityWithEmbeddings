import os
from glob import glob

import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind
import pandas as pd
from sklearn.decomposition import PCA

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


def load_data(data_dir, store_dir, use_tfrecord=True, buffer_size=2000, n_jobs=16):
    """Load the data."""

    data_paths = sorted(glob(f"{data_dir}/sub*/sflip_parc.npy"))
    training_data = Data(
        data_paths,
        sampling_frequency=250,
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


def get_demographics():
    """Get the demographics."""
    df = pd.DataFrame(columns=["data_path", "participant_id"])
    df["data_path"] = sorted(
        glob("/well/woolrich/projects/camcan/winter23/src/sub*/sflip_parc.npy")
    )
    df["participant_id"] = df["data_path"].apply(
        lambda x: os.path.basename(os.path.dirname(x))
    )
    demo_df = pd.read_csv("/well/woolrich/projects/camcan/participants.tsv", sep="\t")
    df = df.merge(demo_df, on="participant_id")

    categories = [
        "18-24",
        "24-30",
        "30-36",
        "36-42",
        "42-48",
        "48-54",
        "54-60",
        "60-66",
        "66-72",
        "72-78",
        "78-84",
        "84+",
    ]
    # Bin the age
    df["age_range"] = pd.cut(
        df["age"],
        bins=[18, 24, 30, 36, 42, 48, 54, 60, 66, 72, 78, 84, 100],
        labels=categories,
        right=False,
    )
    df["age_range"] = pd.Categorical(
        df["age_range"],
        categories=categories,
        ordered=True,
    )
    return df


best_dim = get_best_dimension([3, 5, 10, 20, 50, 100])
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


def plot_results(embeddings, f, psd, nnmf):
    """Plot the subject embeddings."""

    def _get_nearest_neighbours(candidates, target, n_neighbours):
        """Get the nearest neighbours."""
        distances = np.linalg.norm(target[None, ...] - candidates, axis=1)
        return np.argsort(distances)[:n_neighbours]

    embeddings -= embeddings.mean(axis=0)
    embeddings /= embeddings.std(axis=0)
    df = get_demographics()

    # Age categories
    categories = [
        "18-24",
        "24-30",
        "30-36",
        "36-42",
        "42-48",
        "48-54",
        "54-60",
        "60-66",
        "66-72",
        "72-78",
        "78-84",
        "84+",
    ]
    # Young and old centroids
    young_centroid = np.mean(embeddings[df.age_range.isin(categories[:3])], axis=0)
    old_centroid = np.mean(embeddings[df.age_range.isin(categories[-3:])], axis=0)

    # PCA
    pca = PCA(n_components=2)
    pca_subject_embeddings = pca.fit_transform(embeddings)
    pca_young_centroid = np.squeeze(pca.transform(young_centroid.reshape(1, -1)))
    pca_old_centroid = np.squeeze(pca.transform(old_centroid.reshape(1, -1)))

    # 2D plot
    df["pca_se_0"] = list(pca_subject_embeddings[:, 0])
    df["pca_se_1"] = list(pca_subject_embeddings[:, 1])
    fig, ax = plt.subplots()
    sns.scatterplot(
        data=df,
        x="pca_se_0",
        y="pca_se_1",
        hue="age_range",
        ax=ax,
        palette="viridis",
    )
    ax.plot(
        pca_young_centroid[0],
        pca_young_centroid[1],
        marker="*",
        color="red",
        markersize=20,
    )
    ax.plot(
        pca_old_centroid[0],
        pca_old_centroid[1],
        marker="*",
        color="red",
        markersize=20,
    )
    fig.savefig(f"{plot_dir}/2d_se.png")
    plt.close()

    young_neighbours = _get_nearest_neighbours(
        embeddings, young_centroid, n_neighbours=20
    )
    old_neighbours = _get_nearest_neighbours(embeddings, old_centroid, n_neighbours=20)
    cluster_gpsd = [
        np.average(psd[young_neighbours], axis=0),
        np.average(psd[old_neighbours], axis=0),
    ]
    young_power_map = power.variance_from_spectra(f, cluster_gpsd[0], nnmf)
    old_power_map = power.variance_from_spectra(f, cluster_gpsd[1], nnmf)
    old_minus_young_power_map = old_power_map - young_power_map

    power.save(
        old_minus_young_power_map,
        mask_file=training_data.mask_file,
        parcellation_file=training_data.parcellation_file,
        subtract_mean=False,
        filename=f"{plot_dir}/old_vs_young_pow_.png",
        combined=True,
        titles=[f"Mode {i+1}" for i in range(model.config.n_states)],
        plot_kwargs={"views": ["lateral"]},
    )

    P_young = cluster_gpsd[0]
    P_old = cluster_gpsd[1]
    P_old_minus_young = P_old - P_young

    P_mean = np.mean(P_old_minus_young, axis=1)
    P_std = np.std(P_old_minus_young, axis=1)
    fig, axes = plt.subplots(1, P_old_minus_young.shape[0], figsize=(20, 5))
    for j in range(P_old_minus_young.shape[0]):
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


plot_results(embeddings, f, psd, nnmf)
